"""Independent cache/mask/sampler oracles and observable copy/sync contracts."""
import ast
import csv
import weakref
from pathlib import Path
import pytest
import torch
from helper.opd_static_cache import OPDStaticCache
from helper.opd_attention import AttentionWorkspace
from helper.opd_sampling import sample_target_with_metadata
from helper.sampling import sample_target_from_logits
from helper.rollout_metrics import FIELDS,KV_FIELDS,RolloutMetricsWriter

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def test_cache_does_not_retain_pools_in_owner_cycle():
    cache=OPDStaticCache(8);x=torch.zeros(2,3,4,5);cache.update(x,x,0)
    cache_ref=weakref.ref(cache);pool_ref=weakref.ref(cache.layers[0].key_pool)
    del cache
    # No explicit gc.collect: rollout-end must release immediately.
    assert cache_ref() is None and pool_ref() is None


def test_old_iteration_csv_resume_migrates_header_and_logs_exact_kv_counters(tmp_path):
    path=tmp_path/'rollout_timing.csv'
    fields=[name for name in FIELDS if name not in KV_FIELDS]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerow(dict(global_iter=1,method='opd_reflex',grpo_step=0))
    state=dict(global_iter=1,accepted=4,rounds=2,tokens=5,generation=1,accepted_draft=2,proposed=4)
    writer=RolloutMetricsWriter(path,'opd_reflex',state=state)
    writer.begin(1,1,8,0)
    writer.finish(dict(total_acc_length=9,total_decoded_token_num=3,
        opd_host_syncs=2,opd_host_syncs_per_round=1,
        opd_target_kv_cache_bytes=100,opd_draft_kv_cache_bytes=30,
        opd_target_full_kv_reallocations=1,opd_draft_full_kv_reallocations=2,
        opd_target_full_history_copies=4,opd_draft_full_history_copies=5,
        opd_target_kv_rows_moved=3,opd_draft_kv_rows_moved=3,
        opd_target_full_history_copy_bytes=80,opd_draft_full_history_copy_bytes=90),
        grpo_step=1,used_items=8,wall_time_s=2)
    writer.close()
    assert path.with_suffix('.pre_resume.csv').is_file()
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert [row['global_iter'] for row in rows]==['1','2']
    assert list(rows[1])==list(FIELDS) and rows[0]['iter_host_syncs']==''
    assert float(rows[1]['iter_aal'])==3
    for field,value in zip(KV_FIELDS,[2,1,130,3,9,170,3,0]):
        assert float(rows[1][field])==value


@pytest.mark.parametrize('device',DEVICES)
def test_geometric_growth_and_no_pool_replacement_on_repeat_select(device):
    c=OPDStaticCache(8,batch_capacity=8,chunk_size=8)
    full=torch.randn(2,3,39,5,device=device);values=full+1
    for start,end in ((0,5),(5,8),(8,9),(9,17),(17,39)):
        k,v=c.update(full[...,start:end,:],values[...,start:end,:],0)
        assert torch.equal(k,full[...,:end,:]) and torch.equal(v,values[...,:end,:])
        assert c.layers[0].capacity<=2*max(8,end)
    stats=c.statistics();assert stats['full_kv_reallocations']==3
    assert stats['full_history_copies']==3
    ptr=c[0][0].untyped_storage().data_ptr()
    c.batch_repeat_interleave(4)
    oracle=full.repeat_interleave(4,0);v_oracle=values.repeat_interleave(4,0)
    assert torch.equal(c[0][0],oracle)
    for ids in ([7,0,4,2,6],[4,0,0,3],[3,1]):
        ids=torch.tensor(ids,device=device)
        oracle=oracle.index_select(0,ids);v_oracle=v_oracle.index_select(0,ids)
        c.batch_select_indices(ids)
        assert torch.equal(c[0][0],oracle) and torch.equal(c[0][1],v_oracle)
        assert c[0][0].untyped_storage().data_ptr()==ptr
        assert c.statistics()['full_kv_reallocations']==stats['full_kv_reallocations']
    c.crop(13);suffix=torch.randn(2,3,6,5,device=device)
    c.update(suffix,suffix+1,0)
    assert torch.equal(c[0][0],torch.cat((oracle[...,:13,:],suffix),-2))
    assert c[0][0].untyped_storage().data_ptr()==ptr


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA alias/race stress')
@pytest.mark.parametrize('batch',[16,32,64])
def test_compaction_race_free_arbitrary_permutation_and_duplicates(batch):
    c=OPDStaticCache(256,batch_capacity=batch)
    k=torch.randn(batch,3,517,37,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k);c.update(k,v,0)
    ptr=c.layers[0].key_pool.data_ptr()
    for _ in range(5):
        ids=torch.randperm(batch,device='cuda');ids[-4:]=ids[:4]
        k=k.index_select(0,ids);v=v.index_select(0,ids)
        c.batch_select_indices(ids)
        assert torch.equal(c[0][0],k) and torch.equal(c[0][1],v)
        assert c.layers[0].key_pool.data_ptr()==ptr
    assert c.statistics()['full_kv_reallocations']==0
    assert c.statistics()['kv_compaction_workspace_bytes']<=2*c.statistics()['kv_cache_bytes']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_reused_attention_mask_matches_original_and_positions_own_storage(device,dtype):
    w=AttentionWorkspace(device)
    for past,b,q in ((1,3,6),(7,3,4),(1,2,3),(257,2,6),(17,1,2)):
        pad=torch.zeros(b,past+q,device=device,dtype=torch.bool)
        pad[:,1:3]=True
        expected=torch.triu(torch.full((q,past+q),torch.finfo(dtype).min,device=device,dtype=dtype),diagonal=past+1)
        expected=expected[None,None].repeat(b,1,1,1).masked_fill(pad[:,None,None],torch.finfo(dtype).min)
        actual=w.causal('mask',past,q,b,dtype,pad)
        assert torch.equal(actual,expected)
        assert torch.equal(w.positions('position',past,q),torch.arange(past,past+q,device=device))
    allocations=w.allocations
    for _ in range(4):w.causal('mask',17,2,1,dtype,pad)
    assert w.allocations==allocations


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_sampler_bitwise_and_sorted_arrays_freed_after_compaction(device,dtype):
    x=torch.randn(2,3,257,device=device,dtype=dtype);x[0,0].zero_();x[1,0].fill_(float('nan'))
    kwargs=dict(do_sample=True,temperature=.7,top_p=.95,top_k=0,eos_token_id=2)
    refs=[]
    def builder(tokens,p,sort):
        for t in sort:refs.append(weakref.ref(t))
        values,indices=sort
        # Small owned tensors for this memory-lifetime test; production uses
        # its selected-state exact-tie kernel and independent persistent buffers.
        return values[:,:16].clone(),indices[:,:16].clone()
    torch.manual_seed(31);old=sample_target_from_logits(x,**kwargs);after=torch.rand(5,device=device)
    torch.manual_seed(31);new=sample_target_with_metadata(x,**kwargs,metadata_builder=builder)
    assert torch.equal(after,torch.rand(5,device=device))
    assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1])
    assert all(ref() is None for ref in refs)
    assert all(t.shape==(6,16) and t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in new[2])


def test_hot_loop_never_concatenates_kv_history():
    root=Path(__file__).parents[1]
    for path in ('helper/opd_static_cache.py','helper/opd_kv_kernels.py','helper/opd_scheduling.py'):
        tree=ast.parse((root/path).read_text())
        assert not any(isinstance(n,ast.Call) and ast.unparse(n.func) in ('torch.cat','torch.concat') for n in ast.walk(tree))
    tree=ast.parse((root/'helper/specualtive_generate.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='token_num')
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func) in ('torch.cat','torch.concat') for n in ast.walk(loop))


def test_full_probability_alias_does_not_survive_until_next_verification_round():
    root=Path(__file__).parents[1]
    tree=ast.parse((root/'helper/specualtive_generate.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='token_num')
    enabled=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='enabled')
    assert isinstance(enabled.body[-1],ast.Delete)
    assert ast.unparse(enabled.body[-1])=='del teacher'
    deleted={ast.unparse(target) for n in ast.walk(loop) if isinstance(n,ast.Delete) for target in n.targets}
    assert {'teacher','target_sampling_probs','target_outputs_logits','sampling_metadata'}<=deleted


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Triton runtime extent cache')
def test_new_kernels_keep_lengths_and_batches_runtime():
    from helper import opd_attention_kernels,opd_kv_kernels
    for module in (opd_attention_kernels,opd_kv_kernels):
        source=ast.parse(Path(module.__file__).read_text())
        for fn in source.body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('_'):continue
            kernel=getattr(module,fn.name)
            for arg in fn.args.args:
                if arg.arg in ('B','PAST','ROWS','LENGTH','CAPACITY','N'):
                    assert arg.annotation is None
                    assert arg.arg in kernel.do_not_specialize


@pytest.mark.skipif(not torch.cuda.is_available(),reason='production sync counter')
def test_rollout_instrumentation_is_one_packet_per_round_without_finish_reallocation():
    from types import SimpleNamespace
    from opd_fixtures import CountModel,load_rollout
    model=CountModel('cuda');model.supports_opd_static_kv=True
    torch.manual_seed(411)
    out=load_rollout('cuda')(model,torch.tensor([[4,5],[8,9]],device='cuda'),
        torch.ones(2,2,dtype=torch.long,device='cuda'),SimpleNamespace(eos_token_id=16),
        method='opd_reflex',do_sample=True,repeated_generate_nums=8,max_length=22,
        verification_capacity=80,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
        min_draft_token_length=2,opd_update_stream=True)
    assert out['opd_host_syncs']==out['batch_verification_rounds']
    assert out['opd_host_syncs_per_round']==1
    for side in ('target','draft'):
        assert out['opd_'+side+'_full_kv_reallocations']==0
        assert out['opd_'+side+'_row_compactions']>1  # repeat + finished rows
