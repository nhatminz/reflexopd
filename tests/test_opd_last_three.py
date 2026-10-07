"""Parity against the frozen previous rollout and full-probability feedback."""
import ast
import weakref
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_static_cache import OPDStaticCache,swap_remove_plan,persistent_cache
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_reflex import OPDReflex
from helper.tree_verification import PackedTree
from opd_fixtures import CountModel,load_rollout
from test_opd_reflex import state,seed

ROOT=Path(__file__).resolve().parents[1]
DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('finished',[[0,0,0,1],[1,0,0,0],[0,1,1,0,0,1,0],[1,0,1,0,1,0],[1,1,1]])
def test_swap_remove_only_moves_disjoint_live_tail_rows(device,finished):
    b=len(finished);k=torch.randn(b,3,517,7,device=device)
    c=OPDStaticCache(256,batch_capacity=b);c.update(k,k+1,0)
    keep,sources,destinations=swap_remove_plan(finished)
    old=c.statistics();ptr=c.layers[0].key_pool.data_ptr()
    c.swap_remove(len(keep),torch.tensor(sources,device=device,dtype=torch.long),torch.tensor(destinations,device=device,dtype=torch.long))
    expected=k.index_select(0,torch.tensor(keep,device=device,dtype=torch.long))
    assert torch.equal(c[0][0],expected) and torch.equal(c[0][1],expected+1)
    assert c.layers[0].key_pool.data_ptr()==ptr
    stats=c.statistics();copied=len(sources)*3*517*7*4*2
    assert stats['kv_rows_moved']==len(sources)
    assert stats['kv_history_copy_bytes']-old['kv_history_copy_bytes']==copied
    assert set(sources).isdisjoint(destinations)
    assert all(source>=len(keep) for source in sources)
    if finished==[0,0,0,1]:assert copied==0


@pytest.mark.parametrize('device',DEVICES)
def test_persistent_pools_reuse_high_water_mark_without_stale_history(device):
    model=SimpleNamespace();pointers=[]
    for iteration,(b,length) in enumerate(((2,601),(1,17),(2,300),(2,550))):
        c=persistent_cache(model,'_opd_target_kv_pool',b,4,device,torch.float32)
        assert c.get_seq_length()==0 and not c
        k=torch.full((b,2,length,7),float(iteration+1),device=device)
        c.update(k,k+10,0);c.batch_repeat_interleave(4)
        assert torch.equal(c[0][0],k.repeat_interleave(4,0))
        pointers.append(c.layers[0].key_pool.data_ptr())
        if iteration:assert c.statistics()['full_kv_reallocations']==0
        c.end_rollout();assert c.get_seq_length()==0 and c[0][0].shape[0]==0
    assert len(set(pointers))==1
    c.end_rollout(max_retained_tokens=256)
    assert not c.layers and not c._scratch


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('greedy',[False,True])
@pytest.mark.parametrize('subset',[False,True])
def test_compact_teacher_kl_grad_b_grad_a_match_full_probability_reference(device,greedy,subset):
    # Duplicate freshly seeded states so gradients/counters/update begin identical.
    records=[]
    torch.manual_seed(4)
    raw=torch.randn(3,1,37,device=device);h=torch.randn(3,1,32,device=device)
    logits=torch.randn(3,1,43 if subset else 37,device=device)
    for compact in (False,True):
        torch.manual_seed(21)
        _,model,_=state(device,v=37)
        s=OPDReflex(8,16,train_projector=True)
        mapping=torch.arange(37,device=device)+(3 if subset else 0)
        if not subset:
            model.opd_full_vocab_inverse=torch.arange(37,device=device)
        s.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
        seed(s,torch.arange(10,device=device),torch.ones(10,8,device=device)*.01)
        s.propose(raw,h,8,mapping,root=True)
        root=torch.full((3,1),-1,device=device,dtype=torch.long)
        tree=PackedTree(root,root.clone(),torch.zeros_like(root),0)
        path=SimpleNamespace(packed_indices=torch.zeros_like(root))
        kwargs=dict(do_sample=not greedy,temperature=.8,top_p=.95,top_k=0,eos_token_id=2)
        torch.manual_seed(53)
        if compact:
            refs=[]
            def builder(tokens,p,sorted_meta):
                if p is not None:refs.append(weakref.ref(p))
                return s.prepare_compact_teacher(tree,path,tokens if greedy else p,sorted_meta,greedy)
            tokens,p,metadata=sample_target_with_metadata(logits,**kwargs,metadata_builder=builder,return_probs=False)
            assert p is None and all(ref() is None for ref in refs)
            assert len(metadata)==4 and sum(x.numel() for x in metadata)==3*(3*16+1)
            s.feedback(tree,path,None,greedy=greedy,sampling_metadata=metadata)
        else:
            tokens,p,_=sample_target_with_metadata(logits,**kwargs)
            s.feedback(tree,path,tokens if greedy else p,greedy=greedy)
        records.append((tokens.clone(),torch.random.get_rng_state(),s.B_fast.clone(),model.opd_projector_grad_sum.clone(),s.finish()))
    for before,after in zip(records[0][:4],records[1][:4]):
        torch.testing.assert_close(before,after,rtol=2e-5,atol=1e-7)
    for name in ('opd_kl_sum','opd_state_weight','opd_draft_topk_target_mass_sum','opd_compact_mass_sum'):
        assert records[0][4][name]==pytest.approx(records[1][4][name],rel=2e-5,abs=1e-6)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stream',[False,True])
def test_complete_rollout_matches_old_stable_row_order_rng_histories_and_forward_counts(device,stream):
    results=[]
    for old in (True,False):
        torch.manual_seed(121);model=CountModel(device);model.supports_opd_static_kv=True
        generate=load_rollout(device,source_path=ROOT/'tests/oracles/opd_rollout_before_memory.py' if old else None)
        torch.manual_seed(411)
        output=generate(model,torch.tensor([[4,5],[8,9]],device=device),torch.ones(2,2,dtype=torch.long,device=device),
            SimpleNamespace(eos_token_id=16),method='opd_reflex',do_sample=True,repeated_generate_nums=8,
            max_length=28,verification_capacity=80,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_update_stream=stream,return_all_draft_input=True)
        results.append((output,model.calls,model.draft_calls,torch.rand(5,device=device)))
    before,after=results[0][0],results[1][0]
    for name in ('generated_token_ids','response_verification_rounds','response_accepted_length_sum','total_acc_length','total_decoded_token_num'):
        assert before[name]==after[name]
    for name in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
        for a,b in zip(before[name],after[name]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert results[0][1:3]==results[1][1:3]
    assert torch.equal(results[0][3],results[1][3])
    if device=='cuda':assert after['opd_host_syncs_per_round']==1
    assert after['opd_target_kv_history_copy_bytes']<before['opd_target_kv_history_copy_bytes']


@pytest.mark.parametrize('device',DEVICES)
def test_rollout_pool_reuse_across_iterations_is_exact_and_zero_reallocation(device):
    torch.manual_seed(121);model=CountModel(device);model.supports_opd_static_kv=True
    generate=load_rollout(device);outputs=[]
    for _ in range(3):
        torch.manual_seed(411)
        out=generate(model,torch.tensor([[4,5],[8,9]],device=device),torch.ones(2,2,dtype=torch.long,device=device),
            SimpleNamespace(eos_token_id=16),method='opd_reflex',do_sample=True,repeated_generate_nums=8,
            max_length=28,verification_capacity=80,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_update_stream=True,return_all_draft_input=True)
        outputs.append(out)
        for side in ('target','draft'):
            assert out['opd_'+side+'_full_kv_reallocations']==0
            assert getattr(model,'_opd_'+side+'_kv_pool').get_seq_length()==0
    assert all(out['generated_token_ids']==outputs[0]['generated_token_ids'] for out in outputs)


def test_async_feedback_receives_no_full_probability_tensor_and_adds_no_host_sync_calls():
    source=(ROOT/'helper/specualtive_generate.py').read_text();tree=ast.parse(source)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='token_num')
    assert 'return_probs=False' in ast.unparse(loop)
    assert 'teacher = None' in ast.unparse(loop)
    for module in ('helper/opd_reflex_kernels.py','helper/opd_kv_kernels.py'):
        text=(ROOT/module).read_text()
        for forbidden in ('.cpu(','.tolist(','.item(','.synchronize('):assert forbidden not in text
