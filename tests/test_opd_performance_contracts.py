"""Executable performance invariants; no hardware throughput assertions."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from helper.opd_history import ContiguousRolloutHistory
from helper.opd_reflex import OPDReflex
from helper.tree_verification import PackedTree
from helper.step_metrics import StepMetricsWriter
from opd_fixtures import CountModel,load_rollout
from test_opd_reflex import state,seed
from test_rollout_history import ConcatHistoryReference

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def test_dynamic_extents_are_runtime_not_specialized():
    import helper.tree_kernels as tree
    import helper.opd_reflex_kernels as opd
    dynamic={'N','B','C','PAST','ROWS','OFFSET','CACHE','WIDTH','CAP','CAPACITY','CONTEXTS'}
    for module in (tree,opd):
        parsed=ast.parse(Path(module.__file__).read_text())
        for function in parsed.body:
            if not isinstance(function,ast.FunctionDef) or not function.name.startswith('_'):continue
            kernel=getattr(module,function.name)
            if not hasattr(kernel,'do_not_specialize'):continue
            for arg in function.args.args:
                # B in OPD is a tensor pointer, not batch size.
                if arg.arg in dynamic and not (module is opd and arg.arg in ('B','CONTEXTS')) and not (module is tree and arg.arg=='CONTEXTS' and function.name=='_trace_path'):
                    assert arg.annotation is None,(function.name,arg.arg)
                    assert arg.arg in kernel.do_not_specialize,(function.name,arg.arg)


def test_auto_interpolates_unseen_batch_and_uses_dense_without_profile(monkeypatch):
    monkeypatch.delenv('OPD_PROPOSAL_PROFILE',raising=False)
    monkeypatch.setenv('OPD_PROPOSAL_MODE','auto')
    s=OPDReflex();s.vocab=32768;s.logits_dtype=torch.bfloat16;s.host_active_count=32768
    assert s.selected_proposal_backend(37,7)=='dense'
    s.tuning={'thresholds':{'8,1,32768,8,torch.bfloat16':1024,'32,8,32768,8,torch.bfloat16':8192}}
    s._threshold_cache.clear()
    assert s.proposal_threshold(2,4)==1024
    assert 1024<s.proposal_threshold(37,3)<8192
    assert s.proposal_threshold(64,8)==8192


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual Triton CUDA dispatch')
def test_only_selected_proposal_backend_is_launched(monkeypatch):
    from helper import opd_reflex_kernels as kernels
    s,_,mapping=state('cuda:0',v=257)
    seed(s,torch.arange(257,device='cuda'),torch.randn(257,8,device='cuda'))
    raw=torch.randn(3,4,257,device='cuda');h=torch.randn(3,4,32,device='cuda')
    class Forbidden:
        def __getitem__(self,grid):raise AssertionError('unselected backend launched')
    sparse,dense=kernels._sparse_scores,kernels._dense_gemm
    s.proposal_mode='auto';s.host_active_count=257
    monkeypatch.setattr(kernels,'_sparse_scores',Forbidden())
    monkeypatch.setattr(kernels,'_dense_gemm',Forbidden())
    s.dense_implementation='fused'
    q,ids,_=s.propose(raw,h,16,mapping);ref=(q.clone(),ids.clone(),s.proposal_norm.clone())
    monkeypatch.setattr(kernels,'_dense_gemm',dense)
    s.dense_implementation='gemm'
    q,ids,_=s.propose(raw,h,16,mapping)
    assert torch.equal(ref[0],q) and torch.equal(ref[1],ids)
    s.host_active_count=0
    monkeypatch.setattr(kernels,'_sparse_scores',sparse)
    monkeypatch.setattr(kernels,'_dense_gemm',Forbidden())
    q,ids,_=s.propose(raw,h,16,mapping)
    assert torch.equal(ref[0],q) and torch.equal(ref[1],ids)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='compiled variant reuse')
def test_tree_mask_does_not_recompile_with_past_or_live_rows():
    from helper import tree_kernels as kernels
    if not hasattr(kernels._tree_mask,'cache'):
        pytest.skip('Triton compiler cache introspection differs; runtime signature audit still runs')
    def variants():return sum(len(cache) for cache in kernels._tree_mask.cache.values())
    before=None
    for b,rows,past in ((3,7,3),(1,8,17),(7,13,1025),(2,31,2051)):
        parents=torch.arange(-1,rows-1,device='cuda').expand(b,rows).contiguous()
        tree=PackedTree(parents,parents.clone(),parents.clone(),rows)
        mask=torch.empty(b,1,rows,past+rows,device='cuda')
        kernels.tree_mask(tree,past,mask)
        expected=torch.arange(past+rows,device='cuda')[None,:]<=past+torch.arange(rows,device='cuda')[:,None]
        assert torch.equal(mask[:,0]==0,expected.expand(b,-1,-1))
        if before is None:before=variants()
        assert variants()==before


@pytest.mark.parametrize('device',DEVICES)
def test_contiguous_history_finished_rows_do_not_pin_growth_buffers(device):
    initial=torch.arange(24,device=device).reshape(3,4,2)
    h=ContiguousRolloutHistory({'features':initial},max_length=5)
    old=weakref.ref(h.buffers['features'])
    h.mark_finished(1)
    h.append([0,2],{'features':torch.full((2,4,2),99,device=device)})
    gc.collect();assert old() is None
    results=h.finalize()
    assert torch.equal(results[1]['features'],initial[1])
    assert results[0]['features'].shape==(8,2)
    assert results[0]['features'].untyped_storage().nbytes()==(8+4+8)*2*8
    assert not h.buffers


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stream',[False,True])
def test_contiguous_rollout_matches_concat_and_one_packet_per_round(device,stream,monkeypatch):
    records=[];copies=[]
    original=torch.Tensor.cpu
    def audit(tensor,*args,**kwargs):
        import inspect
        if tensor.is_cuda and inspect.currentframe().f_back.f_code.co_name=='schedule':copies.append(tuple(tensor.shape))
        return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',audit)
    for history in (ConcatHistoryReference,ContiguousRolloutHistory):
        copies.clear();torch.manual_seed(411);model=CountModel(device)
        output=load_rollout(device,history)(model,torch.tensor([[4,5],[8,9]],device=device),
            torch.ones(2,2,device=device,dtype=torch.long),SimpleNamespace(eos_token_id=16),
            method='opd_reflex',do_sample=True,repeated_generate_nums=8,max_length=22,
            verification_capacity=80,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_update_stream=stream,return_all_draft_input=True)
        records.append((output,model.calls,model.draft_calls))
        if device!='cpu':
            # Path capacity=4, packet=capacity+4. Other transfers are prefill
            # setup, end-of-rollout padding/counters/returned token lists.
            assert len(copies)==output['batch_verification_rounds']
    for field in ('generated_token_ids','response_verification_rounds','response_accepted_length_sum'):
        assert records[0][0][field]==records[1][0][field]
    for field in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
        for old,new in zip(records[0][0][field],records[1][0][field]):assert torch.equal(old,new)
    assert records[0][1:]==records[1][1:]


def test_step_max_is_not_difference_of_cumulative_max(tmp_path):
    w=StepMetricsWriter(tmp_path/'metrics.jsonl',tmp_path/'timing.csv')
    for label,interval,cumulative in ((1,100,100),(1,50,100),(2,20,100)):
        w.submit(label,{'step_opd_active_rows_max':interval,'cumulative_opd_active_rows_max':cumulative},{})
    import json
    first=json.loads((tmp_path/'metrics.jsonl').read_text())
    second=w.flush()
    assert first['step_opd_active_rows_max']==100
    assert second['step_opd_active_rows_max']==20 and second['cumulative_opd_active_rows_max']==100


@pytest.mark.skipif(not torch.cuda.is_available(),reason='BF16 layout-dependent reduction regression')
@pytest.mark.parametrize('slots',[16,64,32768])
def test_large_vocab_bf16_normalizer_is_bitwise_independent_of_backend(slots):
    torch.manual_seed(42)
    s,_,mapping=state('cuda:0',v=32768,h=32)
    seed(s,torch.randperm(32768,device='cuda')[:slots],torch.randn(slots,8,device='cuda')*.05)
    raw=torch.randn(3,4,32768,device='cuda',dtype=torch.bfloat16)
    h=torch.randn(3,4,32,device='cuda',dtype=torch.bfloat16)
    results=[]
    for mode,implementation in [('sparse','fused'),('dense','fused'),('dense','gemm')]:
        s.proposal_mode=mode;s.dense_implementation=implementation
        q,ids,_=s.propose(raw,h,16,mapping)
        results.append((q.clone(),ids.clone(),s.proposal_norm[:24].clone()))
    for result in results[1:]:
        for a,b in zip(results[0],result):assert torch.equal(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU selected-state teacher extraction')
def test_teacher_scans_only_compacted_states_and_preserves_unselected_tiles():
    from helper import opd_reflex_kernels as kernels
    b,q,v,k=3,7,521,16;n=b*q;tiles=(v+255)//256
    mapping=torch.arange(v,device='cuda');target=torch.rand(b,q,v,device='cuda')
    weights=torch.zeros(b,q,device='cuda');weights[0,0]=1;weights[2,3]=1
    pools=[torch.full((n*tiles*(k if i>=2 else 1),),-123,device='cuda',dtype=torch.long if i==3 else torch.float32) for i in range(4)]
    p=torch.empty(n,k,device='cuda');ids=torch.empty(n,k,device='cuda',dtype=torch.long);mass=torch.empty(n,device='cuda')
    selected=torch.empty(n,device='cuda',dtype=torch.int32);count=torch.empty(1,device='cuda',dtype=torch.int32)
    kernels.teacher(target,mapping,weights,k,pools,(p,ids,mass),selection=(selected,count))
    assert count.item()==2 and selected[:2].tolist()==[0,17]
    inactive=weights.flatten()==0
    assert (p[inactive]==0).all() and (ids[inactive]==-1).all() and (mass[inactive]==0).all()
    assert (pools[1].view(n,tiles)[inactive]==-123).all()
    for row in (0,17):
        reference=target.flatten(0,1)[row];reference=reference/reference.sum()
        torch.testing.assert_close(p[row],reference[ids[row]],rtol=1e-6,atol=1e-7)
