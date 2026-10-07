import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_reflex import OPD_COUNTER_NAMES,initialize_projector,union_reference
from helper.step_metrics import StepMetricsWriter,step_record
from helper.eagle3_specforge import Eagle3FastGRPOAdapter
from scripts.benchmark_opd_reflex import summarize,parse_args
from test_opd_reflex import state
from opd_fixtures import CountModel,load_rollout


def test_union_coordinate_gradient_is_forward_kl_including_tail():
    torch.manual_seed(47)
    z=torch.randn(3,31,requires_grad=True);q=z.softmax(-1);p=torch.randn(3,31).softmax(-1)
    ti=p.topk(4).indices;di=q.topk(4).indices
    ids,valid,pu,qu,pt,qt,kl=union_reference(p,q,ti,di)
    gradient=torch.autograd.grad(kl.sum(),z)[0]
    torch.testing.assert_close(gradient.gather(-1,ids)[valid],(qu-pu)[valid],rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('norm_output',[False,True])
def test_actual_adapter_selected_head_input_normalization_matches_specforge(norm_output):
    draft=SimpleNamespace(norm_output=norm_output,norm=torch.nn.LayerNorm(8),lm_head=torch.nn.Linear(8,19))
    model=SimpleNamespace(draft_model=draft)
    h=torch.randn(2,3,8)
    raw,inputs=Eagle3FastGRPOAdapter.compute_compact_logits_with_inputs(model,h)
    expected=h if norm_output else draft.norm(h)
    assert torch.equal(inputs,expected)
    assert torch.equal(raw,draft.lm_head(expected))
    ids=torch.tensor([0,5,9,17])
    selected=torch.nn.functional.linear(inputs,draft.lm_head.weight[ids],draft.lm_head.bias[ids])
    torch.testing.assert_close(selected,raw[...,ids],rtol=1e-6,atol=1e-6)


def test_projector_persists_in_real_draft_checkpoint_api_without_touching_rng(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.draft_model=torch.nn.Linear(8,17)
            self.draft_model.register_parameter('opd_projector',torch.nn.Parameter(initialize_projector(8,8,head=self.draft_model.weight)))
        @property
        def opd_projector(self):return self.draft_model.opd_projector
        def checkpoint_metadata(self):return {'opd_rank':8}
        def load_opd_projector(self,x):return Eagle3FastGRPOAdapter.load_opd_projector(self,x)
    model=Model();saved=model.opd_projector.clone();path=tmp_path/'draft.pth'
    Eagle3FastGRPOAdapter.save_model(model,path)
    payload=torch.load(path,weights_only=True)
    assert torch.equal(payload['opd_projector'],saved)
    with torch.no_grad():model.opd_projector.zero_()
    Eagle3FastGRPOAdapter.load_model(model,path)
    assert torch.equal(model.opd_projector,saved) and model.opd_projector.requires_grad
    rng=torch.random.get_rng_state();initialize_projector(8,8)
    assert torch.equal(rng,torch.random.get_rng_state())


def snap(**kwargs):
    result={f'cumulative_{k}':0. for k in (
        'wall_time_s','generation_time_s','target_train_time_s','draft_train_time_s',
        'rollout_tokens','accepted_tokens','verification_rounds','accepted_draft_tokens','proposed_draft_tokens',
        *OPD_COUNTER_NAMES,'verification_batches','active_response_rounds','verified_tree_nodes')}
    result.update({f'cumulative_{k}':v for k,v in kwargs.items()});return result


def test_step_opd_metrics_are_exact_counter_differences_and_both_methods_share_csv_schema(tmp_path):
    a=snap(accepted_tokens=10,verification_rounds=5,opd_kl_sum=8,opd_state_weight=4,opd_selected_states=3)
    b=snap(accepted_tokens=16,verification_rounds=7,opd_kl_sum=11,opd_state_weight=6,opd_selected_states=5,
           opd_union_size_sum=40,opd_compact_mass_sum=1.6,opd_frontier_states=2,opd_visited_states=3,
           active_response_rounds=7,verification_batches=2,verified_tree_nodes=21)
    row=step_record(1,b,a)
    assert row['step_aal']==3 and row['step_opd_kl']==1.5
    assert row['step_opd_selected_states']==2 and row['step_opd_mean_union_size']==20
    assert row['step_mean_active_responses_per_verify_round']==3.5
    schemas=[]
    for method in ('fastgrpo','opd_reflex'):
        root=tmp_path/method;writer=StepMetricsWriter(root/'metrics.jsonl',root/'timing.csv')
        writer.submit(0,a,{'phase':'target_train','method':method});writer.submit(1,b,{'phase':'target_train','method':method});writer.flush()
        csv_rows=list(csv.DictReader((root/'timing.csv').open()))
        assert len(csv_rows)==2 and csv_rows[1]['step_aal']=='3.0'
        schemas.append(set(csv_rows[0]))
    assert schemas[0]==schemas[1]
    broken=dict(b,cumulative_opd_nonfinite_kl_states=1)
    assert step_record(1,broken,a)['step_opd_kl'] is None


def test_benchmark_aal_uses_weighted_rounds_and_kl_uses_state_weights():
    base=dict(peak_allocated_bytes=1,peak_reserved_bytes=1,accepted_draft_tokens=1,proposed_draft_tokens=2,
              opd_kl_sum=2.,opd_state_weight=1.,opd_selected_states=1.)
    rows=[dict(base,generated_tokens=10,generation_wall_s=1.,accepted_length_sum=8,verification_rounds=2),
          dict(base,generated_tokens=20,generation_wall_s=3.,accepted_length_sum=10,verification_rounds=10)]
    result=summarize(rows)
    assert result['aal']==1.5 and result['tokens_per_s']==7.5 and result['opd_kl']==2.


def test_benchmark_copy_sync_and_replay_timing_do_not_fake_baseline_measurements():
    common=dict(generated_tokens=5,generation_wall_s=1.,accepted_length_sum=3,
        verification_rounds=2,accepted_draft_tokens=1,proposed_draft_tokens=4,
        peak_allocated_bytes=100,peak_reserved_bytes=120,batch_verification_rounds=2,
        observed_kv_cache_bytes=80)
    baseline=summarize([dict(common,method='fastgrpo')])
    assert baseline['host_syncs_per_round'] is None
    assert baseline['full_kv_reallocations'] is None
    assert baseline['kv_compaction_profile_ms'] is None
    assert baseline['kv_cache_bytes']==80
    row=dict(common,method='opd_reflex',opd_host_syncs=2,
             opd_target_full_kv_reallocations=1,opd_draft_full_kv_reallocations=2,
             kv_compaction_profile_ms=.5)
    measured=summarize([row,row])
    assert measured['host_syncs_per_round']==1
    assert measured['full_kv_reallocations']==6
    assert measured['kv_compaction_profile_ms']==1


def test_no_host_sync_in_production_feedback_and_no_dense_vocab_correction_or_target_forward():
    root=Path(__file__).resolve().parents[1]/'helper'
    kernels=(root/'opd_reflex_kernels.py').read_text()
    for forbidden in ('.cpu(','.item(','.tolist(','.numpy(','.synchronize(','.softmax(','.backward(','.step('):
        assert forbidden not in kernels
    tree=ast.parse(kernels)
    scan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_corrected_scan')
    # Dense branch fuses correction with scan; sparse branch only reads scores.
    dense=next(n for n in ast.walk(scan) if isinstance(n,ast.If) and ast.unparse(n.test)=='DENSE')
    sparse_code='\n'.join(ast.unparse(n) for n in dense.orelse)
    assert 'tl.load(B +' not in sparse_code and 'tl.load(U +' not in sparse_code
    active=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sparse_scores')
    assert 'range(0, count, BS)' in ast.unparse(active)
    assert 'probability_pool' not in (root/'opd_reflex.py').read_text()
    assert 'all_reduce' not in kernels


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA event/runtime cache contract')
def test_two_events_per_rollout_and_no_default_diagnostics_or_profiling_sync(monkeypatch):
    original=torch.cuda.Event;calls=[]
    def event(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(torch.cuda,'Event',event)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:pytest.fail('production synchronization'))
    model=CountModel('cuda:0');pointers=[]
    for _ in range(2):
        torch.manual_seed(411)
        out=load_rollout('cuda:0')(model,torch.tensor([[4,5],[8,9]],device='cuda'),torch.ones(2,2,device='cuda',dtype=torch.long),
            SimpleNamespace(eos_token_id=16),method='opd_reflex',do_sample=True,max_length=18,
            verification_capacity=48,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_update_stream=True,opd_profile=False,opd_diagnostics=False)
        runtime=next(iter(model._opd_runtime_cache.values()));pointers.append(runtime.B_fast.data_ptr())
        assert out['batch_verification_rounds']>1 and 'opd_final_b_norm' not in out
    assert len(calls)==4 and pointers[0]==pointers[1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA profiling and stream smoke')
def test_profile_sections_present_only_when_requested_and_do_not_change_zero_state():
    results=[]
    for profile in (False,True):
        torch.manual_seed(411);model=CountModel('cuda:0')
        out=load_rollout('cuda:0')(model,torch.tensor([[4,5],[8,9]],device='cuda'),torch.ones(2,2,device='cuda',dtype=torch.long),
            SimpleNamespace(eos_token_id=16),method='opd_reflex',do_sample=True,max_length=18,
            verification_capacity=48,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_fast_lr=0.,opd_update_stream=True,opd_profile=profile)
        results.append(out)
    assert results[0]['generated_token_ids']==results[1]['generated_token_ids']
    assert results[0]['opd_profile_sections_ms'] is None
    assert {'opd_feature_ms','proposal_ms','opd_state_select_ms','opd_teacher_extract_ms','opd_union_loss_ms',
            'opd_update_ms','opd_wait_ms'}<=set(results[1]['opd_profile_sections_ms'])
