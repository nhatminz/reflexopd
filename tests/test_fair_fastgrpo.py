"""Persistent upstream loss and shared generic infrastructure parity contracts."""
import ast
import csv
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from helper.eagle3_online_objective import persistent_loss
from helper.rollout_metrics import RolloutMetricsWriter
from helper.shared_rollout import FastGRPORuntime
from opd_fixtures import CountModel,load_rollout,load_historical

ROOT=Path(__file__).resolve().parents[1]
DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


class ObjectiveFixture(torch.nn.Module):
    dtype=torch.float32
    def __init__(self):
        super().__init__();self.prediction=torch.nn.Parameter(torch.randn(2,6,4))
        self.head=torch.nn.Linear(4,3);self.mapping=torch.tensor([6,0,3]);self.calls=0
    def forward(self,**kwargs):
        self.calls+=1
        return dict(next_feature_states=self.prediction,hidden_states=self.prediction)
    def compute_compact_logits(self,h):return self.head(h)
    def compact_to_target_ids(self,device=None):return self.mapping.to(device)


def test_loss_formula_masking_compact_support_and_gradients_match_direct_upstream_reference():
    torch.manual_seed(31);model=ObjectiveFixture();teacher_head=torch.nn.Linear(4,7)
    targets=torch.randn(2,6,4,requires_grad=True)
    ids=torch.zeros(2,6,dtype=torch.long);inputs=torch.randn(2,6,12)
    attention=torch.tensor([[1,1,1,1,1,1],[1,1,1,1,0,0]])
    loss_mask=torch.tensor([[0,0,0,1,1,1],[0,0,1,1,0,0]])
    a,b=persistent_loss(model,ids,inputs,targets,attention,loss_mask,teacher_head,chunk_size=2)
    valid=loss_mask[:,:-1].bool()&attention[:,:-1].bool()&attention[:,1:].bool()
    denom=valid.sum(1).clamp_min(1)
    ref_feature=2*(F.smooth_l1_loss(model.prediction[:,:-1],targets[:,1:].detach(),reduction='none').mean(-1)*valid).sum(1)/denom
    with torch.no_grad():
        p=teacher_head(targets[:,1:]).softmax(-1)[...,model.mapping]
        p=p/p.sum(-1,keepdim=True)
    ref_distribution=.1*(-(p*model.head(model.prediction[:,:-1]).log_softmax(-1)).sum(-1)*valid).sum(1)/denom
    torch.testing.assert_close(a,ref_feature,rtol=0,atol=0)
    torch.testing.assert_close(b,ref_distribution,rtol=1e-6,atol=1e-7)
    actual=torch.autograd.grad((a+b).mean(),(model.prediction,model.head.weight),retain_graph=True)
    reference=torch.autograd.grad((ref_feature+ref_distribution).mean(),(model.prediction,model.head.weight))
    for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
    assert model.calls==1 and targets.grad is None and teacher_head.weight.grad is None
    assert not actual[0][0,:3].any() and not actual[0][1,3:].any()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('sample',[False,True])
def test_shared_fastgrpo_matches_historical_tokens_rng_histories_and_no_opd(device,sample,monkeypatch):
    from helper.opd_reflex import OPDReflex
    monkeypatch.setattr(OPDReflex,'__init__',lambda *a,**k:pytest.fail('FastGRPO created OPD engine'))
    result=[]
    for previous in (True,False):
        torch.manual_seed(411);m=CountModel(device);m.supports_opd_static_kv=True
        fn=load_historical(device) if previous else load_rollout(device)
        output=fn(m,torch.tensor([[0,0,4,5],[3,7,8,9]],device=device),
            torch.tensor([[0,0,1,1],[1,1,1,1]],device=device),SimpleNamespace(eos_token_id=16),
            do_sample=sample,repeated_generate_nums=8,max_length=28,verification_capacity=80,
            max_draft_k=3,max_draft_token_length=3,min_draft_token_length=2,max_verification_num=16,
            return_all_draft_input=True)
        result.append((output,m,torch.rand(7,device=device)))
    before,after=result[0][0],result[1][0]
    for key in ('generated_token_ids','total_acc_length','total_decoded_token_num','response_verification_rounds'):
        assert before[key]==after[key]
    for key in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
        for a,b in zip(before[key],after[key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(result[0][2],result[1][2])
    assert result[0][1].calls==result[1][1].calls and result[0][1].draft_calls==result[1][1].draft_calls
    runtime=result[1][1]._fastgrpo_runtime
    assert isinstance(runtime,FastGRPORuntime)
    for forbidden in ('B_fast','projector','teacher_p','u_cache','profile_selector'):
        assert not hasattr(runtime,forbidden)
    for side in ('target','draft'):
        assert getattr(result[1][1],'_opd_'+side+'_kv_pool').get_seq_length()==0
    assert after['opd_backend']=='off'


def test_fast_runtime_native_topk_preserves_exact_ties():
    raw=torch.zeros(3,2,37);mapping=torch.randperm(37)
    r=FastGRPORuntime()
    for k in (1,3,8,16):
        actual=r.propose(raw,None,k,mapping)
        q,ids=torch.topk(raw.float().softmax(-1),k,dim=-1)
        assert torch.equal(actual[0],q) and torch.equal(actual[1],ids) and torch.equal(actual[2],mapping[ids])


def test_per_iteration_loss_logging_same_schema_for_both_methods(tmp_path):
    for method in ('fastgrpo','opd_reflex'):
        path=tmp_path/method/'rollout.csv';w=RolloutMetricsWriter(path,method)
        w.begin(1,0,8,0)
        w.finish(dict(iter_draft_feature_loss=2.,iter_draft_distribution_loss=.3,iter_draft_total_loss=2.3),
                 grpo_step=0,used_items=8,wall_time_s=1)
        w.close()
        row=next(csv.DictReader(path.open()))
        assert [float(row['iter_draft_'+name+'_loss']) for name in ('feature','distribution','total')]==[2.,.3,2.3]


def test_parser_defaults_are_upstream_regime():
    tree=ast.parse((ROOT/'grpo_speculative.py').read_text())
    defaults={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='parser.add_argument' and n.args and isinstance(n.args[0],ast.Constant):
            for kw in n.keywords:
                if kw.arg=='default' and isinstance(kw.value,ast.Constant):defaults[n.args[0].value]=kw.value.value
    assert defaults['--target_lr']==1e-6 and defaults['--draft_lr']==1e-4
    assert defaults['--draft_accumulation_steps']==1
