import csv
import copy
from types import SimpleNamespace
import pytest
import torch
from helper.opd_sampling import sample_target_with_metadata
from helper.sampling import sample_target_from_logits as sample_target
from helper.opd_static_cache import OPDStaticCache
from helper.rollout_metrics import RolloutMetricsWriter
from helper.opd_optimizer import draft_optimizer, load_draft_optimizer
from helper.tree_verification import PackedTree
from test_opd_reflex import state

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stream',[False,True])
def test_static_rollout_matches_dynamic_histories_rng_forward_counts(device,stream):
    from opd_fixtures import CountModel,load_rollout
    records=[]
    for static in (False,True):
        torch.manual_seed(411);model=CountModel(device);model.supports_opd_static_kv=static
        out=load_rollout(device)(model,torch.tensor([[4,5],[8,9]],device=device),
            torch.ones(2,2,device=device,dtype=torch.long),SimpleNamespace(eos_token_id=16),
            method='opd_reflex',do_sample=True,repeated_generate_nums=8,max_length=22,
            verification_capacity=80,max_verification_num=16,max_draft_k=3,max_draft_token_length=3,
            min_draft_token_length=2,opd_update_stream=stream,return_all_draft_input=True)
        records.append((out,model.calls,model.draft_calls,torch.rand(5,device=device)))
    for key in ('generated_token_ids','response_verification_rounds','response_accepted_length_sum'):
        assert records[0][0][key]==records[1][0][key]
    for key in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
        for a,b in zip(records[0][0][key],records[1][0][key]):assert torch.equal(a,b)
    assert records[0][1:3]==records[1][1:3]
    assert torch.equal(records[0][3],records[1][3])

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('top_p,top_k',[(.95,0),(1.,0),(.95,7),(1.,7)])
def test_sampling_metadata_preserves_distribution_rng(device,top_p,top_k):
    logits=torch.randn(3,4,41,device=device);logits[0,0]=float('nan')
    kwargs=dict(do_sample=True,temperature=.8,top_p=top_p,top_k=top_k,eos_token_id=2)
    torch.manual_seed(17);old=sample_target(logits,**kwargs)
    after=torch.rand(7,device=device)
    torch.manual_seed(17);new=sample_target_with_metadata(logits,**kwargs)
    assert torch.equal(after,torch.rand(7,device=device))
    assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1])
    assert new[2] is None  # full sorted arrays never escape sampler

@pytest.mark.skipif(not torch.cuda.is_available(),reason='compiled teacher metadata kernel')
@pytest.mark.parametrize('ties',[False,True])
@pytest.mark.parametrize('top_k',[0,5])
def test_teacher_full_permutation_reuses_sampler_and_matches_reference(ties,top_k,monkeypatch):
    from helper import opd_reflex_kernels as kernels
    s,model,_=state('cuda',v=37)
    mapping=torch.randperm(37,device='cuda')
    inverse=torch.empty_like(mapping);inverse[mapping]=torch.arange(37,device='cuda')
    model.opd_full_vocab_inverse=inverse
    s.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
    h=torch.randn(3,1,32,device='cuda');s.propose(model.draft_head(h),h,8,mapping,root=True)
    root=torch.full((3,1),-1,device='cuda',dtype=torch.long)
    tree=PackedTree(root,root.clone(),torch.zeros_like(root),0)
    path=SimpleNamespace(packed_indices=torch.zeros_like(root))
    logits=torch.zeros(3,1,37,device='cuda') if ties else torch.randn(3,1,37,device='cuda')
    def build(tokens,p,sorted_metadata):
        return s.prepare_sampler_teacher(tree,path,p,sorted_metadata)
    _,p,meta=sample_target_with_metadata(logits,do_sample=True,temperature=1.,top_p=.95,top_k=top_k,eos_token_id=2,metadata_builder=build)
    def forbidden(*a,**k):raise AssertionError('full-vocab teacher rescan')
    monkeypatch.setattr(kernels,'teacher',forbidden)
    s.feedback(tree,path,p,sampling_metadata=meta)
    expected=p[...,mapping].reshape(3,37)
    ids=torch.argsort(expected,descending=True,stable=True)[:,:16]
    prob=expected.gather(1,ids);ids=torch.where(prob>0,ids,-1)
    assert torch.equal(s.teacher_ids[:48].view(3,16),ids)
    assert torch.equal(s.teacher_p[:48].view(3,16),prob)
    assert torch.equal(s.teacher_mass[:3],torch.ones(3,device='cuda'))

@pytest.mark.parametrize('device',DEVICES)
def test_static_append_crop_repeat_compact_without_prefix_copy(device):
    c=OPDStaticCache(64);parts=[]
    for length in (5,3,7):
        x=torch.randn(2,3,length,4,device=device);parts.append(x)
        k,v=c.update(x,x+1,0)
        if len(parts)==1:ptr=k.untyped_storage().data_ptr()
        assert k.untyped_storage().data_ptr()==ptr
        assert torch.equal(k,torch.cat(parts,dim=-2))
    c.crop(6);snapshot=c[0][0].clone()
    c.batch_repeat_interleave(2)
    assert torch.equal(c[0][0],snapshot.repeat_interleave(2,0))
    c.batch_select_indices(torch.tensor([1,3],device=device))
    assert torch.equal(c[0][0],snapshot)
    assert torch.equal(c[0][1],snapshot+1)
    c.crop(0);c.update(torch.ones(2,3,1,4,device=device),torch.ones(2,3,1,4,device=device),0)
    assert c.get_seq_length()==1

def test_iteration_csv_zero_reward_weighted_aal_resume_and_idempotence(tmp_path):
    path=tmp_path/'rollout_timing.csv';w=RolloutMetricsWriter(path,'fastgrpo',flush_interval=10)
    used=0
    for i,(acc,rounds,eligible) in enumerate(((9,3,0),(2,2,2),(30,5,0))):
        w.begin(1,i,8,used);used+=eligible
        w.finish(dict(total_acc_length=acc,total_decoded_token_num=rounds,total_time_cost=2.,response_generated_tokens=[4,5]),
                 grpo_step=0,used_items=used,wall_time_s=(i+1)*3)
        w.finish(None,grpo_step=0,used_items=used,wall_time_s=0)
        if i==1:checkpoint=copy.deepcopy(w.state)
    w.close();rows=list(csv.DictReader(path.open()))
    assert len(rows)==3 and float(rows[-1]['cumulative_aal'])==4.1
    assert [int(r['eligible_prompts']) for r in rows]==[0,2,0]
    w=RolloutMetricsWriter(path,'fastgrpo',state=checkpoint)
    w.begin(2,0,8,2);w.finish(None,grpo_step=1,used_items=2,wall_time_s=10.);w.close()
    rows=list(csv.DictReader(path.open()))
    assert [r['global_iter'] for r in rows]==['1','2','3']
    assert float(rows[-1]['cumulative_aal'])==11/5
    assert all(float(r['iter_opd_kl'])==0 for r in rows)

def test_projector_lr_migrates_optimizer_moments():
    model=torch.nn.Linear(3,4);model.register_parameter('opd_projector',torch.nn.Parameter(torch.ones(3,2)))
    old=draft_optimizer(model,1e-5)
    for p in model.parameters():p.grad=torch.ones_like(p)
    old.step();saved=copy.deepcopy(old.state_dict())
    new=draft_optimizer(model,1e-5,3e-4);load_draft_optimizer(new,saved,model)
    assert new.param_groups[1]['lr']==3e-4
    for p in model.parameters():
        assert torch.equal(old.state[p]['exp_avg'],new.state[p]['exp_avg'])
        assert torch.equal(old.state[p]['step'],new.state[p]['step'])
    resumed=draft_optimizer(model,1e-5)
    load_draft_optimizer(resumed,new.state_dict(),model)
    assert resumed.param_groups[1]['lr']==3e-4
    assert torch.equal(resumed.state[model.opd_projector]['exp_avg'],old.state[model.opd_projector]['exp_avg'])

@pytest.mark.parametrize('family',['qwen2','qwen3','llama'])
@pytest.mark.parametrize('attention',['eager','sdpa'])
@pytest.mark.parametrize('device',DEVICES)
def test_pinned_hf_static_cache_native_logits(family,attention,device):
    hf=pytest.importorskip('transformers')
    if hf.__version__!='5.12.1':pytest.skip('run with isolated pinned HF 5.12.1 validation environment')
    classes={'qwen2':(hf.Qwen2Config,hf.Qwen2ForCausalLM),'qwen3':(hf.Qwen3Config,hf.Qwen3ForCausalLM),'llama':(hf.LlamaConfig,hf.LlamaForCausalLM)}
    cfg,cls=classes[family]
    config=cfg(vocab_size=41,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation=attention
    torch.manual_seed(5);model=cls(config).eval().to(device)
    from transformers.cache_utils import DynamicCache
    caches=[DynamicCache(config=config),OPDStaticCache(64)]
    with torch.no_grad():
        for x in (torch.tensor([[1,2,3],[2,4,5]]),torch.tensor([[4,5],[6,7]]),torch.tensor([[8],[9]])):
            x=x.to(device)
            outputs=[model(x,past_key_values=c,use_cache=True).logits for c in caches]
            torch.testing.assert_close(*outputs,rtol=0,atol=0)
        for c in caches:c.crop(4)
        x=torch.tensor([[10,11],[12,13]],device=device)
        torch.testing.assert_close(*[model(x,past_key_values=c,use_cache=True).logits for c in caches],rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='greedy teacher CUDA')
def test_greedy_full_vocabulary_never_scans_teacher(monkeypatch):
    from helper import opd_reflex_kernels as kernels
    s,model,_=state('cuda',v=37)
    mapping=torch.randperm(37,device='cuda');inverse=torch.empty_like(mapping)
    inverse[mapping]=torch.arange(37,device='cuda');model.opd_full_vocab_inverse=inverse
    s.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
    h=torch.randn(3,1,32,device='cuda');s.propose(model.draft_head(h),h,8,mapping,root=True)
    root=torch.full((3,1),-1,device='cuda',dtype=torch.long)
    def forbidden(*a,**k):raise AssertionError('unnecessary greedy teacher scan')
    monkeypatch.setattr(kernels,'teacher',forbidden)
    targets=torch.tensor([[1],[17],[21]],device='cuda')
    s.feedback(PackedTree(root,root.clone(),torch.zeros_like(root),0),
               SimpleNamespace(packed_indices=torch.zeros_like(root)),targets,greedy=True)
    assert torch.equal(s.teacher_ids[:48].view(3,16)[:,0],inverse[targets[:,0]])
    assert not s.teacher_p[:48].view(3,16)[:,1:].any()


def test_iteration_telemetry_does_not_retain_gpu_history():
    data={'all_target_hidden_states':torch.empty(2,3,8),'response_generated_tokens':[2,3],
          'opd_selected_states':2.,'opd_extra_tensor':torch.empty(4),'total_acc_length':5}
    captured=RolloutMetricsWriter.capture(data)
    assert set(captured)=={'response_generated_tokens','opd_selected_states','total_acc_length'}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='sparse CUDA scratch capacity')
def test_sparse_scratch_never_reserves_full_context_vocabulary(monkeypatch):
    from test_opd_reflex import seed
    monkeypatch.setenv('OPD_PROPOSAL_MODE','sparse')
    s,model,mapping=state('cuda',v=65537)
    seed(s,torch.arange(16,device='cuda'),torch.randn(16,8,device='cuda'))
    h=torch.randn(3,4,32,device='cuda')
    s.propose(model.draft_head(h),h,8,mapping)
    assert s.sparse_capacity==16+2*24*16
    assert s.sparse_scores.numel()==3*4*s.sparse_capacity
    assert s.score_workspace.numel()==1
    before=s.sparse_scores.untyped_storage().data_ptr()
    s.propose(model.draft_head(h[:1,:1]),h[:1,:1],8,mapping)
    assert s.sparse_scores.untyped_storage().data_ptr()==before
