"""Actual EAGLE/RoPE parity and tiny HF target + EAGLE GPU rollout integration."""

import ast
from pathlib import Path

import pytest
import torch

from helper.eagle3_specforge import Eagle3FastGRPOAdapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.long])
def test_rollout_prefill_creates_long_ids_without_changing_padding_positions(mask_dtype):
    # Execute only the production position-ID construction block: importing the
    # whole rollout module would require unrelated reward/runtime dependencies.
    tree = ast.parse((ROOT / "helper/specualtive_generate.py").read_text(encoding="utf-8"))
    generate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "speculative_generate")
    start = next(index for index, node in enumerate(generate.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "position_ids"
                 and isinstance(node.value, ast.ListComp))
    end = next(index for index in range(start, len(generate.body))
               if isinstance(generate.body[index], ast.Assign)
               and isinstance(generate.body[index].value, ast.Call)
               and ast.unparse(generate.body[index].value.func) == "torch.stack")
    code = compile(ast.Module(body=generate.body[start:end + 1], type_ignores=[]),
                   "specualtive_generate.py", "exec")
    scope = {
        "torch": torch, "input_ids": torch.ones(4, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1],
                                        [0, 1, 1, 1], [0, 0, 0, 0]], dtype=mask_dtype),
    }
    exec(code, scope)
    assert scope["position_ids"].dtype == torch.long
    assert torch.equal(scope["position_ids"], torch.tensor(
        [[0, 0, 0, 1], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 0]], dtype=torch.long))
    assert scope["past_position_ids"] == [1, 3, 2, -1]


@pytest.fixture
def tiny_adapter(monkeypatch):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from specforge.modeling.draft import llama3_eagle as eagle

    # Only unwrap helper compilation for ordinary CPU parity tests; production
    # decorators remain unchanged. A separate test exercises Dynamo below.
    for cls in (eagle.LlamaRMSNorm, eagle.LlamaRotaryEmbedding):
        monkeypatch.setattr(cls, "forward", cls.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, num_hidden_layers=1,
                      max_position_embeddings=2048, vocab_size=32, draft_vocab_size=16,
                      target_hidden_size=16, pretraining_tp=1, tie_word_embeddings=False)
    torch.manual_seed(42)
    draft = eagle.LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    # Bypass only checkpoint/target loading; use real draft weights, adapter
    # forward, SDPA, feature projection, RoPE and the flat KV-cache code.
    adapter = Eagle3FastGRPOAdapter.__new__(Eagle3FastGRPOAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.draft_model = draft
    adapter.config = cfg
    adapter.dtype = torch.float32
    return adapter


def prefill_inputs(dtype):
    positions = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]], dtype=torch.long)
    mask = torch.zeros(2, 1, 4, 4, dtype=dtype).masked_fill(
        torch.ones(4, 4, dtype=torch.bool).triu(1), torch.finfo(dtype).min)
    mask[0, :, :, :2] = torch.finfo(dtype).min
    return {
        "hidden_states": torch.randn(2, 4, 48, dtype=dtype),
        "input_ids": torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
        "attention_mask": mask,
        "position_ids": positions,
    }


def assert_outputs_identical(actual, expected):
    for key in ("hidden_states", "next_feature_states"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert torch.isfinite(actual[key]).all()
    for actual_layer, expected_layer in zip(actual["past_key_values"], expected["past_key_values"]):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("position_dtype", [torch.float32, torch.float64, torch.bfloat16,
                                           torch.int32, torch.int64])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_adapter_prefill_and_cached_decode_match_long_reference(tiny_adapter, position_dtype, model_dtype):
    adapter = tiny_adapter.to(model_dtype)
    adapter.dtype = model_dtype
    inputs = prefill_inputs(model_dtype)
    expected = adapter(**inputs)
    supplied = dict(inputs, position_ids=inputs["position_ids"].to(position_dtype))
    actual = adapter(**supplied)
    assert_outputs_identical(actual, expected)
    assert actual["past_key_values"][0][0].shape[-2] == 4

    # Subsequent draft steps use native predicted features and flattened KV.
    decode_mask = torch.zeros(2, 1, 1, 5, dtype=model_dtype)
    decode_mask[0, :, :, :2] = torch.finfo(model_dtype).min
    decode = {
        "hidden_states": actual["next_feature_states"][:, -1:, :],
        "input_ids": torch.tensor([[9], [10]], dtype=torch.long),
        "attention_mask": decode_mask, "past_key_values": actual["past_key_values"],
        "position_ids": torch.tensor([[2], [4]], dtype=torch.long),
    }
    reference = adapter(**decode)
    converted = adapter(**dict(decode, position_ids=decode["position_ids"].to(position_dtype)))
    assert_outputs_identical(converted, reference)
    assert converted["past_key_values"][0][0].shape[-2] == 5
    assert torch.equal(supplied["position_ids"], inputs["position_ids"].to(position_dtype))


def test_adapter_default_positions_preserve_cache_offset_and_backward(tiny_adapter):
    inputs = prefill_inputs(torch.float32)
    inputs.pop("position_ids")
    expected = tiny_adapter(**dict(inputs, position_ids=torch.arange(4).unsqueeze(0)))
    actual = tiny_adapter(**inputs)
    assert_outputs_identical(actual, expected)
    decode = {
        "hidden_states": actual["hidden_states"][:, -1:, :],
        "input_ids": torch.tensor([[9], [10]], dtype=torch.long),
        "attention_mask": torch.zeros(2, 1, 1, 5),
        "past_key_values": actual["past_key_values"],
    }
    expected = tiny_adapter(**dict(decode, position_ids=torch.tensor([[4]], dtype=torch.long)))
    actual = tiny_adapter(**decode)
    assert_outputs_identical(actual, expected)
    actual["hidden_states"].square().mean().backward()
    assert tiny_adapter.draft_model.fc.weight.grad is not None
    assert torch.isfinite(tiny_adapter.draft_model.fc.weight.grad).all()
    uncached = tiny_adapter(**dict(inputs, use_cache=False))
    assert uncached["past_key_values"] == []


def test_float_positions_work_with_dynamo_rotary_compilation(tiny_adapter, monkeypatch):
    from specforge.modeling.draft import llama3_eagle as eagle

    # Exercise real Dynamo tracing/fake tensor indexing with fixed test shapes.
    # Dynamic symbolic guards may invoke MSVC even with backend="eager" on
    # Windows; no production compile decorator/configuration is changed.
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        torch.compile(eagle.apply_rotary_pos_emb, backend="eager", dynamic=False))
    inputs = prefill_inputs(torch.float32)
    reference = tiny_adapter(**inputs)
    actual = tiny_adapter(**dict(inputs, position_ids=inputs["position_ids"].float()))
    assert_outputs_identical(actual, reference)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_eagle_static_cache_matches_legacy_prefill_append_and_rollback(tiny_adapter,dtype,device):
    from helper.opd_static_cache import OPDStaticCache
    adapter=tiny_adapter.to(device=device,dtype=dtype);adapter.dtype=dtype
    with torch.no_grad():
        inputs={k:v.to(device) for k,v in prefill_inputs(dtype).items()}
        legacy=adapter(**inputs)
        static=adapter(**inputs,past_key_values=OPDStaticCache(32))
        assert_outputs_identical(static,legacy)
        ptr=static['past_key_values'][0][0].untyped_storage().data_ptr()
        for length in (2,1):
            past=legacy['past_key_values'][0][0].shape[-2]
            args=dict(hidden_states=legacy['next_feature_states'][:,-1:].expand(2,length,16),
                input_ids=torch.ones(2,length,dtype=torch.long,device=device),
                attention_mask=torch.zeros(2,1,length,past+length,dtype=dtype,device=device),
                position_ids=torch.arange(past,past+length,device=device)[None,:])
            legacy=adapter(**args,past_key_values=legacy['past_key_values'])
            static=adapter(**args,past_key_values=static['past_key_values'])
            assert_outputs_identical(static,legacy)
            assert static['past_key_values'][0][0].untyped_storage().data_ptr()==ptr
        static['past_key_values'].crop(4)
        assert torch.equal(static['past_key_values'][0][0],legacy['past_key_values'][0][0][...,:4,:])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real tiny HF + SpecForge GPU rollout')
@pytest.mark.parametrize('update_stream',[False,True])
def test_real_hf_eagle_rollout_mask_cache_sampler_feedback_integration(tiny_adapter,update_stream):
    from types import SimpleNamespace
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from helper.opd_reflex import initialize_projector
    from helper.specualtive_generate import speculative_generate
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4)
    config._attn_implementation='sdpa'
    adapter=tiny_adapter.cuda().to(torch.bfloat16);adapter.dtype=torch.bfloat16
    adapter.target_model=Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    adapter.target_model._fastgrpo_eagle3_capture_layers=[0,0,1]
    adapter.draft_model.register_parameter('opd_projector',torch.nn.Parameter(initialize_projector(16,8).cuda()))
    adapter.draft_model.register_buffer('opd_projector_grad_sum',torch.zeros(16,8,device='cuda'))
    adapter.draft_model.register_buffer('opd_projector_grad_weight',torch.zeros(1,device='cuda'))
    adapter.draft_model.d2t.zero_()
    calls={'target':0,'draft':0}
    hooks=[adapter.target_model.model.register_forward_pre_hook(lambda *a:calls.__setitem__('target',calls['target']+1)),
        adapter.register_forward_pre_hook(lambda *a:calls.__setitem__('draft',calls['draft']+1))]
    try:
        results=[]
        import importlib.util
        spec=importlib.util.spec_from_file_location('_previous_opd_integration',ROOT/'tests/oracles/opd_rollout_before_memory.py')
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        for mode in ('previous','dynamic','persistent','reuse'):
            adapter.supports_opd_static_kv=mode!='dynamic'
            calls.update(target=0,draft=0)
            torch.manual_seed(71)
            generate=previous.speculative_generate if mode=='previous' else speculative_generate
            output=generate(adapter,torch.tensor([[1,2,3],[3,4,5]],device='cuda'),
                torch.ones(2,3,dtype=torch.long,device='cuda'),SimpleNamespace(eos_token_id=31),
                do_sample=True,repeated_generate_nums=2,max_length=24,verification_capacity=24,
                max_draft_k=2,max_draft_token_length=2,min_draft_token_length=2,max_verification_num=8,
                method='opd_reflex',opd_update_stream=update_stream,return_all_draft_input=True)
            assert calls['target']==1+output['batch_verification_rounds']
            assert calls['draft']==2*output['batch_verification_rounds']
            assert len(output['generated_token_ids'])==4
            assert output['opd_host_syncs_per_round']==1
            if mode=='reuse':
                for side in ('target','draft'):
                    assert output['opd_'+side+'_full_kv_reallocations']==0
                    assert getattr(adapter,'_opd_'+side+'_kv_pool').get_seq_length()==0
            results.append((output,torch.cuda.get_rng_state()))
        legacy=results[0][0]
        for growable,rng in results[1:]:
            assert torch.equal(results[0][1],rng)
            for key in ('generated_token_ids','total_acc_length','total_decoded_token_num',
                        'response_accepted_length_sum','response_verification_rounds'):
                assert legacy[key]==growable[key]
            for key in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
                for before,after in zip(legacy[key],growable[key]):
                    torch.testing.assert_close(before,after,rtol=0,atol=0)
    finally:
        for hook in hooks:hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native HF + EAGLE benchmark driver smoke')
def test_before_after_benchmark_driver_smoke_with_real_tiny_models(tiny_adapter,tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace,ModuleType
    import transformers
    from helper import eagle3_specforge
    from helper.opd_reflex import initialize_projector
    from scripts import benchmark_opd_reflex as bench
    config=transformers.Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4)
    config._attn_implementation='sdpa'
    target=transformers.Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    target._fastgrpo_eagle3_capture_layers=[0,0,1]
    adapter=tiny_adapter.cuda().to(torch.bfloat16);adapter.dtype=torch.bfloat16
    adapter.target_model=target
    adapter.lm_head=eagle3_specforge._TargetVocabHead(adapter.draft_model)
    adapter.draft_model.d2t.zero_()
    adapter.draft_model.register_parameter('opd_projector',torch.nn.Parameter(initialize_projector(16,8).cuda()))
    adapter.draft_model.register_buffer('opd_projector_grad_sum',torch.zeros(16,8,device='cuda'))
    adapter.draft_model.register_buffer('opd_projector_grad_weight',torch.zeros(1,device='cuda'))
    tokenizer=SimpleNamespace(eos_token_id=31,pad_token_id=0,pad_token='<pad>')
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:target)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tokenizer)
    monkeypatch.setattr(eagle3_specforge,'Eagle3FastGRPOAdapter',lambda *a,**k:adapter)
    # ONLY external loading/collation mocked; model forwards, KV, native sampler,
    # tree/verifier, OPD kernels, CUDA timing and report aggregation are REAL.
    data=ModuleType('helper.get_QAs');data.get_QAs_from_path=lambda *a:[{'prompt':'tiny fixture'}]
    monkeypatch.setitem(sys.modules,'helper.get_QAs',data)
    monkeypatch.setattr(bench,'collator',lambda *a:lambda rows:dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long)))
    args=bench.parse_args(['--target-model','tiny-fixture','--draft-checkpoint','tiny-fixture',
        '--draft-config','tiny-fixture','--vocab-mapping','tiny-fixture','--dataset-path','tiny-fixture',
        '--output',str(tmp_path),'--batch-size','1','--responses','2','--iterations','1','--warmup','1',
        '--max-length','12','--max-prompt-length','3','--max-draft-k','2','--max-draft-length','2',
        '--min-draft-length','2','--verification-capacity','24','--max-verification-num','8',
        '--fast-lrs','0.01','--streams','1','--seeds','71','--include-previous-opd'])
    result=bench.benchmark(args)
    assert [r['method'] for r in result['reports']]==['fastgrpo','opd_reflex_previous','opd_reflex']
    before,after=result['reports'][1:]
    assert before['aal']==after['aal']
    assert after['host_syncs_per_round']==1
    assert after['kv_pool_allocations_per_iter']==0
    assert after['kv_reallocations_per_iter']==0
    assert after['tokens_per_s']>0 and after['peak_allocated_bytes']>0
    assert result['fastest_observed']['method']=='opd_reflex'


def test_fastgrpo_adapter_has_no_projector_and_keeps_checkpoint_format(tmp_path,monkeypatch):
    from transformers import LlamaConfig,Qwen2Config,Qwen2ForCausalLM
    from specforge.modeling.draft import llama3_eagle as eagle
    for cls in (eagle.LlamaRMSNorm,eagle.LlamaRotaryEmbedding):
        monkeypatch.setattr(cls,'forward',cls.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle,'apply_rotary_pos_emb',eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)
    config=LlamaConfig(hidden_size=16,intermediate_size=32,num_attention_heads=4,
        num_key_value_heads=2,head_dim=4,num_hidden_layers=1,vocab_size=32,draft_vocab_size=16,
        target_hidden_size=16,pretraining_tp=1,tie_word_embeddings=False,architectures=['LlamaForCausalLMEagle3'])
    path=tmp_path/'config.json';config.to_json_file(path)
    mapping=tmp_path/'mapping.pt'
    torch.save(dict(d2t=torch.zeros(16,dtype=torch.long),t2d=torch.arange(32)<16),mapping)
    target=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4))
    adapter=Eagle3FastGRPOAdapter(target,str(path),vocab_mapping=str(mapping),
        initialization_mode='random',feature_layers=[0,0,1],opd_rank=None)
    assert adapter.opd_projector is None
    assert not any('opd_projector' in name for name,_ in adapter.draft_model.named_parameters())
    checkpoint=tmp_path/'draft.pt';adapter.save_model(checkpoint)
    saved=torch.load(checkpoint,weights_only=True)
    assert saved['format']=='specforge_eagle3_fastgrpo_v1'
    assert saved['opd_projector'] is None
    adapter.load_model(checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual historical/shared GPU EAGLE3 parity')
def test_actual_eagle_fastgrpo_shared_rollout_preserves_historical_rng_and_tokens(tiny_adapter):
    from types import SimpleNamespace
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from helper.eagle3_specforge import _TargetVocabHead
    from helper.historical_fastgrpo import speculative_generate as historical
    from helper.specualtive_generate import speculative_generate as shared
    adapter=tiny_adapter.cuda().to(torch.bfloat16);adapter.dtype=torch.bfloat16
    cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4)
    cfg._attn_implementation='sdpa'
    adapter.target_model=Qwen2ForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    adapter.target_model._fastgrpo_eagle3_capture_layers=[0,0,1]
    adapter.lm_head=_TargetVocabHead(adapter.draft_model);adapter.draft_model.d2t.zero_()
    results=[]
    for old in (True,False):
        torch.manual_seed(71)
        kwargs={} if old else dict(method='fastgrpo')
        with torch.inference_mode():
            output=(historical if old else shared)(adapter,torch.tensor([[1,2,3],[3,4,5]],device='cuda'),
                torch.ones(2,3,dtype=torch.long,device='cuda'),SimpleNamespace(eos_token_id=31),
                do_sample=True,repeated_generate_nums=2,max_length=24,verification_capacity=24,
                max_draft_k=2,max_draft_token_length=2,min_draft_token_length=2,max_verification_num=8,
                return_all_draft_input=True,**kwargs)
        results.append((output,torch.cuda.get_rng_state()))
    assert torch.equal(results[0][1],results[1][1])
    for key in ('generated_token_ids','total_acc_length','total_decoded_token_num'):
        assert results[0][0][key]==results[1][0][key]
    for key in ('all_draft_input_states','all_target_hidden_states','all_draft_input_ids'):
        for a,b in zip(results[0][0][key],results[1][0][key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
