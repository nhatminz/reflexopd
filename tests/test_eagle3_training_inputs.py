"""Inference rollout -> real EAGLE draft backward, using tiny CPU weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helper.eagle3_specforge import rollout_tensor_for_training,Eagle3FastGRPOAdapter


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "third_party/SpecForge/specforge"


@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.bfloat16])
def test_inference_conversion_preserves_values_and_supports_backward(dtype):
    with torch.inference_mode():
        original = torch.arange(12).reshape(3, 4).to(dtype).clone()
        # The helper must disable inference mode for the clone, not just clone
        # under an inherited inference context.
        converted = rollout_tensor_for_training(original)
    assert torch.is_inference(original)
    assert not torch.is_inference(converted)
    assert not converted.requires_grad
    assert converted.dtype == original.dtype
    assert converted.device == original.device
    assert converted.data_ptr() != original.data_ptr()
    torch.testing.assert_close(converted, original, rtol=0, atol=0)
    if dtype == torch.long:
        layer = torch.nn.Embedding(12, 2)
    else:
        layer = torch.nn.Linear(4, 2, bias=False).to(dtype)
    layer(converted).float().sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_normal_input_is_reused_without_detaching_graph():
    leaf = torch.randn(3, requires_grad=True)
    original = leaf * 2
    assert rollout_tensor_for_training(original) is original
    original.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


@pytest.fixture
def cpu_training_model(monkeypatch):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from specforge.modeling.draft import llama3_eagle as eagle

    # Windows CPU has neither Triton nor a C++ compiler. Unwrap only the tiny
    # fixture's compiled RoPE/norm helpers, without changing production code.
    for cls in (eagle.LlamaRMSNorm, eagle.LlamaRotaryEmbedding):
        monkeypatch.setattr(cls, "forward", cls.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)

    # Use SpecForge's own reference loss rather than its CUDA-only Triton kernel.
    # Execute the real OnlineEagle3Model source with only that import replaced;
    # feature projection, SDPA, teacher, TTT and LK equations stay real.
    loss_tree = ast.parse((SPEC / "core/loss.py").read_text(encoding="utf-8"))
    reference = next(node for node in loss_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    reference.decorator_list = []
    scope = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[reference], type_ignores=[]), "loss.py", "exec"), scope)
    def reference_loss(logits, target_p, position_mask, per_sample=False):
        if not per_sample:
            return scope["_compute_loss"](logits, target_p, position_mask)
        logp = torch.log_softmax(logits.float(), dim=-1)
        return -(target_p * logp * position_mask).sum(dim=-1).mean(dim=1)
    scope["LogSoftmaxLoss"] = SimpleNamespace(apply=reference_loss)
    model_tree = ast.parse((SPEC / "algorithms/eagle3/model.py").read_text(encoding="utf-8"))
    model_tree.body = [node for node in model_tree.body
                       if not (isinstance(node, ast.ImportFrom)
                               and node.module == "specforge.core.loss")]
    exec(compile(model_tree, "eagle3/model.py", "exec"), scope)

    torch.manual_seed(42)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, num_hidden_layers=1,
                      max_position_embeddings=2048, vocab_size=32, draft_vocab_size=16,
                      target_hidden_size=16, pretraining_tp=1, tie_word_embeddings=False)
    draft = eagle.LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    draft.t2d[16:] = False
    target = torch.nn.Module()
    target.lm_head = torch.nn.Linear(16, 32, bias=False)
    # Concentrated full-vocab teacher: argmax=0 is inside the selected draft
    # vocab, with probabilities straddling the draft's near-uniform outputs.
    # This exercises nonzero gradients for acceptance-only LK objectives too.
    with torch.no_grad():
        target.lm_head.weight.zero_()
        target.lm_head.weight[0, 0] = 5
    target.requires_grad_(False)
    online = scope["OnlineEagle3Model"](draft, length=7, attention_backend="sdpa")
    model=Eagle3FastGRPOAdapter.__new__(Eagle3FastGRPOAdapter)
    torch.nn.Module.__init__(model)
    model.draft_model=draft;model.target_model=target;model.specforge_training_model=online;model.config=cfg
    model.dtype=torch.float32
    return model


def training_entrypoint():
    # Importing grpo_speculative runs CUDA/model/reward initialization. Execute
    # just the unchanged production function instead, as other source tests do.
    tree = ast.parse((ROOT / "grpo_speculative.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "training_eagle3_specforge")
    scope = {"torch": torch, "repeated_generate_nums": 2,
             "_get_base_causal_lm": lambda model: model,
             "rollout_tensor_for_training": rollout_tensor_for_training}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "grpo_speculative.py", "exec"), scope)
    return scope["training_eagle3_specforge"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lk_loss_type", [None, "alpha", "tv", "lambda"])
@pytest.mark.parametrize("token_budget", [None, 5])
def test_rollout_training_matches_normal_inputs_loss_and_gradients(
    cpu_training_model, dtype, lk_loss_type, token_budget,
):
    model = cpu_training_model
    model.draft_model.to(dtype)
    model.target_model.to(dtype)
    model.dtype = dtype
    model.specforge_training_model.lk_loss_type = lk_loss_type
    outputs = {
        "all_draft_input_states": [torch.randn(16, 48, dtype=dtype) for _ in range(2)],
        "all_target_hidden_states": [torch.randn(16, 16, dtype=dtype) for _ in range(2)],
        "all_draft_input_ids": [torch.randint(0, 32, (16,)) for _ in range(2)],
    }
    for row in outputs["all_target_hidden_states"]:
        row[:, 0] = 1
    with torch.inference_mode():
        inference_outputs = {key: [row.clone() for row in rows] for key, rows in outputs.items()}
    assert all(torch.is_inference(row) for rows in inference_outputs.values() for row in rows)
    training = training_entrypoint()
    prompt_mask = torch.ones(1, 4, dtype=torch.long)
    expected = training(model, outputs, prompt_mask, token_budget)
    grads = {name: param.grad.clone() for name, param in model.draft_model.named_parameters()
             if param.grad is not None}
    assert "fc.weight" in grads
    assert torch.count_nonzero(grads["fc.weight"]) > 0
    model.draft_model.zero_grad(set_to_none=True)
    # Observe actual boundary tensors as well as checking successful backward.
    def check_inputs(module, args, kwargs):
        for key in ("input_ids", "hidden_states"):
            assert not torch.is_inference(kwargs[key])
            assert not kwargs[key].requires_grad
    hook = model.register_forward_pre_hook(check_inputs, with_kwargs=True)
    try:
        actual = training(model, inference_outputs, prompt_mask, token_budget)
    finally:
        hook.remove()
    assert actual == expected
    assert actual[-1] == (22 if token_budget is None else 5)
    for name, param in model.draft_model.named_parameters():
        if name in grads:
            assert torch.isfinite(param.grad).all()
            torch.testing.assert_close(param.grad, grads[name], rtol=0, atol=0)
        else:
            assert param.grad is None
    assert model.target_model.lm_head.weight.grad is None
    for key in outputs:
        for inference_row, normal_row in zip(inference_outputs[key], outputs[key]):
            assert torch.is_inference(inference_row)
            torch.testing.assert_close(inference_row, normal_row, rtol=0, atol=0)


def test_zero_budget_skips_training(cpu_training_model):
    model = cpu_training_model
    model.dtype = torch.float32
    with torch.inference_mode():
        outputs = {
            "all_draft_input_states": [torch.randn(16, 48)],
            "all_target_hidden_states": [torch.randn(16, 16)],
            "all_draft_input_ids": [torch.randint(0, 32, (16,))],
        }
    assert training_entrypoint()(model, outputs, torch.ones(1, 4), 0) == (0, 0, 0, 0, 0)
    assert all(param.grad is None for param in model.draft_model.parameters())


@pytest.mark.parametrize("lk_loss_type", [None, "alpha", "tv", "lambda"])
@pytest.mark.parametrize("token_budget", [None, 14])
def test_batched_matches_per_response_for_ragged_rows(cpu_training_model, lk_loss_type, token_budget):
    model = cpu_training_model
    model.dtype = torch.float32
    model.specforge_training_model.lk_loss_type = lk_loss_type
    lengths = [13, 17, 15, 12]
    outputs = {
        "all_draft_input_states": [torch.randn(n, 48) for n in lengths],
        "all_target_hidden_states": [torch.randn(n, 16) for n in lengths],
        "all_draft_input_ids": [torch.randint(0, 32, (n,)) for n in lengths],
    }
    for row in outputs["all_target_hidden_states"]:
        row[:, 0] = 1
    prompt_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    train = training_entrypoint()
    old = train(model, outputs, prompt_mask, token_budget, mode="per_response")
    old_grads = {name: p.grad.detach().clone() for name, p in model.draft_model.named_parameters()
                 if p.grad is not None}
    model.draft_model.zero_grad(set_to_none=True)
    new = train(model, outputs, prompt_mask, token_budget, mode="batched",
                max_batch_size=4, max_tokens=100, max_padding_ratio=1.5)
    assert old[-1] == new[-1]
    assert old[0] == pytest.approx(new[0], rel=2e-4, abs=2e-5)
    assert old[1] == pytest.approx(new[1], rel=2e-4, abs=2e-5)
    for name, p in model.draft_model.named_parameters():
        if name in old_grads:
            torch.testing.assert_close(p.grad, old_grads[name], rtol=4e-3, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SpecForge Triton loss requires CUDA")
def test_triton_per_sample_loss_matches_reference_gradients():
    from specforge.core.loss import LogSoftmaxLoss
    logits = torch.randn(3, 5, 16, device="cuda", requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    teacher = torch.softmax(torch.randn_like(logits), dim=-1)
    mask = torch.randint(0, 2, (3, 5, 1), device="cuda")
    actual = LogSoftmaxLoss.apply(logits, teacher, mask, True)
    reference = -(teacher * torch.log_softmax(reference_logits.float(), dim=-1)
                  * mask).sum(dim=-1).mean(dim=-1)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-5)
