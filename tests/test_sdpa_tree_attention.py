"""Native PyTorch SDPA tree-mask/backward smoke, not an HF/B200 benchmark."""
import math

import pytest
import torch
import torch.nn.functional as F

from helper.tree_verification import PackedTree

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_sdpa_custom_tree_mask_cached_prefix_and_backward(device, dtype):
    torch.manual_seed(41)
    parents = torch.tensor([[-1, 0, 0, 1, 2]] * 2, device=device)
    tree = PackedTree(parents, torch.zeros_like(parents), torch.zeros_like(parents), 2)
    mask = tree.attention_mask(3, dtype)
    mask[0, :, :, 0] = torch.finfo(dtype).min  # left-padded cached prefix
    q = torch.randn(2, 4, 5, 16, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 8, 16, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 2, 8, 16, device=device, dtype=dtype, requires_grad=True)
    kr, vr = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    result = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask,
                                          dropout_p=0., is_causal=False)
    weights = (q.float().matmul(kr.float().transpose(-1, -2)) / math.sqrt(16) + mask.float()).softmax(-1)
    reference = weights.matmul(vr.float()).to(dtype)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-5, 2e-6)
    torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
    grad = torch.randn_like(result)
    actual_grad = torch.autograd.grad(result, (q, k, v), grad, retain_graph=True)
    reference_grad = torch.autograd.grad(reference, (q, k, v), grad)
    for actual, expected in zip(actual_grad, reference_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # For query node3, node2 is NOT its ancestor; altered KV cannot leak across.
    with torch.no_grad():
        changed_k, changed_v = kr.clone(), vr.clone()
        changed_k[:, :, 3 + 2] = 10000.
        changed_v[:, :, 3 + 2] = -10000.
        changed = F.scaled_dot_product_attention(q, changed_k, changed_v,
            attn_mask=mask, dropout_p=0., is_causal=False)
        torch.testing.assert_close(result[:, :, 3], changed[:, :, 3], rtol=rtol, atol=atol)
