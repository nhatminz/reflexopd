"""Explicit EAGLE3 backend validation; never substitute another backend."""

from __future__ import annotations


def validate_attention_backend(backend: str, *, probe: bool = False) -> str:
    if backend not in {"fa", "sdpa", "flex_attention"}:
        raise ValueError("PRETRAIN_ATTENTION_BACKEND must be fa, sdpa or flex_attention")
    if backend != "fa":
        return backend
    try:
        from specforge.modeling.draft import llama3_eagle as eagle

        required = (
            eagle._std_flash_attn_varlen_func,
            eagle._std_flash_attn_varlen_backward,
            eagle._std_flash_unpad_input,
            eagle._std_flash_pad_input,
        )
        if any(function is None for function in required):
            eagle._raise_standard_flash_attn_unavailable()
        if probe:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            # Exercise the same standard varlen forward/backward API as the
            # EAGLE3 FA implementation, not merely the package import.
            q = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            mask = torch.ones(1, 16, device="cuda", dtype=torch.bool)
            out, lse, _, _ = eagle._standard_flash_attn_varlen_forward(
                q, k, v, mask, 128 ** -0.5, True,
            )
            # The helper has exactly the backward interface used in TTT.
            eagle._standard_flash_attn_varlen_backward_call(
                torch.ones_like(out), q, k, v, out, lse, mask,
                torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
                128 ** -0.5, True,
            )
            torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "PRETRAIN_ATTENTION_BACKEND=fa is unavailable on this runtime/device. "
            "EAGLE requires flash_attn.flash_attn_varlen_func, "
            "flash_attn.flash_attn_interface._flash_attn_varlen_backward and "
            "flash_attn.bert_padding.{pad_input,unpad_input}; an importable "
            "flash_attn namespace alone does not establish compatibility. "
            "Install a compatible FlashAttention build or explicitly choose "
            "PRETRAIN_ATTENTION_BACKEND=sdpa (or flex_attention). "
            f"No fallback was selected. Cause: {type(exc).__name__}: {exc}"
        ) from exc
    return backend
