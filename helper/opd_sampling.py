"""Identical target sampling/RNG; transient sort consumed into small metadata."""
import torch
from helper.sampling import sample_from_probs


def sample_target_with_metadata(logits,*,do_sample,temperature,top_p,top_k,eos_token_id,metadata_builder=None,return_probs=True):
    if not do_sample:
        tokens=logits.argmax(-1)
        metadata=metadata_builder(tokens,None,None) if metadata_builder is not None else None
        return tokens,None,metadata
    if temperature<=0:raise ValueError('sampling temperature must be positive')
    vocabulary=logits.shape[-1]
    flat=logits.reshape(-1,vocabulary).float()/float(temperature)
    flat=torch.where(torch.isfinite(flat),flat,torch.full_like(flat,-torch.inf))
    invalid=torch.isneginf(flat).all(dim=-1)
    fallback=torch.full_like(flat,-torch.inf);fallback[:,int(eos_token_id)]=0.
    flat=torch.where(invalid.unsqueeze(-1),fallback,flat)
    probs=flat.softmax(dim=-1);metadata=None
    del flat,fallback,invalid
    if top_p is not None and 0.<float(top_p)<1.:
        values,indices=torch.sort(probs,descending=True,dim=-1)
        remove=torch.cumsum(values,dim=-1)>float(top_p)
        remove=torch.roll(remove,shifts=1,dims=-1);remove[...,0]=False
        values.masked_fill_(remove,0.)
        del remove
        values.div_(values.sum(dim=-1,keepdim=True).clamp_min(1.e-20))
        probs.zero_().scatter_(-1,indices,values)
        metadata=(values,indices)
    if top_k is not None and 0<int(top_k)<vocabulary:
        values,indices=torch.topk(probs,k=int(top_k),dim=-1)
        values.div_(values.sum(dim=-1,keepdim=True).clamp_min(1.e-20))
        probs.zero_().scatter_(-1,indices,values)
        metadata=(values,indices)
    probs=probs.reshape(*logits.shape[:-1],vocabulary)
    tokens=sample_from_probs(probs)
    # Consume the existing sort into independently owned small buffers while
    # its lifetime is confined to this sampler. Never return an [N,V] view.
    small_metadata=metadata_builder(tokens,probs,metadata) if metadata_builder is not None else None
    return tokens,probs if return_probs else None,small_metadata
