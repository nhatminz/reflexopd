"""Persistent FastGRPO loss on actual EAGLE3 outputs; independent of OPD A/B."""
import torch
import torch.nn.functional as F


def persistent_loss(model,ids,features,target_features,attention_mask,loss_mask,target_head,chunk_size=256):
    b,length=ids.shape
    # Same causal attention/teacher forcing as generation; no TTT unrolling.
    minimum=torch.finfo(model.dtype).min
    causal=torch.triu(torch.full((length,length),minimum,device=ids.device,dtype=model.dtype),diagonal=1)
    mask=causal[None,None].expand(b,1,length,length).clone()
    mask.masked_fill_(~attention_mask[:,None,None,:].bool(),minimum)
    position_ids=attention_mask.long().cumsum(1).sub(1).clamp_min(0)
    output=model(hidden_states=features,input_ids=ids,attention_mask=mask,position_ids=position_ids,use_cache=False)
    predicted=output['next_feature_states'][:,:-1]
    teacher=target_features[:,1:].detach()
    if predicted.shape!=teacher.shape:
        raise ValueError('EAGLE3 predicted H must match final target feature H (not concatenated 3H inputs)')
    valid=loss_mask[:,:-1].bool() & attention_mask[:,:-1].bool() & attention_mask[:,1:].bool()
    denom=valid.sum(1).clamp_min(1).float()
    feature=(F.smooth_l1_loss(predicted.float(),teacher.float(),reduction='none').mean(-1)*valid).sum(1)/denom*2.
    distribution=feature.new_zeros(b)
    # Head work only on supervised positions, chunked to bound full-vocab teacher
    # temporary memory. Full target transformer never runs here.
    positions=valid.nonzero(as_tuple=False)
    mapping=model.compact_to_target_ids(device=ids.device)
    for start in range(0,positions.shape[0],chunk_size):
        rows,columns=positions[start:start+chunk_size].unbind(1)
        with torch.no_grad():
            target_logits=target_head(teacher[rows,columns].to(target_head.weight.dtype))
            target_prob=target_logits.float().softmax(-1).detach()
            compact_prob=target_prob.index_select(-1,mapping)
            mass=compact_prob.sum(-1,keepdim=True)
            torch._assert_async((torch.isfinite(mass)&(mass>0)).all(),'invalid compact teacher probability mass')
            compact_prob=compact_prob/mass
        draft_logits=model.compute_compact_logits(output['hidden_states'][rows,columns])
        ce=-(compact_prob*F.log_softmax(draft_logits.float(),-1)).sum(-1)
        distribution=distribution.index_add(0,rows,ce)
    distribution=distribution/denom*.1
    return feature,distribution
