"""Method-neutral tree/mask/path buffers and historical raw FastGRPO proposal."""
import importlib
import torch
from helper.opd_attention import AttentionWorkspace


def allocate_tree_buffers(runtime,alloc,batch,max_contexts,max_path,max_proposal_contexts):
    """One allocator shared by FastGRPO and OPD; contains no Reflex state."""
    runtime.path_workspace=[alloc((batch,max_path),torch.long) for _ in range(3)]+[alloc(batch,torch.long)]
    runtime.padded_path_workspace=[alloc((batch,max_path),torch.bool if i==2 else torch.long) for i in range(3)]+[alloc((batch,1),torch.long)]
    runtime.scheduling_packet=alloc((batch,max_path+4),torch.long)
    full_nodes=max_proposal_contexts*max_contexts
    runtime.tree_buffers={name:alloc((batch,full_nodes),torch.float32 if name=='confidence' else torch.long)
        for name in ('parents','contexts','tokens','positions','confidence')}
    runtime.tree_arange=torch.arange(full_nodes+1,device=runtime.attention_workspace.device,dtype=torch.long)
    runtime.tree_seen=[alloc((batch,max_proposal_contexts,max_path),torch.long) for _ in range(2)]
    runtime.tree_positions=alloc((batch,max_proposal_contexts),torch.long)
    runtime.tree_branch_confidence=alloc((batch,max_proposal_contexts,max_proposal_contexts))
    runtime.tree_top_values=alloc((batch,max_proposal_contexts))
    runtime.tree_top_indices=alloc((batch,max_proposal_contexts),torch.long)
    runtime.pack_workspace=[alloc(batch*(full_nodes+1),torch.long) for _ in range(4)]
    runtime.confidence_key_workspace=alloc(batch*full_nodes,torch.int64)
    runtime.tree_root_confidences=alloc((batch,max_proposal_contexts))


class FastGRPORuntime:
    """Infrastructure only: no A/B, teacher/union, OPD dispatcher or update stream."""
    enabled=False
    def __init__(self):self._layout=None
    def start(self,model,batch,mapping,hidden_size,*,max_contexts,max_nodes,max_path,max_proposal_contexts):
        device=mapping.device
        if not hasattr(self,'attention_workspace') or self.attention_workspace.device!=device:
            self.attention_workspace=AttentionWorkspace(device)
        self._kernels=importlib.import_module('helper.tree_kernels') if device.type=='cuda' else None
        layout=(batch,max_contexts,max_path,max_proposal_contexts,str(device))
        if layout!=self._layout:
            self._layout=layout
            def alloc(shape,dtype=torch.float32):return torch.empty(shape,device=device,dtype=dtype)
            allocate_tree_buffers(self,alloc,batch,max_contexts,max_path,max_proposal_contexts)
        self.host_sync_count=0;self.host_active_count=0
        self.feedback_row_map=None
    def propose(self,logits,hidden,k,mapping,**unused):
        # Exactly historical FP32 probability calculation and native TopK(k).
        probabilities=logits.float().softmax(-1)
        values,ids=torch.topk(probabilities,k=int(k),dim=-1)
        return values,ids,mapping[ids]
    def begin(self,label):return None
    def end(self,ticket):pass
    def finish(self):return dict(opd_backend='off',opd_profile_time_ms=0.,opd_profile_sections_ms=None)
    def clear(self):pass
