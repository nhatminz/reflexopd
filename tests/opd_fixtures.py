"""Deterministic tiny fixtures, REAL sampler/tree/verifier/OPD, not throughput data."""
import ast
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import time
import os
import torch
from helper.opd_reflex import OPDReflex,OPD_COUNTER_NAMES
from helper.shared_rollout import FastGRPORuntime
from helper.method_config import resolve_method
from helper.rollout_history import RolloutHistory
from helper.opd_history import ContiguousRolloutHistory
from helper.tree_verification import pack_tree,trace_verified_path,select_confidence_nodes,PackedTree,VerifiedPath
from helper.sampling import build_sampling_probs,sample_from_probs,sample_target_from_logits
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_static_cache import OPDStaticCache,persistent_cache,swap_remove_plan
from helper.opd_attention import AttentionWorkspace
from helper.opd_scheduling import schedule,compact_suffix_inplace

class Cache:
    def __init__(self):
        self.layers = []

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2] if self.layers else 0

    def crop(self, length):
        for layer in self.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(repeats, 0)
            layer.values = layer.values.repeat_interleave(repeats, 0)

class TinyModel:
    is_eagle3_specforge = True
    device, dtype, compact_vocab_size = torch.device("cpu"), torch.bfloat16, 17

    def __init__(self):
        generator = torch.Generator().manual_seed(121)
        self.embedding = torch.randn(17, 8, generator=generator).bfloat16()
        self.target_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        self.draft_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        with torch.no_grad():
            self.target_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
            self.draft_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
        self.calls, self.masks = 0, []
        self.target_model = SimpleNamespace(device=self.device, dtype=self.dtype,
                                           model=self.target_forward, lm_head=self.target_head)

    def target_forward(self, input_ids, attention_mask, past_key_values, **kwargs):
        self.calls += 1
        self.masks.append(attention_mask.clone())
        hidden = self.embedding[input_ids]
        keys = hidden.unsqueeze(1)
        if isinstance(past_key_values,OPDStaticCache):
            keys,_=past_key_values.update(keys,keys.clone(),0)
        elif past_key_values.layers:
            keys = torch.cat((past_key_values.layers[0].keys, keys), -2)
        visible = (attention_mask[:, 0] == 0).to(self.dtype)
        # Attention depends on ancestry/history, not just current token.
        hidden = hidden + visible.matmul(keys[:, 0]) / visible.sum(-1, keepdim=True).clamp_min(1)
        if not isinstance(past_key_values,OPDStaticCache):
            past_key_values.layers = [SimpleNamespace(keys=keys, values=keys.clone())]
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)

    def __call__(self, hidden_states, input_ids, past_key_values=None, **kwargs):
        hidden = (hidden_states + self.embedding[input_ids]) * .5
        keys = hidden.unsqueeze(1)
        if isinstance(past_key_values,OPDStaticCache):
            keys,_=past_key_values.update(keys,keys.clone(),0)
        elif past_key_values is not None:
            keys = torch.cat((past_key_values[0][0], keys), -2)
        return dict(hidden_states=hidden, next_feature_states=hidden,
                    past_key_values=past_key_values if isinstance(past_key_values,OPDStaticCache) else [(keys,keys.clone())])

    def compute_compact_logits(self, hidden):
        return self.draft_head(hidden)

    def compact_to_target_ids(self, device):
        return torch.arange(17, device=device)

class CountModel(TinyModel):
    def __init__(self,device='cpu'):
        super().__init__()
        self.device=torch.device(device);self.target_model.device=self.device
        self.embedding=self.embedding.to(device);self.target_head.to(device);self.draft_head.to(device)
        self.mapping=torch.arange(17,device=device);self.draft_calls=0
    def __call__(self,*args,**kwargs):
        self.draft_calls+=1
        return super().__call__(*args,**kwargs)
    def compact_to_target_ids(self,device=None):return self.mapping

def load_rollout(device='cpu',history_type=None,source_path=None):
    historical_history=history_type or RolloutHistory
    opd_history=history_type or ContiguousRolloutHistory
    if not hasattr(opd_history,'finalize'):
        parent=opd_history
        class HistoryAdapter(parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.results={}
                self.count=next(iter(args[0].values())).shape[0]*kwargs.get('repeats',1)
            def append(self,rows,chunks,owners=None):
                if hasattr(self,'active') and self.active!=rows:
                    # Test concat oracle formerly assumed stable physical rows.
                    # Preserve each response's history under swap-remove order.
                    assert set(self.active)==set(rows)
                    order=[self.active.index(original) for original in rows]
                    for name,tensor in self.tensors.items():self.tensors[name]=tensor[order]
                    self.active=list(rows)
                super().append(rows,chunks)
            def mark_finished(self,row):self.results[row]=super().finish(row)
            def finalize(self):
                for row in range(self.count):
                    if row not in self.results:self.mark_finished(row)
                return [self.results[row] for row in range(self.count)]
        opd_history=HistoryAdapter
    path=Path(source_path) if source_path is not None else Path(__file__).resolve().parents[1]/'helper/specualtive_generate.py'
    tree=ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=='get_attention_mask':
            n.args.defaults[1]=ast.Constant(device)
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    scope=dict(torch=torch,time=time,math=math,os=os,deepcopy=deepcopy,DynamicCache=Cache,
               persistent_cache=persistent_cache,swap_remove_plan=swap_remove_plan,PackedTree=PackedTree,VerifiedPath=VerifiedPath,
               AttentionWorkspace=AttentionWorkspace,OPDStaticCache=OPDStaticCache,OPDReflex=OPDReflex,FastGRPORuntime=FastGRPORuntime,resolve_method=resolve_method,RolloutHistory=opd_history,
               OPD_COUNTER_NAMES=OPD_COUNTER_NAMES,historical_generate=load_historical(device,historical_history),
               schedule=schedule,compact_suffix_inplace=compact_suffix_inplace,
               pack_tree=pack_tree,trace_verified_path=trace_verified_path,
               select_confidence_nodes=select_confidence_nodes,
               build_sampling_probs=build_sampling_probs,sample_from_probs=sample_from_probs,
               sample_target_from_logits=sample_target_from_logits,sample_target_with_metadata=sample_target_with_metadata)
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns,type_ignores=[])),str(path),'exec'),scope)
    generate=scope['speculative_generate'];generate._test_scope=scope
    return generate


def load_historical(device='cpu',history_type=RolloutHistory):
    path=Path(__file__).resolve().parents[1]/'helper/historical_fastgrpo.py'
    tree=ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=='get_attention_mask':n.args.defaults[1]=ast.Constant(device)
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    scope=dict(torch=torch,time=time,math=math,deepcopy=deepcopy,DynamicCache=Cache,
               ThreadPoolExecutor=ThreadPoolExecutor,RolloutHistory=history_type,
               build_sampling_probs=build_sampling_probs,sample_from_probs=sample_from_probs,
               sample_target_from_logits=sample_target_from_logits)
    if device=='cpu':
        class TorchProxy:
            cuda=SimpleNamespace(Stream=lambda device:object(),set_device=lambda device:None,
                                 stream=lambda stream:nullcontext(),synchronize=torch.cuda.synchronize)
            def __getattr__(self,name):return getattr(torch,name)
        scope['torch']=TorchProxy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns,type_ignores=[])),str(path),'exec'),scope)
    return scope['speculative_generate']
