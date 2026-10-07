#!/usr/bin/env python3
"""Real CUDA KV microbenchmark, old fixed-pool reference vs growable OPD.

No model download/forward/training. Reports actual allocated KV bytes, peak
VRAM, append/suffix/finished-row compaction separately. CUDA event synchronization
belongs ONLY to this measurement script. OOM is reported, never replaced by data.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class FixedPoolReference:
    """Previous cache behavior, benchmark-only: reserve full bound/reallocate rows."""
    def __init__(self,capacity):
        self.capacity=capacity;self.layers=[]
        self.copies=self.copy_bytes=self.reallocations=0
    def update(self,k,v,layer):
        if layer==len(self.layers):
            shape=(*k.shape[:-2],self.capacity,k.shape[-1])
            self.layers.append([k.new_empty(shape),v.new_empty(shape),0])
        row=self.layers[layer];end=row[2]+k.shape[-2]
        row[0][...,row[2]:end,:].copy_(k);row[1][...,row[2]:end,:].copy_(v);row[2]=end
        return self[layer]
    def __getitem__(self,i):return tuple(t[...,:self.layers[i][2],:] for t in self.layers[i][:2])
    def crop(self,length):
        for row in self.layers:row[2]=min(row[2],length)
    def batch_select_indices(self,indices):
        for row in self.layers:
            for j in range(2):
                live=row[j][...,:row[2],:].index_select(0,indices)
                pool=live.new_empty((*live.shape[:-2],self.capacity,live.shape[-1]))
                pool[...,:row[2],:].copy_(live);row[j]=pool
                self.copy_bytes+=live.numel()*live.element_size()
            self.reallocations+=1;self.copies+=1
    def statistics(self):
        return dict(kv_cache_bytes=sum(t.numel()*t.element_size() for row in self.layers for t in row[:2]),
            kv_compaction_workspace_bytes=0,full_kv_reallocations=self.reallocations,
            full_history_copies=self.copies,full_history_copy_bytes=self.copy_bytes)


def measure(fn,torch,iterations):
    samples=[]
    for _ in range(iterations):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--config',help='Actual target/draft config.json; infer layers/KV heads/head dim')
    p.add_argument('--lengths',default='256,512,1024,2048');p.add_argument('--batches',default='16,32,64')
    p.add_argument('--heads',type=int,default=4);p.add_argument('--dim',type=int,default=64)
    p.add_argument('--layers',type=int,default=None)
    p.add_argument('--dtype',choices=['bf16','fp16','fp32'],default='bf16')
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--initial-capacity',type=int,default=256)
    p.add_argument('--prompt-length',type=int,default=128);p.add_argument('--max-draft-length',type=int,default=5)
    p.add_argument('--verification-capacity',type=int,default=512);p.add_argument('--max-draft-k',type=int,default=8)
    a=p.parse_args(argv);output=Path(a.output)
    if output.exists():p.error('choose a new output directory')
    if a.config:
        config=json.loads(Path(a.config).read_text())
        a.heads=int(config.get('num_key_value_heads',config['num_attention_heads']))
        a.dim=int(config.get('head_dim') or config['hidden_size']//config['num_attention_heads'])
        if a.layers is None:a.layers=int(config['num_hidden_layers'])
    if a.layers is None:a.layers=1
    lengths=[int(x) for x in a.lengths.split(',')];batches=[int(x) for x in a.batches.split(',')]
    if min(lengths+batches+[a.layers,a.heads,a.dim,a.iterations,a.initial_capacity])<1:p.error('positive shapes required')
    import torch
    from types import SimpleNamespace
    from helper.opd_static_cache import OPDStaticCache
    from helper.opd_scheduling import compact_suffix_inplace
    if not torch.cuda.is_available():p.error('actual CUDA required; no synthetic timings')
    torch.manual_seed(42);dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[a.dtype]
    rows=[]
    for batch in batches:
        for length in lengths:
            for method in ('previous_fixed_pool','growable_opd'):
                cache=None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    legacy_capacity=a.prompt_length+length*(a.max_draft_length+1)+a.verification_capacity+a.max_draft_k*a.max_draft_length
                    cache=(FixedPoolReference(legacy_capacity) if method=='previous_fixed_pool' else
                           OPDStaticCache(a.initial_capacity,batch_capacity=batch))
                    prefix=min(a.prompt_length,length)
                    inputs=torch.randn(batch,a.heads,prefix,a.dim,device='cuda',dtype=dtype)
                    suffix=torch.randn(batch,a.heads,8,a.dim,device='cuda',dtype=dtype)
                    for layer in range(a.layers):cache.update(inputs,inputs,layer)
                    position=prefix
                    while position<length:
                        width=min(8,length-position)
                        for layer in range(a.layers):cache.update(suffix[...,:width,:],suffix[...,:width,:],layer)
                        position+=width
                    stats_before=cache.statistics()
                    def append():
                        cache.crop(length)
                        for layer in range(a.layers):cache.update(suffix,suffix,layer)
                    append_ms=measure(append,torch,a.iterations)
                    keep=torch.arange(batch//2,device='cuda')*2
                    # Warm compaction on a permutation retaining the same batch.
                    permutation=torch.arange(batch-1,-1,-1,device='cuda')
                    for _ in range(2):cache.batch_select_indices(permutation)
                    remap_ms=measure(lambda:cache.batch_select_indices(permutation),torch,a.iterations)
                    reallocations_before_finish=cache.statistics()['full_kv_reallocations']
                    if method=='growable_opd':
                        # Warm the smaller batch bucket BEFORE timing, without
                        # changing logical cache batch size used by this test.
                        from helper.opd_kv_kernels import remap_rows
                        for layer in cache.layers:
                            scratch=layer.owner._scratch[('cuda_live',layer.key_pool.dtype,layer.key_pool.device)]
                            remap_rows(layer.key_pool,layer.value_pool,keep,layer.length,scratch)
                        torch.cuda.synchronize()
                    finish_ms=measure(lambda:cache.batch_select_indices(keep),torch,1)
                    finish_reallocations=cache.statistics()['full_kv_reallocations']-reallocations_before_finish
                    if method=='growable_opd' and finish_reallocations:
                        raise AssertionError('finished-row compaction reallocated KV pool')
                    b=keep.numel();chosen=(length+torch.tensor([0,2,5,7],device='cuda')).expand(b,-1)
                    model=SimpleNamespace(_opd_initial_batch=batch,_opd_max_path_capacity=8)
                    def compact_suffix():
                        for layer in range(a.layers):
                            k,v=cache[layer];compact_suffix_inplace(k,v,chosen,length,1,model)
                    suffix_ms=measure(compact_suffix,torch,a.iterations)
                    row=dict(method=method,batch=batch,sequence_length=length,layers=a.layers,heads=a.heads,dim=a.dim,
                        allocated_kv_before_finish_bytes=stats_before['kv_cache_bytes'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        append_ms=append_ms,batch_permutation_ms=remap_ms,finished_row_compaction_ms=finish_ms,
                        accepted_suffix_compaction_ms=suffix_ms,finish_pool_reallocations=finish_reallocations,
                        **cache.statistics(),status='ok')
                    rows.append(row)
                except torch.cuda.OutOfMemoryError as exc:
                    rows.append(dict(method=method,batch=batch,sequence_length=length,status='oom',error=str(exc)))
                finally:
                    cache=None
                    layer=None;scratch=None
                    # Benchmark boundary only: release completed cases, not production.
                    for name in ('inputs','suffix','model','keep','chosen','permutation'):
                        if name in locals():
                            if name=='inputs':inputs=None
                            elif name=='suffix':suffix=None
                            elif name=='model':model=None
                    torch.cuda.empty_cache()
    output.mkdir(parents=True)
    payload=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        config=a.config,dtype=a.dtype,rows=rows,note='KV component only; no generation/AAL claim. Previous fixed-pool reference reproduces old reserve and live-prefix gather/reallocation.')
    (output/'report.json').write_text(json.dumps(payload,indent=2)+'\n')
    fields=sorted({k for row in rows for k in row})
    with (output/'kv.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
