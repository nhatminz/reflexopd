#!/usr/bin/env python3
"""Frozen-checkpoint end-to-end OPD sweep. Never train policy/draft or fake AAL."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('target-model','draft-checkpoint','draft-config','vocab-mapping','dataset-path','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--target-adapter',default='')
    for key,value in (('batch-size',8),('responses',8),('max-length',512),('max-prompt-length',256),
                      ('verification-capacity',512),('max-verification-num',512),('max-draft-k',8),
                      ('max-draft-length',5),('min-draft-length',3),('rank',8),('topk',16),
                      ('iterations',2),('warmup',1)):
        p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--fast-lrs',default='0.001,0.01,0.05,0.1')
    p.add_argument('--streams',default='0,1');p.add_argument('--seeds',default='42,43')
    p.add_argument('--visited-weight',type=float,default=1.)
    p.add_argument('--frontier-weight',type=float,default=1.)
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--top-k',type=int,default=0);p.add_argument('--draft-length-c',type=float,default=.75)
    p.add_argument('--attn-implementation',default='sdpa');p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--profile',action='store_true',help='Separate same-seed replay, excluded from wall/throughput')
    p.add_argument('--diagnostics',action='store_true',help='Opt-in end-rollout B norm; excluded by default')
    p.add_argument('--gpu-utilization',action='store_true',help='Benchmark-only 0.5s nvidia-smi sampling; may perturb host timing')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--include-previous-opd',action='store_true',help='Also run frozen pre-memory-optimization rollout with same policy/draft/proposal settings')
    a=p.parse_args(argv)
    a.lr_values=[float(x) for x in a.fast_lrs.split(',')];a.stream_values=[int(x) for x in a.streams.split(',')]
    a.seed_values=[int(x) for x in a.seeds.split(',')]
    if any(x<0 for x in a.lr_values) or not set(a.stream_values)<={0,1} or not a.seed_values:
        p.error('LR>=0, streams=0/1, nonempty seeds required')
    if min(a.batch_size,a.responses,a.iterations,a.max_length,a.max_prompt_length,a.rank,a.topk)<=0 or a.topk<a.max_draft_k:
        p.error('positive sizes and OPD_TOPK>=max_draft_k required')
    if a.warmup<0:p.error('warmup cannot be negative')
    return a


def collator(tokenizer,max_prompt_length):
    source=ROOT/'grpo_speculative.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TrainDataCollator')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    return scope['TrainDataCollator'](tokenizer,max_prompt_length)


def release_runtime_cache(model):
    # Benchmark boundaries only. Don't charge a previous mode's retained pools.
    for name in ('_opd_runtime_cache','_fastgrpo_runtime','_opd_tree_mask_workspace','_opd_attention_workspace','_opd_padding_workspace','_opd_kv_scratch','_opd_target_kv_pool','_opd_draft_kv_pool'):
        if hasattr(model,name):delattr(model,name)


def summarize(rows):
    tokens=sum(r['generated_tokens'] for r in rows);wall=sum(r['generation_wall_s'] for r in rows)
    rounds=sum(r['verification_rounds'] for r in rows);accepted=sum(r['accepted_length_sum'] for r in rows)
    proposed=sum(r['proposed_draft_tokens'] for r in rows)
    result=dict(aal=accepted/rounds if rounds else 0.,verification_rounds=rounds,
        draft_acceptance_rate=sum(r['accepted_draft_tokens'] for r in rows)/proposed if proposed else 0.,
        generated_tokens=tokens,generation_wall_s=wall,tokens_per_s=tokens/wall if wall else 0.,
        median_wall_s=statistics.median(r['generation_wall_s'] for r in rows),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows))
    for name,denominator in (('opd_kl','opd_state_weight'),('opd_mean_union_size','opd_selected_states'),
            ('opd_target_compact_mass','opd_selected_states'),('opd_target_mass_in_draft_top16','opd_selected_states'),
            ('opd_active_token_rows','opd_rounds')):
        raw={'opd_kl':'opd_kl_sum','opd_mean_union_size':'opd_union_size_sum',
            'opd_target_compact_mass':'opd_compact_mass_sum','opd_target_mass_in_draft_top16':'opd_draft_topk_target_mass_sum',
            'opd_active_token_rows':'opd_active_rows_sum'}[name]
        d=sum(r.get(denominator,0.) for r in rows)
        result[name]=sum(r.get(raw,0.) for r in rows)/d if d else None
    for name in ('opd_selected_states','opd_visited_states','opd_frontier_states','opd_updates','opd_invalid_states',
                 'opd_proposal_mode_sparse_rounds','opd_proposal_mode_dense_rounds','opd_proposal_mode_fused_rounds','opd_proposal_mode_gemm_rounds'):
        result[name]=sum(r.get(name,0.) for r in rows)
    result['opd_nonfinite_kl_states']=sum(r.get('opd_nonfinite_kl_states',0.) for r in rows)
    result['opd_active_rows_mean']=result['opd_active_token_rows']
    result['opd_active_rows_max']=max(r.get('opd_active_rows_max',0.) for r in rows)
    result['opd_sparse_rounds']=result['opd_proposal_mode_sparse_rounds']
    result['opd_dense_rounds']=result['opd_proposal_mode_dense_rounds']
    for mode in ('fused','gemm'):result['opd_'+mode+'_rounds']=result['opd_proposal_mode_'+mode+'_rounds']
    host_syncs=sum(r.get('opd_host_syncs',0) for r in rows)
    batches=sum(r.get('batch_verification_rounds',0) for r in rows)
    result['host_syncs_per_round']=host_syncs/batches if batches and all('opd_host_syncs' in r for r in rows) else None
    result['kv_cache_bytes']=max(r.get('observed_kv_cache_bytes',0) for r in rows)
    for field in ('full_kv_reallocations','full_history_copies','full_history_copy_bytes','row_compactions','row_compaction_bytes','kv_rows_moved','kv_history_copy_bytes','pool_allocations'):
        result[field]=(sum(r.get('opd_target_'+field,0)+r.get('opd_draft_'+field,0) for r in rows)
                       if all('opd_target_'+field in r or 'opd_draft_'+field in r for r in rows) else None)
    if result['full_kv_reallocations'] is not None:
        result['kv_rows_moved']=sum(max(r.get('opd_target_kv_rows_moved',0),r.get('opd_draft_kv_rows_moved',0)) for r in rows)
        result['kv_reallocations_per_iter']=result['full_kv_reallocations']/len(rows)
        result['kv_pool_allocations_per_iter']=result['pool_allocations']/len(rows) if result['pool_allocations'] is not None else None
        result['kv_history_copy_bytes_per_iter']=result['kv_history_copy_bytes']/len(rows) if result['kv_history_copy_bytes'] is not None else None
        result['kv_rows_moved_per_iter']=result['kv_rows_moved']/len(rows)
    else:
        result.update(kv_reallocations_per_iter=None,kv_pool_allocations_per_iter=None,kv_history_copy_bytes_per_iter=None,kv_rows_moved_per_iter=None)
    measured_compactions=[r['kv_compaction_profile_ms'] for r in rows
                          if r.get('kv_compaction_profile_ms') is not None]
    result['kv_compaction_profile_ms']=sum(measured_compactions) if measured_compactions else None
    if result['opd_nonfinite_kl_states']:result['opd_kl']=None
    return result


def benchmark(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from helper.eagle3_specforge import Eagle3FastGRPOAdapter
    from helper.get_QAs import get_QAs_from_path
    from helper.specualtive_generate import speculative_generate
    previous_generate=None
    if args.include_previous_opd:
        import importlib.util
        spec=importlib.util.spec_from_file_location('_opd_previous_rollout',ROOT/'tests/oracles/opd_rollout_before_memory.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        previous_generate=module.speculative_generate
    from helper.opd_reflex import OPD_COUNTER_NAMES
    from scripts.gpu_utilization import GPUUtilization
    if not torch.cuda.is_available():raise RuntimeError('real CUDA + configured pretrained checkpoints required')
    dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
    torch.cuda.set_device(0);torch.manual_seed(args.seed_values[0])
    target=AutoModelForCausalLM.from_pretrained(args.target_model,torch_dtype=dtype,
        attn_implementation=args.attn_implementation,local_files_only=True).cuda().eval()
    model=Eagle3FastGRPOAdapter(target,args.draft_config,args.draft_checkpoint,args.vocab_mapping,
                              opd_rank=args.rank).cuda().eval()
    if args.target_adapter:
        from peft import PeftModel
        model.target_model=PeftModel.from_pretrained(model.target_model,args.target_adapter).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.target_model,padding_side='left',local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if target.config.model_type=='llama':tokenizer.pad_token,tokenizer.pad_token_id='<|end_of_text|>',128001
    dataset=get_QAs_from_path(args.dataset_path,'train')
    required=args.batch_size*args.iterations
    if len(dataset)<required:raise ValueError(f'need {required} ordered prompts; found {len(dataset)}')
    make_batch=collator(tokenizer,args.max_prompt_length)
    batches=[]
    for offset in range(0,required,args.batch_size):
        batch=make_batch(dataset[offset:offset+args.batch_size])
        if batch['input_ids'].shape[1]>=args.max_length:raise ValueError('prompt exhausts total max_length')
        batches.append({k:batch[k].cuda() for k in ('input_ids','attention_mask')})
    count={'target':0,'draft':0}
    observed_kv={'target':0,'draft':0}
    def observe_cache(kind,output):
        cache=output.get('past_key_values') if isinstance(output,dict) else getattr(output,'past_key_values',None)
        if cache is None:return
        if hasattr(cache,'statistics'):
            size=cache.statistics()['kv_cache_bytes']
        else:
            layers=((layer.keys,layer.values) for layer in cache.layers) if hasattr(cache,'layers') else iter(cache)
            size=sum(t.untyped_storage().nbytes() for layer in layers for t in layer if t is not None)
        observed_kv[kind]=max(observed_kv[kind],size)
    target_hook=target.model.register_forward_pre_hook(lambda *unused:count.__setitem__('target',count['target']+1))
    draft_hook=model.register_forward_pre_hook(lambda *unused:count.__setitem__('draft',count['draft']+1))
    target_cache_hook=target.model.register_forward_hook(lambda module,args,out:observe_cache('target',out))
    draft_cache_hook=model.register_forward_hook(lambda module,args,out:observe_cache('draft',out))
    base=dict(do_sample=True,repeated_generate_nums=args.responses,max_length=args.max_length,
        temperature=args.temperature,top_p=args.top_p,top_k=args.top_k or None,
        verification_capacity=args.verification_capacity,max_verification_num=args.max_verification_num,
        max_draft_token_length=args.max_draft_length,min_draft_token_length=args.min_draft_length,
        draft_token_length_c=args.draft_length_c,max_draft_k=args.max_draft_k,opd_rank=args.rank,opd_topk=args.topk,
        opd_visited_weight=args.visited_weight,opd_frontier_weight=args.frontier_weight,
        opd_profile=False,opd_diagnostics=args.diagnostics,statistical_time=False)
    def run(method,lr,stream,batch,seed):
        torch.manual_seed(seed);before=count.copy()
        observed_kv.update(target=0,draft=0)
        generator=previous_generate if method=='opd_reflex_previous' else speculative_generate
        out=generator(model,batch['input_ids'],batch['attention_mask'],tokenizer,
            method='opd_reflex' if method=='opd_reflex_previous' else method,opd_fast_lr=lr,opd_update_stream=bool(stream),**base)
        return out,{k:count[k]-before[k] for k in count}
    configurations=[('fastgrpo',0.,0)]+[(method,lr,s) for lr in args.lr_values for s in args.stream_values
        for method in (('opd_reflex_previous','opd_reflex') if args.include_previous_opd else ('opd_reflex',))]
    reports=[];response_file=Path(args.output)/'responses.jsonl'
    try:
        off,c0=run('fastgrpo',0.,0,batches[0],args.seed_values[0])
        empty,c1=run('opd_reflex',0.,1,batches[0],args.seed_values[0])
        # OPD Top16 ties may differ from historical TopK(draft_k), by explicit
        # user choice. Never slow OPD merely to match historical candidates.
        if c0['target']!=1+off['batch_verification_rounds'] or c1['target']!=1+empty['batch_verification_rounds']:
            raise AssertionError('extra target forward; discard benchmark')
        for method,lr,stream in configurations:
            torch.cuda.synchronize();release_runtime_cache(model)
            # Warm exactly the measured seed/prompt schedule: compaction gives
            # shape-specialized kernels, so warming only one seed hides JIT cost.
            for _ in range(args.warmup):
                for seed in args.seed_values:
                    for i,batch in enumerate(batches):run(method,lr,stream,batch,seed+i)
            rows=[];measured_total_begin=time.perf_counter()
            utilization=GPUUtilization(args.gpu_utilization).start()
            for seed in args.seed_values:
                for i,batch in enumerate(batches):
                    actual_seed=seed+i
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    output,forwards=run(method,lr,stream,batch,actual_seed)
                    torch.cuda.synchronize();wall=time.perf_counter()-started
                    allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
                    if forwards['target']!=1+output['batch_verification_rounds']:raise AssertionError('extra target forward')
                    row=dict(method=method,fast_lr=lr,update_stream=stream,seed=actual_seed,prompt_offset=i*args.batch_size,
                        generation_wall_s=wall,generated_tokens=sum(output['response_generated_tokens']),
                        accepted_length_sum=output['total_acc_length'],verification_rounds=output['total_decoded_token_num'],
                        accepted_draft_tokens=output['total_accepted_draft_tokens'],proposed_draft_tokens=output['total_proposed_draft_tokens'],
                        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,**forwards,
                        **{k:output.get(k,0.) for k in OPD_COUNTER_NAMES})
                    row.update({k:v for k,v in output.items() if k.startswith('opd_final_')})
                    row['batch_verification_rounds']=output['batch_verification_rounds']
                    row['observed_kv_cache_bytes']=sum(observed_kv.values())
                    for name,value in output.items():
                        if isinstance(value,(int,float)) and (name.startswith(('opd_target_','opd_draft_','opd_host_sync','opd_attention_'))):row[name]=value
                    if args.profile:
                        # Diagnostic replay does not evict warmed non-profile pools.
                        cache=getattr(model,'_opd_runtime_cache',None)
                        model._opd_runtime_cache={};base['opd_profile']=True
                        try:profile,_=run(method,lr,stream,batch,actual_seed)
                        finally:base['opd_profile']=False;model._opd_runtime_cache=cache
                        if profile['generated_token_ids']!=output['generated_token_ids']:
                            raise AssertionError('profiling changed output; discard timings')
                        row['profile_sections_ms']=profile['opd_profile_sections_ms']
                        sections=profile.get('opd_profile_sections_ms') or {}
                        row['kv_compaction_profile_ms']=(sum(sections.get(key,0) for key in ('kv_batch_compaction_ms','kv_suffix_compaction_ms'))
                            if method.startswith('opd_reflex') else None)
                    with response_file.open('a') as f:
                        for response,(tokens,a,n) in enumerate(zip(output['generated_token_ids'],output['response_accepted_length_sum'],output['response_verification_rounds'])):
                            f.write(json.dumps(dict(method=method,fast_lr=lr,update_stream=stream,seed=actual_seed,
                                prompt_index=i*args.batch_size+response//args.responses,response_index=response%args.responses,
                                accepted_length_sum=a,verification_rounds=n,generated_tokens=len(tokens),
                                token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))+'\n')
                    rows.append(row)
            result=dict(method=method,fast_lr=lr,update_stream=stream,**summarize(rows),rollouts=rows,
                measured_total_wall_s=time.perf_counter()-measured_total_begin,**utilization.finish())
            reports.append(result)
        baseline=reports[0]
        for result in reports:
            result['delta_aal']=result['aal']-baseline['aal']
            result['relative_generation_overhead_percent']=100*(result['generation_wall_s']/baseline['generation_wall_s']-1)
            prior=next((x for x in reports if x['method']=='opd_reflex_previous' and x['fast_lr']==result['fast_lr'] and x['update_stream']==result['update_stream']),None)
            result['delta_aal_vs_previous_opd']=result['aal']-prior['aal'] if prior and result['method']=='opd_reflex' else None
            result['tokens_per_s_ratio_vs_previous_opd']=result['tokens_per_s']/prior['tokens_per_s'] if prior and result['method']=='opd_reflex' else None
        # Keep all results, including negative AAL/cost > benefit. Throughput wins
        # are not automatically called AAL improvements or production defaults.
        current=[x for x in reports if x['method']=='opd_reflex']
        best=max(current,key=lambda x:x['tokens_per_s'])
        goal_candidates=[x for x in current if x['delta_aal']>0 and x['tokens_per_s']>=baseline['tokens_per_s']]
        recommendation=max(goal_candidates,key=lambda x:x['tokens_per_s']) if goal_candidates else None
        payload=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,config=vars(args),
            proposal_mode=os.environ.get('OPD_PROPOSAL_MODE','auto'),
            proposal_profile=os.environ.get('OPD_PROPOSAL_PROFILE',''),
            dense_implementation=os.environ.get('OPD_DENSE_IMPLEMENTATION','auto'),
            gpu_utilization_note='Per-configuration nvidia-smi samples when --gpu-utilization is set; includes diagnostic replay if requested.',
            baseline='FastGRPO shared optimized infrastructure; historical native torch.topk(draft_k) semantics',
            cold_invariant='B=0 leaves raw logits/distribution unchanged; Top16 tie IDs need not equal historical K',reports=reports,
            fastest_observed=dict(method=best['method'],fast_lr=best['fast_lr'],update_stream=best['update_stream'],
                delta_aal=best['delta_aal'],tokens_per_s=best['tokens_per_s']),
            recommendation=None if recommendation is None else dict(fast_lr=recommendation['fast_lr'],
                update_stream=recommendation['update_stream'],delta_aal=recommendation['delta_aal']),
            note='Fair shared cache/memory/tree infrastructure. FastGRPO raw native TopK; OPD correction only. Frozen policy/draft/A, no persistent training or A gradient accumulation in evaluation. Profiling replay excluded from wall. Same seed does not imply same responses.')
        return payload
    finally:
        target_hook.remove();draft_hook.remove()
        target_cache_hook.remove();draft_cache_hook.remove()


def main():
    args=parse_args()
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    output=Path(args.output)
    if output.exists():raise FileExistsError('use a NEW benchmark output directory')
    for path in (Path(args.target_model)/'config.json',Path(args.draft_config),Path(args.vocab_mapping),Path(args.dataset_path)):
        if not path.is_file():raise FileNotFoundError(path)
    if not Path(args.draft_checkpoint).exists():raise FileNotFoundError(args.draft_checkpoint)
    output.mkdir(parents=True)
    result=benchmark(args)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    flat=[{k:v for k,v in x.items() if k!='rollouts'} for x in result['reports']]
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    fastest=result['fastest_observed']
    (output/'fastest_observed.env').write_text(f"# NOT an automatic recommendation. Measured subset delta AAL={fastest['delta_aal']}. Validate held-out prompts.\nexport OPD_FAST_LR={fastest['fast_lr']}\nexport OPD_UPDATE_STREAM={fastest['update_stream']}\n")
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))


if __name__=='__main__':main()
