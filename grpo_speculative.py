import os
import sys
import random
import time
from pathlib import Path

process_started_at=time.perf_counter()

REPO_ROOT = Path(__file__).resolve().parent
# Keep this checkout ahead of parent/PYTHONPATH packages named "helper", even
# when torchrun has already inserted the repository further down sys.path.
repo_import_path = str(REPO_ROOT)
if repo_import_path in sys.path:
    sys.path.remove(repo_import_path)
sys.path.insert(0, repo_import_path)

import pandas as pd
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,GenerationConfig
from helper.rewards import accuracy_reward_func , format_reward_func
from helper.get_QAs import get_test_QAs , get_train_QAs, get_QAs_from_path, select_train_subset
from helper.specualtive_generate import speculative_generate
from helper.eagle3_specforge import Eagle3FastGRPOAdapter, rollout_tensor_for_training
from helper.checkpointing import capture_rng_state, restore_rng_state
from helper.method_config import resolve_method
from helper.opd_reflex import OPD_COUNTER_NAMES, GENERATION_COUNTER_NAMES
from helper.step_metrics import PhaseTimings, StepMetricsWriter, completed_step_snapshot
from helper.rollout_metrics import RolloutMetricsWriter
from helper.opd_optimizer import draft_optimizer,load_draft_optimizer
from policy_lag_analysis import (
    BranchSummary,
    bootstrap_delta_by_prompt,
    state_digest,
    teacher_shift_tv,
    weighted_aal,
    write_results,
)
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch import nn
import time
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
import signal
import torch
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None
from datetime import datetime
import argparse 
from statistics import mean , stdev
import pickle
import csv
import importlib.util
from tqdm.auto import tqdm
from helper.drift_metrics import sparse_union_metrics

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def _dtype_from_name(name):
    name = str(name or "auto").lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")


def _resolve_attn_implementation(requested):
    requested = str(requested or "")
    if not requested:
        return None
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed. Falling back to eager."
        )
        return "eager"
    return requested


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _prune_checkpoints(checkpoint_dir, keep_last):
    keep_last = int(keep_last or 0)
    if keep_last <= 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("step*.pt"), key=lambda p: p.stat().st_mtime)
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def _target_lora_state_dict(target_model):
    if get_peft_model_state_dict is not None:
        return get_peft_model_state_dict(target_model)
    return target_model.state_dict()


def _load_target_lora_state_dict(target_model, state_dict):
    if set_peft_model_state_dict is not None:
        set_peft_model_state_dict(target_model, state_dict)
    else:
        target_model.load_state_dict(state_dict, strict=False)


def _gradient_state(module):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _restore_gradient_state(module, state):
    parameters = dict(module.named_parameters())
    for name, value in (state or {}).items():
        if name not in parameters:
            raise ValueError(f"checkpoint gradient parameter is missing: {name}")
        parameters[name].grad = value.to(parameters[name].device, parameters[name].dtype)


def save_training_checkpoint(
    checkpoint_dir,
    *,
    model,
    optimizer_target,
    optimizer_draft,
    epoch,
    next_batch,
    step,
    used_items,
    draft_step,
    draft_accumulated_step,
    batch_data,
    keep_last,
    cumulative_elapsed_time_s,
):
    current_rank = dist.get_rank() if dist.is_initialized() else 0
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_state = {
        "rank": int(current_rank),
        "used_items": int(used_items),
        "batch_data": batch_data,
        "target_gradients": _gradient_state(model.target_model),
        "draft_gradients": _gradient_state(model.draft_model),
        "rng": capture_rng_state(),
        "cumulative_elapsed_time_s": float(cumulative_elapsed_time_s),
    }
    # Pending analytical gradients are rank-local until the existing optimizer
    # boundary. Never restore rank 0's pending feedback into every rollout worker.
    for key, attribute in (
        ('opd_projector_pending_sum', 'opd_projector_grad_sum'),
        ('opd_projector_pending_weight', 'opd_projector_grad_weight'),
    ):
        value = getattr(model, attribute, None)
        local_state[key] = None if value is None else value.detach().cpu().clone()
    if dist.is_initialized():
        rank_states = [None] * current_world_size
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states = [local_state]
    if current_rank != 0:
        return None
    checkpoint_dir = Path(checkpoint_dir)
    state = {
        "format": "opd_fastgrpo_checkpoint_v5",
        "world_size": int(current_world_size),
        "rank_states": rank_states,
        "cumulative_elapsed_time_s": max(
            float(item["cumulative_elapsed_time_s"]) for item in rank_states
        ),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "step": int(step),
        "used_items": int(used_items),
        "draft_step": int(draft_step),
        "draft_accumulated_step": int(draft_accumulated_step),
        "target_lora": _target_lora_state_dict(model.target_model),
        "draft_model": model.draft_model.state_dict(),
        "opd_projector": getattr(model,"opd_projector",None),
        "opd_projector_pending_sum": getattr(model,'opd_projector_grad_sum',None),
        "opd_projector_pending_weight": getattr(model,'opd_projector_grad_weight',None),
        "method": getattr(model,"_training_method","fastgrpo"),
        "optimizer_target": optimizer_target.state_dict(),
        "optimizer_draft": optimizer_draft.state_dict(),
        "scheduler_target": None,
        "scheduler_draft": None,
    }
    checkpoint_path = checkpoint_dir / f"step{int(step)}_epoch{int(epoch) + 1}_batch{int(next_batch)}.pt"
    _atomic_torch_save(state, checkpoint_path)
    _atomic_torch_save(state, checkpoint_dir / "latest.pt")
    _prune_checkpoints(checkpoint_dir, keep_last)
    print(f"Saved FastGRPO checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(path, *, model, optimizer_target, optimizer_draft):
    checkpoint = torch.load(path, map_location="cpu")
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    current_rank = dist.get_rank() if dist.is_initialized() else 0
    saved_world_size = int(checkpoint.get("world_size", 1))
    if saved_world_size != current_world_size:
        raise RuntimeError(
            "checkpoint world-size mismatch: "
            f"saved={saved_world_size}, current={current_world_size}; "
            "resume with the same number of ranks"
        )
    rank_states = checkpoint.get("rank_states")
    if rank_states is not None:
        if len(rank_states) != saved_world_size:
            raise RuntimeError("checkpoint rank_states does not match saved world_size")
        local_state = rank_states[current_rank]
        if int(local_state.get("rank", current_rank)) != current_rank:
            raise RuntimeError(f"checkpoint has no state for rank {current_rank}")
    else:
        local_state = {
            "used_items": checkpoint.get("used_items", 0),
            "batch_data": checkpoint.get("batch_data", {}),
            "target_gradients": checkpoint.get("target_gradients"),
            "draft_gradients": checkpoint.get("draft_gradients"),
            "rng": checkpoint.get("all_rng_states"),
        }
    if checkpoint.get("method","fastgrpo")!=getattr(model,"_training_method","fastgrpo"):
        raise ValueError("resume method mismatch; start a new run for replacement OPD")
    if hasattr(model, 'opd_projector') and 'opd_projector' not in checkpoint['draft_model']:
        raise ValueError(
            'resume checkpoint predates the learned draft projector; start a new '
            'run using its draft weights as initialization, not optimizer resume'
        )
    model.draft_model.load_state_dict(checkpoint["draft_model"])
    if checkpoint.get("opd_projector") is not None:model.load_opd_projector(checkpoint["opd_projector"])
    for key, attribute in (
        ('opd_projector_pending_sum', 'opd_projector_grad_sum'),
        ('opd_projector_pending_weight', 'opd_projector_grad_weight'),
    ):
        value = local_state.get(key, checkpoint.get(key))
        if value is not None:
            getattr(model, attribute).copy_(value)
    _load_target_lora_state_dict(model.target_model, checkpoint["target_lora"])
    optimizer_target.load_state_dict(checkpoint["optimizer_target"])
    load_draft_optimizer(optimizer_draft,checkpoint["optimizer_draft"],model.draft_model)
    _restore_gradient_state(model.target_model, local_state.get("target_gradients"))
    _restore_gradient_state(model.draft_model, local_state.get("draft_gradients"))
    if local_state.get("rng") is not None:
        restore_rng_state(local_state["rng"])
    checkpoint["used_items"] = int(local_state.get("used_items", 0))
    checkpoint["batch_data"] = local_state.get("batch_data", {})
    return checkpoint


parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument('--model_dir',type=str)
parser.add_argument('--adapter_path',type=str)
parser.add_argument('--draft_backend', type=str, default='legacy', choices=['legacy', 'eagle3'])
parser.add_argument('--draft_config', type=str, default='')
parser.add_argument('--draft_initialization_mode', type=str, default='pretrained', choices=['pretrained', 'random'])
parser.add_argument('--vocab_mapping', type=str, default='')
parser.add_argument('--eagle_feature_layers', type=str, default='',
                    help='Comma-separated decoder layer ids; empty uses the SpecForge EAGLE-3 rule.')
parser.add_argument('--eagle_ttt_length', type=int, default=7)
parser.add_argument('--eagle_lk_loss_type', type=str, default='', choices=['', 'lambda', 'alpha', 'tv'])
parser.add_argument('--eagle_kl_scale', type=float, default=1.0)
parser.add_argument('--eagle_kl_decay', type=float, default=1.0)
parser.add_argument('--method',default='fastgrpo',choices=['fastgrpo','opd_reflex'])
parser.add_argument('--opd_rank',type=int,default=8)
parser.add_argument('--opd_topk',type=int,default=16)
parser.add_argument('--opd_fast_lr',type=float,default=.01)
parser.add_argument('--opd_visited_weight',type=float,default=1.)
parser.add_argument('--opd_frontier_weight',type=float,default=1.)
parser.add_argument('--opd_update_stream',default='1',choices=['0','1'])
parser.add_argument('--opd_profile',default='0',choices=['0','1'])
parser.add_argument('--opd_diagnostics',default='0',choices=['0','1'])
parser.add_argument('--opd_backend',default='triton',choices=['auto','torch','triton'])
parser.add_argument('--opd_train_projector',default='1',choices=['0','1'])
parser.add_argument('--draft_train_mode', default='batched', choices=['batched', 'per_response'])
parser.add_argument('--draft_train_max_batch_size', type=int, default=8)
parser.add_argument('--draft_train_max_tokens', type=int, default=2048)
parser.add_argument('--draft_train_max_padding_ratio', type=float, default=1.25)
parser.add_argument('--draft_train_profile', default='0', choices=['0', '1'])
parser.add_argument('--kv_gather_strategy', default='stacked', choices=['stacked', 'per_layer'])
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
parser.add_argument('--attn_implementation', type=str, default='sdpa')
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--top_p',type=float,default=0.95)
parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps for target model')
parser.add_argument('--draft_accumulation_steps', type=int, default=1, help='Gradient accumulation steps for draft model')
parser.add_argument('--target_lr', type=float, default=1e-6, help='Learning rate for target model')
parser.add_argument('--draft_lr', type=float, default=1e-4, help='Learning rate for draft model')
parser.add_argument('--is_train_draft', type=lambda x: x.lower() == 'true', default=True, help='Whether to train the draft model (True/False)')
parser.add_argument('--model_type', type=str, default='Qwen2___5-Math-7B', help='Version name for saving checkpoints')
parser.add_argument('--train_option',type=str,default="simplelr_abel_level3to5")
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--eval_dataset_path', type=str, default='',
                    help='Optional held-out evaluation dataset. Defaults to dataset_path only when that path has a real eval split.')
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--eval_split', type=str, default='test')
parser.add_argument('--load_lora_path',type=str,default="")
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--version_name',type=str,default='normal')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--sample_num',type=int,default=100)
parser.add_argument('--train_data_fraction', type=float, default=0.4,
                    help='Fraction of the loaded train split to use. Applied to any train_option dataset.')
parser.add_argument('--train_subset_seed', type=int, default=42,
                    help='Seed for the deterministic train subset selection.')
parser.add_argument('--max_train_samples', type=int, default=0,
                    help='Optional hard cap after train_data_fraction, useful for smoke/debug runs.')
parser.add_argument('--grpo_iteration_num',type=int,default=1)
parser.add_argument('--repeated_generate_nums',type=int,default=8)
parser.add_argument('--beta',type=float,default=0.01)
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--max_length',type=int,default=2048)
parser.add_argument('--max_prompt_length', type=int, default=2048,
                    help='Maximum prompt tokens before rollout; kept separate from max generated sequence length.')
parser.add_argument('--max_training_padding_gap',type=int,default=256)
parser.add_argument('--max_training_token',type=int,default=3072)
parser.add_argument('--logps_chunk_size', type=int, default=256,
                    help='Sequence chunk size for token-logprob computation. Lower values reduce peak VRAM.')
parser.add_argument('--verification_capacity', type=int, default=160)
parser.add_argument('--max_draft_token_length', type=int, default=5)
parser.add_argument('--max_draft_k', type=int, default=8)
parser.add_argument('--max_verification_num', type=int, default=160)
parser.add_argument('--min_draft_token_length', type=int, default=3)
parser.add_argument('--draft_token_length_c', type=float, default=0.75)
parser.add_argument('--statistical_time', type=lambda x: x.lower() == 'true', default=False,
                    help='Collect detailed speculative timing counters. False avoids extra CUDA synchronizations.')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--persistent_workers', default=True)
parser.add_argument('--log_file', type=str, required=True,
                    help="Full path to training log file, e.g., /path/to/train.log")
parser.add_argument('--summary_file', type=str, default='',
                    help="Optional summary JSON path. Defaults to summary.json next to log_file.")
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help="Directory to save trained target adapter/model checkpoints")
parser.add_argument('--saved_draft_model_dir', type=str, required=True,
                    help="Directory to save trained draft model checkpoints")
parser.add_argument('--saved_statistics_dir', type=str, required=True,
                    help="Directory to save statistics of generated sequence lengths.")
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument('--timing_file', type=str, default='')
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--rollout_log_flush_interval', type=int, default=1)
parser.add_argument('--opd_projector_lr', type=float, default=None)
parser.add_argument('--save_checkpoint_steps', type=int, default=0)
parser.add_argument('--keep_last_checkpoints', type=int, default=3)
parser.add_argument('--resume_checkpoint', type=str, default='')
parser.add_argument('--append_log', default='',
                    help='Append JSONL explicitly. Empty preserves legacy behavior (append when resuming).')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--reset_rng_on_resume', default=False,
                    help='Reset RNGs to --seed after loading a checkpoint; use true for paired trace runs.')
parser.add_argument('--max_grpo_steps', type=int, default=0,
                    help='Stop after this many newly completed GRPO steps; 0 disables the trace stop.')
parser.add_argument('--drift_topk', type=int, default=16,
                    help='Per-distribution top-k used by the bidirectional sparse-union drift metric.')
parser.add_argument('--drift_temperature', type=float, default=1.0,
                    help='Temperature used for sparse TV and forward-KL logging.')
parser.add_argument('--drift_row_chunk_size', type=int, default=32,
                    help='Valid token rows per sparse-drift chunk; lower values reduce peak VRAM.')
parser.add_argument('--draft_lr_multiplier', type=float, default=1.0,
                    help='Explicit FastGRPO ablation multiplier applied after optimizer resume; 1.0 is the fair baseline.')
parser.add_argument('--policy_lag_output_dir', type=str, default='')
parser.add_argument('--analysis_boundaries', type=str, default='')
parser.add_argument('--analysis_interval', type=int, default=0)
parser.add_argument('--analysis_eval_prompts', type=int, default=8)
parser.add_argument('--analysis_seeds', type=str, default='11,29,47')
parser.add_argument('--analysis_training_token_budget', type=int, default=0)
parser.add_argument('--analysis_draft_update_steps', type=int, default=1)
parser.add_argument('--analysis_bootstrap_samples', type=int, default=2000)
parser.add_argument('--analysis_resume', default='true')
args = parser.parse_args()
method,_=resolve_method(args.method)
world_size = int(os.environ.get('WORLD_SIZE', '1'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
if world_size > 1:
    if not torch.cuda.is_available():
        raise RuntimeError('multi-GPU mode requires CUDA')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank() if dist.is_initialized() else 0
is_main_process = rank == 0


def _sync_gradients(module):
    if not dist.is_initialized():
        return
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _aggregate_job_metrics(
    data,device,cumulative_elapsed_time_s,*,
    include_opd_profile=False,
):
    """Aggregate cumulative counters only at log/final boundaries."""
    sum_names = (
        'total_rollout_tokens', 'total_acc_length', 'total_decoded_token_num',
        'total_accepted_draft_tokens', 'total_proposed_draft_tokens',
        'reward_sum', 'reward_count', 'target_loss_sum', 'target_loss_count',
        'draft_loss1_sum', 'draft_loss2_sum', 'draft_loss_count',
        'draft_sparse_tv_sum', 'draft_sparse_kl_sum', 'draft_sparse_count',
        'opd_updates',
        'trace_rollout_count', 'used_items', 'ignore_due_correct', 'ignore_due_incorrect',
    )
    sum_names += tuple(name for name in OPD_COUNTER_NAMES+GENERATION_COUNTER_NAMES if name not in ('opd_updates','opd_active_rows_max'))
    max_names = (
        'generate_time_cost', 'train_time_cost', 'draft_train_time_cost',
        'prefill_time_cost', 'target_time_cost', 'draft_time_cost',
        'check_time_cost',
        'opd_active_rows_max','opd_interval_active_rows_max',
    )
    if include_opd_profile:
        max_names += ('opd_profile_time_ms',)
    sums = torch.tensor(
        [float(data.get(name, 0.0)) for name in sum_names],
        device=device, dtype=torch.float64,
    )
    maxima = torch.tensor(
        [float(data.get(name, 0.0)) for name in max_names]
        + [float(cumulative_elapsed_time_s)],
        device=device, dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    result = dict(zip(sum_names, sums.tolist()))
    result.update(zip(max_names + ('cumulative_elapsed_time_s',), maxima.tolist()))
    rounds = max(result['total_decoded_token_num'], 1.0)
    proposed = max(result['total_proposed_draft_tokens'], 1.0)
    result['average_accept_length'] = result['total_acc_length'] / rounds
    result['accepted_tokens_per_medusa_step'] = (
        result['total_accepted_draft_tokens'] / rounds
    )
    result['draft_acceptance_rate'] = (
        result['total_accepted_draft_tokens'] / proposed
    )
    result['mean_reward'] = result['reward_sum'] / max(result['reward_count'], 1.0)
    result['target_loss'] = (
        result['target_loss_sum'] / max(result['target_loss_count'], 1.0)
    )
    result['draft_loss1'] = (
        result['draft_loss1_sum'] / max(result['draft_loss_count'], 1.0)
    )
    result['draft_loss2'] = (
        result['draft_loss2_sum'] / max(result['draft_loss_count'], 1.0)
    )
    result['draft_sparse_tv'] = (
        result['draft_sparse_tv_sum'] / max(result['draft_sparse_count'], 1.0)
    )
    result['draft_sparse_kl'] = (
        result['draft_sparse_kl_sum'] / max(result['draft_sparse_count'], 1.0)
    )
    result['tokens_per_s'] = (
        result['total_rollout_tokens'] /
        max(result['cumulative_elapsed_time_s'], 1.0e-9)
    )
    return result
num_epochs=args.num_epochs
sample_num=args.sample_num
train_data_fraction=args.train_data_fraction
train_subset_seed=args.train_subset_seed
max_train_samples=args.max_train_samples
grpo_iteration_num=args.grpo_iteration_num
repeated_generate_nums=args.repeated_generate_nums
beta=args.beta
epsilon=args.epsilon
max_length=args.max_length
max_prompt_length=args.max_prompt_length
max_training_padding_gap=args.max_training_padding_gap
max_training_token=args.max_training_token
logps_chunk_size=max(1, args.logps_chunk_size)
verification_capacity=args.verification_capacity
max_draft_token_length=args.max_draft_token_length
max_draft_k=args.max_draft_k
max_verification_num=args.max_verification_num
min_draft_token_length=args.min_draft_token_length
draft_token_length_c=args.draft_token_length_c
statistical_time=args.statistical_time
num_workers=args.num_workers
persistent_workers=_as_bool(args.persistent_workers)
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
draft_accumulation_steps = args.draft_accumulation_steps
target_lr = args.target_lr
draft_lr = args.draft_lr
is_train_draft = args.is_train_draft
model_type = args.model_type
model_dir = args.model_dir
adapter_path = args.adapter_path
temperature = args.temperature
top_p = args.top_p
version_name = args.version_name
log_file = args.log_file
summary_file = args.summary_file or os.path.join(os.path.dirname(log_file), "summary.json")
timing_file = args.timing_file or os.path.join(os.path.dirname(log_file), "timing.csv")
if not is_main_process:
    log_file = os.devnull
    summary_file = os.devnull
    timing_file = os.devnull
saved_model_dir = args.saved_model_dir
saved_draft_model_dir = args.saved_draft_model_dir
saved_statistics_dir = args.saved_statistics_dir
checkpoint_dir = args.checkpoint_dir or os.path.join(os.path.dirname(saved_model_dir), "checkpoints")
save_checkpoint_steps = int(args.save_checkpoint_steps or 0)
keep_last_checkpoints = int(args.keep_last_checkpoints or 0)
resume_checkpoint = args.resume_checkpoint
append_log = bool(resume_checkpoint) if str(args.append_log) == '' else _as_bool(args.append_log)
trace_seed = int(args.seed)
log_interval = max(1, int(args.log_interval))
opd_kwargs={"method":method,"opd_rank":args.opd_rank,"opd_topk":args.opd_topk,
    "opd_fast_lr":args.opd_fast_lr,"opd_visited_weight":args.opd_visited_weight,
    "opd_frontier_weight":args.opd_frontier_weight,"opd_update_stream":_as_bool(args.opd_update_stream),
    "opd_profile":_as_bool(args.opd_profile),"opd_diagnostics":_as_bool(args.opd_diagnostics),
    "opd_backend":args.opd_backend,"kv_gather_strategy":args.kv_gather_strategy,
    "opd_train_projector":_as_bool(args.opd_train_projector) and is_train_draft}
effective_opd_backend = 'off'
opd_eval_kwargs=dict(opd_kwargs,opd_train_projector=False)
reset_rng_on_resume = _as_bool(args.reset_rng_on_resume)
_seed_everything(trace_seed)
max_grpo_steps = max(0, int(args.max_grpo_steps))
drift_topk = max(1, int(args.drift_topk))
drift_temperature = float(args.drift_temperature)
drift_row_chunk_size = max(1, int(args.drift_row_chunk_size))
draft_lr_multiplier = float(args.draft_lr_multiplier)
policy_lag_output_dir = args.policy_lag_output_dir
analysis_boundaries = {
    int(item) for item in args.analysis_boundaries.split(',') if item.strip()
}
analysis_interval = max(0, int(args.analysis_interval))
analysis_seeds = [int(item) for item in args.analysis_seeds.split(',') if item.strip()]
analysis_enabled = bool(policy_lag_output_dir) and bool(analysis_boundaries or analysis_interval)
analysis_completed = set()
analysis_per_response = []
analysis_summaries = []
if policy_lag_output_dir and _as_bool(args.analysis_resume):
    for completed_path in Path(policy_lag_output_dir).glob('boundaries/step_*/complete.json'):
        try:
            analysis_completed.add(int(json.loads(completed_path.read_text())['policy_step']))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    response_file = Path(policy_lag_output_dir) / 'per_response.jsonl'
    summary_jsonl_file = Path(policy_lag_output_dir) / 'summary.jsonl'
    if response_file.is_file():
        analysis_per_response = [json.loads(line) for line in response_file.read_text().splitlines() if line.strip()]
    if summary_jsonl_file.is_file():
        analysis_summaries = [
            BranchSummary(**json.loads(line))
            for line in summary_jsonl_file.read_text().splitlines()
            if line.strip()
        ]
if analysis_enabled and args.draft_backend != 'eagle3':
    raise ValueError('policy-lag analysis requires --draft_backend=eagle3')
if analysis_enabled and world_size > 1:
    raise ValueError('policy-lag side-branch analysis must be run with NPROC_PER_NODE=1')
if analysis_enabled and draft_accumulation_steps != 1:
    raise ValueError('paired policy-lag branches require --draft_accumulation_steps=1')
if analysis_enabled and grpo_iteration_num != 1:
    raise ValueError('policy-lag boundary pairing currently requires --grpo_iteration_num=1')
if int(args.analysis_draft_update_steps) <= 0:
    raise ValueError('--analysis_draft_update_steps must be positive')
if drift_temperature <= 0.0:
    raise ValueError('--drift_temperature must be positive')
if draft_lr_multiplier <= 0.0:
    raise ValueError('--draft_lr_multiplier must be positive')
if (args.draft_train_max_batch_size < 1 or args.draft_train_max_tokens < 1 or
        args.draft_train_max_padding_ratio < 1.0):
    raise ValueError('draft microbatch size/tokens must be positive and padding ratio >= 1')
if args.draft_backend!='eagle3':raise ValueError('fair OPD/FastGRPO pipeline requires eagle3')
if not 1<=args.opd_rank<=64 or args.opd_topk<max_draft_k or args.opd_fast_lr<0 or min(args.opd_visited_weight,args.opd_frontier_weight)<0:
    raise ValueError('invalid OPD rank/topk/lr/weights')
fastgrpo_ablation = not np.isclose(draft_lr_multiplier, 1.0)
model_torch_dtype = _dtype_from_name(args.dtype)
attn_impl = _resolve_attn_implementation(args.attn_implementation)

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(saved_draft_model_dir):
    os.makedirs(saved_draft_model_dir)
if not os.path.exists(saved_statistics_dir):
    os.makedirs(saved_statistics_dir)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)
if os.path.dirname(summary_file):
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)


if is_main_process:
    print(datetime.now())
    print(model_type,os.getenv('CUDA_VISIBLE_DEVICES'))
    print("=" * 60)
    print("Training & Generation Configuration")
    print("=" * 60)
print(f"Model: {model_type} | Version: {version_name}")
print(f"Path: model={model_dir}, adapter={adapter_path}")
print(f"Train: epochs={num_epochs}, batch={batch_size}, "
      f"acc_steps={accumulation_steps}, draft_acc_steps={draft_accumulation_steps}")
print(f"LR: target={target_lr}, draft={draft_lr} | "
      f"Seq: max_len={max_length}, max_prompt={max_prompt_length}, "
      f"max_tokens={max_training_token}, pad_gap={max_training_padding_gap}, "
      f"logps_chunk={logps_chunk_size}")
print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
print(f"B200/spec: dtype={args.dtype}, attn_impl={attn_impl or 'default'}, "
      f"verification_capacity={verification_capacity}, max_verification_num={max_verification_num}, "
      f"max_draft_len={max_draft_token_length}, max_draft_k={max_draft_k}, "
      f"statistical_time={statistical_time}")
print(f"Draft: train={is_train_draft}")
print('Persistent EAGLE3 objective: 2.0*SmoothL1 + 0.1*soft CE; generated positions only; shared by both methods')
print(f"Method: {method} | OPD: rank={args.opd_rank}, topk={args.opd_topk}, fast_lr={args.opd_fast_lr}, stream={args.opd_update_stream}")
print(f"Trace: max_new_grpo_steps={max_grpo_steps}, drift_topk={drift_topk}, "
      f"drift_temperature={drift_temperature}, drift_row_chunk={drift_row_chunk_size}")
print(f"FastGRPO ablation: enabled={fastgrpo_ablation}, draft_lr_multiplier={draft_lr_multiplier}")
print(f"Iteration: grpo_iter={grpo_iteration_num}, sample={sample_num}, "
      f"repeat_gen={repeated_generate_nums}")
print(f"Dataset subset: fraction={train_data_fraction}, max_samples={max_train_samples}, seed={train_subset_seed}")
print("=" * 60)


target_config=AutoConfig.from_pretrained(model_dir)
if model_torch_dtype != "auto":
    target_config.torch_dtype = model_torch_dtype
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=model_torch_dtype, config=target_config, attn_implementation=attn_impl).cuda()
target_model.eval()

config=AutoConfig.from_pretrained(model_dir)
if args.draft_backend == 'eagle3':
    configured_layers = None
    if args.eagle_feature_layers.strip():
        configured_layers = [int(item) for item in args.eagle_feature_layers.split(',')]
    model = Eagle3FastGRPOAdapter(
        target_model=target_model,
        draft_config=args.draft_config,
        draft_checkpoint=adapter_path,
        vocab_mapping=args.vocab_mapping,
        initialization_mode=args.draft_initialization_mode,
        feature_layers=configured_layers,
        ttt_length=args.eagle_ttt_length,
        lk_loss_type=args.eagle_lk_loss_type or None,
        kl_scale=args.eagle_kl_scale,
        kl_decay=args.eagle_kl_decay,
        opd_rank=args.opd_rank if method=='opd_reflex' else None,
    )
else:
    # The legacy implementation is not a dependency of the EAGLE-3 backend.
    from helper.modeling_draft import Model

    config.rope_scaling=None
    config.num_hidden_layers=1
    if model_torch_dtype != "auto":
        config.torch_dtype = model_torch_dtype
    model=Model(config,target_model=target_model)
    model.load_model(adapter_path)
print(adapter_path)
model=model.cuda()
model._training_method=method
tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side="left")

if args.draft_backend == 'eagle3':
    if len(tokenizer) > int(model.draft_model.vocab_size):
        raise ValueError(
            f'tokenizer size {len(tokenizer)} exceeds EAGLE-3 target vocabulary '
            f'{model.draft_model.vocab_size}'
        )
    if model.draft_model.embed_tokens.weight.shape[0] != model.target_model.get_input_embeddings().weight.shape[0]:
        raise ValueError('EAGLE-3 and target embedding vocabularies are not identical')


if config.model_type == 'llama':
    tokenizer.pad_token = "<|end_of_text|>" 
    tokenizer.pad_token_id = 128001
    

QAs = (
    get_QAs_from_path(args.dataset_path, args.train_split)
    if args.dataset_path else get_train_QAs(args.train_option)
)
full_train_samples = len(QAs)
QAs = select_train_subset(
    QAs,
    fraction=train_data_fraction,
    max_samples=max_train_samples,
    seed=train_subset_seed,
)
selected_train_samples = len(QAs)
print(
    f"Train dataset: option={args.train_option}, full={full_train_samples}, "
    f"selected={selected_train_samples}"
)
df = pd.DataFrame(QAs)

for param in model.draft_model.parameters():
    param.requires_grad=True

for param in model.target_model.parameters():
    param.requires_grad=False
for param in model.lm_head.parameters():
    param.requires_grad=False
for param in model.embed_tokens.parameters():
    param.requires_grad=False
if hasattr(model, 'opd_projector'):
    model.opd_projector.requires_grad_(
        method == 'opd_reflex' and _as_bool(args.opd_train_projector) and is_train_draft
    )
    

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          
    r=64,                           
    lora_alpha=32,                
    lora_dropout=0.0,              
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model.target_model = get_peft_model(model.target_model,lora_config)
if  args.load_lora_path != "":
    model.target_model.load_adapter(args.load_lora_path,adapter_name="default")
model.target_model.print_trainable_parameters()

def _get_base_causal_lm(causal_lm):
    """Return the underlying causal LM while preserving injected LoRA modules."""
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm


def _autocast_dtype(causal_lm):
    dtype = getattr(causal_lm, "dtype", None)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16


def _token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    """Compute selected-token log-probabilities without a full [B, T, vocab] tensor."""
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    logps_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected_logits = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_denominator = torch.logsumexp(logits, dim=-1)
        logps_chunks.append(selected_logits - log_denominator)
        del logits, selected_logits, log_denominator

    if logps_chunks:
        return torch.cat(logps_chunks, dim=1)
    return hidden_states.new_zeros((hidden_states.shape[0], 0))


def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    """Forward the backbone once, then apply the LM head in bounded chunks."""
    base_model = _get_base_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type

    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    return _token_logps_from_hidden(
        hidden_states, base_model.lm_head, input_ids, chunk_size
    )


def compute_target_loss_and_backward(
    model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    grpo_iteration,
    old_logps=None,
    ref_logps=None,
    chunk_size=256,
    loss_scale=1.0,
):
    """Compute the GRPO loss and backpropagate without full-vocabulary logits."""
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    if grpo_iteration == 0:
        model.target_model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps_gpu = compute_model_token_logps(
                model.target_model,
                input_ids,
                attention_mask,
                chunk_size,
            ).detach()
        model.target_model.enable_adapter_layers()
        ref_logps_for_loss = ref_logps_gpu
        old_logps_for_loss = None
    else:
        if old_logps is None or ref_logps is None:
            raise ValueError(
                "old_logps and ref_logps are required when grpo_iteration > 0"
            )
        old_logps_for_loss = old_logps.to(device, non_blocking=True)
        ref_logps_for_loss = ref_logps.to(device, non_blocking=True)

    model.target_model.enable_adapter_layers()
    base_model = _get_base_causal_lm(model.target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    old_chunks_to_store = []
    loss_value = 0.0
    abs_loss1_value = 0.0
    loss2_value = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        cur_mask = token_mask[:, start:end]

        if grpo_iteration == 0:
            cur_old_logps = logps.detach()
            old_chunks_to_store.append(cur_old_logps.cpu())
        else:
            cur_old_logps = old_logps_for_loss[:, start:end]

        cur_ref_logps = ref_logps_for_loss[:, start:end]
        coef1 = torch.exp(logps - cur_old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)

        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        scaled_loss = chunk_loss * loss_scale
        scaled_loss.backward(retain_graph=(end < seq_len))

        with torch.no_grad():
            loss_value += float(chunk_loss.detach().cpu())
            abs_loss1_value += float(
                torch.abs((loss1 * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )
            loss2_value += float(
                ((kl * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )

        del (
            logits,
            logps,
            coef1,
            coef2,
            loss1,
            coef3,
            kl,
            token_loss,
            chunk_loss,
            scaled_loss,
        )

    if grpo_iteration == 0:
        if old_chunks_to_store:
            old_logps_out = torch.cat(old_chunks_to_store, dim=1)
        else:
            old_logps_out = torch.empty((input_ids.shape[0], 0))
        ref_logps_out = ref_logps_for_loss.detach().cpu()
    else:
        old_logps_out = old_logps
        ref_logps_out = ref_logps

    del hidden_states, policy_hidden
    if grpo_iteration == 0:
        del ref_logps_gpu, ref_logps_for_loss
    else:
        del old_logps_for_loss, ref_logps_for_loss

    return (
        loss_value,
        abs_loss1_value,
        loss2_value,
        old_logps_out,
        ref_logps_out,
    )


def training_draft_model(model,outputs,prompt_mask,token_budget=None):
    if getattr(model, 'is_eagle3_specforge', False):
        return training_eagle3_specforge(
            model, outputs, prompt_mask, token_budget=token_budget,
            mode=args.draft_train_mode,
            max_batch_size=args.draft_train_max_batch_size,
            max_tokens=args.draft_train_max_tokens,
            max_padding_ratio=args.draft_train_max_padding_ratio,
            profile=_as_bool(args.draft_train_profile),
            accumulation_steps=draft_accumulation_steps,
        )
    

    all_draft_input_states = outputs['all_draft_input_states']
    all_draft_input_ids = outputs['all_draft_input_ids']
    all_prompt_length = [prompt_mask[idx // repeated_generate_nums].sum().item() for idx in range(len(all_draft_input_states))]
    
    prompt_mask=prompt_mask.cpu()
    device=model.target_model.device
    
    sorted_pairs = sorted(
        zip(all_draft_input_ids, all_draft_input_states, all_prompt_length),
        key=lambda x: len(x[0]),
        reverse=False  
    )

    all_draft_input_ids_sorted, all_draft_input_states_sorted, all_prompt_length_sorted = zip(*sorted_pairs)

    all_draft_input_ids = list(all_draft_input_ids_sorted)
    all_draft_input_states = list(all_draft_input_states_sorted)
    all_prompt_length = list(all_prompt_length_sorted)
    
    l1_loss=torch.nn.SmoothL1Loss(reduction='none')
    total_loss1,total_loss2=0,0
    drift_tv_sum = 0.0
    drift_kl_sum = 0.0
    drift_count = 0
    
    draft_input_states_list=[]
    draft_input_ids_list=[]
    prompt_length_list=[]
    
    cur_max_length=0
    hidden_size=all_draft_input_states[0].shape[-1]
    
    for idx , (draft_input_states,draft_input_ids,prompt_length) in enumerate(zip(all_draft_input_states,all_draft_input_ids,all_prompt_length)):
        
        if ((draft_input_ids.shape[-1]*(len(draft_input_states_list)+1)<=max_training_token*2 and
            (draft_input_ids.shape[-1]-cur_max_length)*len(draft_input_states_list)<=max_training_padding_gap) or
            len(draft_input_states_list)==0):
            
                draft_input_states_list.append(draft_input_states)
                draft_input_ids_list.append(draft_input_ids)
                prompt_length_list.append(prompt_length)
                
                cur_max_length=max(cur_max_length, draft_input_ids.shape[-1])
            
        else:
            
            cur_batch=len(draft_input_states_list)

            loss_mask=[[] for _ in range(cur_batch)]
            attention_mask=[[] for _ in range(cur_batch)]
            
            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
                attention_mask[idx_seq]=[1]*cur_len

            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    draft_input_states_list[idx_seq]=torch.concat(
                        [draft_input_states_list[idx_seq],
                        torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                        dim=-2)
                    
                    draft_input_ids_list[idx_seq]=torch.concat(
                        [draft_input_ids_list[idx_seq],
                        torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                        dim=-1)
                    
                    loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
                    attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
            
            draft_input_states=torch.stack(draft_input_states_list,dim=0)
            draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
            loss_mask=torch.tensor(loss_mask,device=device)
            attention_mask=torch.tensor(attention_mask,device=device)

            with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
                draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                                attention_mask=attention_mask,use_cache=False)
                
            next_feature_states=draft_outputs['next_feature_states']
            draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
            draft_logits=model.lm_head(draft_hidden_states)
            
            with torch.no_grad():
                target_hidden_states=draft_input_states
                target_logits_raw=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))[:,1:,:]

            draft_logits_raw=draft_logits[:,:-1,:]
            drift_tv, drift_kl, cur_drift_count = sparse_union_metrics(
                draft_logits_raw.detach(),
                target_logits_raw.detach(),
                loss_mask[...,:-1].bool(),
                topk=drift_topk,
                temperature=drift_temperature,
                row_chunk_size=drift_row_chunk_size,
            )
            drift_tv_sum += drift_tv * cur_drift_count
            drift_kl_sum += drift_kl * cur_drift_count
            drift_count += cur_drift_count
            target_logits=target_logits_raw.float().softmax(dim=-1).detach()
                
            loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

            loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss1=loss1.sum(-1)
            loss1=loss1*2.0
            
            draft_logits=draft_logits_raw.float().softmax(dim=-1)

            plogp=target_logits*torch.log(draft_logits)
            loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
            loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss2= - loss2.sum(-1)

            loss2=loss2*0.1
            
            loss=loss1+loss2
                
            total_loss1+=loss1.item()
            total_loss2+=loss2.item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                
                loss = loss.detach()
                del loss
                torch.cuda.empty_cache()
            else:

                loss=loss/len(all_draft_input_states)
                loss=loss/draft_accumulation_steps
                loss.backward()
                
            draft_input_states_list=[all_draft_input_states[idx]]
            draft_input_ids_list=[all_draft_input_ids[idx]]
            prompt_length_list=[all_prompt_length[idx]]
            cur_max_length=all_draft_input_ids[idx].shape[-1]
            
    cur_batch=len(draft_input_states_list)

    loss_mask=[[] for _ in range(cur_batch)]
    attention_mask=[[] for _ in range(cur_batch)]
    
    cur_max_length=0
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
        attention_mask[idx_seq]=[1]*cur_len
        
        cur_max_length=max(cur_max_length, cur_len)
        
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        padding_len=cur_max_length-cur_len
        
        if padding_len>0:
            draft_input_states_list[idx_seq]=torch.concat(
                [draft_input_states_list[idx_seq],
                torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                dim=-2)
            
            draft_input_ids_list[idx_seq]=torch.concat(
                [draft_input_ids_list[idx_seq],
                torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                dim=-1)
            
            loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
            attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
    
    draft_input_states=torch.stack(draft_input_states_list,dim=0)
    draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
    loss_mask=torch.tensor(loss_mask,device=device)
    attention_mask=torch.tensor(attention_mask,device=device)
    
    with torch.amp.autocast(str(model.target_model.device),
                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                        attention_mask=attention_mask,use_cache=False)
        
    next_feature_states=draft_outputs['next_feature_states']
    draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
    draft_logits=model.lm_head(draft_hidden_states)
    
    with torch.no_grad():
        target_hidden_states=draft_input_states
        target_logits_raw=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))[:,1:,:]

    draft_logits_raw=draft_logits[:,:-1,:]
    drift_tv, drift_kl, cur_drift_count = sparse_union_metrics(
        draft_logits_raw.detach(),
        target_logits_raw.detach(),
        loss_mask[...,:-1].bool(),
        topk=drift_topk,
        temperature=drift_temperature,
        row_chunk_size=drift_row_chunk_size,
    )
    drift_tv_sum += drift_tv * cur_drift_count
    drift_kl_sum += drift_kl * cur_drift_count
    drift_count += cur_drift_count
    target_logits=target_logits_raw.float().softmax(dim=-1).detach()
        
    loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

    loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
    loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss1=loss1.sum(-1)
    loss1=loss1*2.0
    
    draft_logits=draft_logits_raw.float().softmax(dim=-1)

    plogp=target_logits*torch.log(draft_logits)
    loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
    loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss2= - loss2.sum(-1)

    loss2=loss2*0.1
    
    loss=loss1+loss2
        
    total_loss1+=loss1.item()
    total_loss2+=loss2.item()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        
        loss = loss.detach()
        del loss
        torch.cuda.empty_cache()
    else:

        loss=loss/len(all_draft_input_states)
        loss.backward()
            
        
    total_loss1/=len(all_draft_input_states)
    total_loss2/=len(all_draft_input_states)
    
    mean_drift_tv = drift_tv_sum / max(drift_count, 1)
    mean_drift_kl = drift_kl_sum / max(drift_count, 1)
    return total_loss1,total_loss2,mean_drift_tv,mean_drift_kl,drift_count


def training_eagle3_specforge(model, outputs, prompt_mask, token_budget=None,
                              mode='batched', max_batch_size=8, max_tokens=2048,
                              max_padding_ratio=1.25, profile=False, accumulation_steps=1):
    """Persistent upstream-equivalent FastGRPO objective on the real EAGLE3 model.

    Inputs are SpecForge 3H features; feature supervision is final target H.
    Shared by both methods; no OPD A/B objective or SpecForge TTT/LK loss here.
    """
    from helper.eagle3_online_objective import persistent_loss
    feature_rows = outputs['all_draft_input_states']
    target_rows = outputs.get('all_target_hidden_states')
    token_rows = outputs['all_draft_input_ids']
    if target_rows is None:
        raise RuntimeError('EAGLE-3 rollout is missing final target hidden states')
    if not (len(feature_rows) == len(target_rows) == len(token_rows)):
        raise RuntimeError('misaligned EAGLE-3 rollout feature tensors')
    if mode not in ('batched', 'per_response'):
        raise ValueError('draft train mode must be batched or per_response')
    if max_batch_size < 1 or max_tokens < 1 or max_padding_ratio < 1:
        raise ValueError('draft microbatch limits must be positive and padding ratio >= 1')
    if accumulation_steps<1:raise ValueError('draft accumulation steps must be positive')
    nrows = len(feature_rows)
    if nrows == 0:
        return 0, 0, 0, 0, 0
    profile_start = time.perf_counter() if profile else None
    if prompt_mask.shape[0] * repeated_generate_nums < nrows:
        raise ValueError('prompt mask has fewer rows than the rollout')
    # A single device-to-host transfer before packing; no per-response sync.
    prompt_lengths = prompt_mask.sum(dim=1).repeat_interleave(repeated_generate_nums)[:nrows].tolist()
    lengths = [int(row.shape[-1]) for row in token_rows]
    allowed = []
    remaining = None if token_budget is None else max(0, int(token_budget))
    for seq_len, prompt_len in zip(lengths, prompt_lengths):
        count = max(0, seq_len - min(int(prompt_len), seq_len) - 1)
        if remaining is not None:
            count = min(count, remaining)
            remaining -= count
        allowed.append(count)
    valid_tokens = sum(allowed)
    if not valid_tokens:
        return 0, 0, 0, 0, 0
    # Shortest-first bucketing is deterministic and reduces both padding and
    # compact-teacher temporary memory. A singleton is always legal.
    ordered = [i for i in range(nrows) if allowed[i] > 0]
    if mode == 'batched':
        ordered.sort(key=lambda i: (lengths[i], i))
    packs = []
    pack = []
    pack_sum = 0
    for i in ordered:
        prospective_max = lengths[i]
        prospective_count = len(pack) + 1
        prospective_sum = pack_sum + lengths[i]
        if pack and (prospective_count > (1 if mode == 'per_response' else max_batch_size)
                     or prospective_max * prospective_count > max_tokens
                     or prospective_max * prospective_count > max_padding_ratio * prospective_sum):
            packs.append(pack)
            pack, pack_sum = [], 0
        pack.append(i)
        pack_sum += lengths[i]
    if pack:
        packs.append(pack)
    target_head = _get_base_causal_lm(model.target_model).lm_head
    loss_total = torch.zeros((), device=model.device, dtype=torch.float32)
    distribution_total = torch.zeros_like(loss_total)
    timing = ({'packs': len(packs), 'responses': len(ordered),
               'bucketing_cpu_ms': (time.perf_counter() - profile_start) * 1000,
               'packing_cpu_ms': 0.0} if profile else None)
    cuda_events = []
    for pack in packs:
        packing_start = time.perf_counter() if profile else None
        seq_len = max(lengths[i] for i in pack)
        row_lengths = torch.tensor([lengths[i] for i in pack], device=model.device)
        row_prompts = torch.tensor([min(int(prompt_lengths[i]), lengths[i]) for i in pack], device=model.device)
        row_allowed = torch.tensor([allowed[i] for i in pack], device=model.device)
        positions = torch.arange(seq_len, device=model.device).unsqueeze(0)
        attention_mask = (positions < row_lengths[:, None]).long()
        loss_mask = ((positions >= row_prompts[:, None]) &
                     (positions < row_prompts[:, None] + row_allowed[:, None]) &
                     (positions < row_lengths[:, None] - 1)).unsqueeze(-1).float()
        # inference_mode tensors cannot be saved by autograd. Convert each row
        # before padding; this changes neither values nor teacher supervision.
        ids = torch.nn.utils.rnn.pad_sequence(
            [rollout_tensor_for_training(token_rows[i]) for i in pack],
            batch_first=True, padding_value=0)
        features = torch.nn.utils.rnn.pad_sequence(
            [rollout_tensor_for_training(feature_rows[i].to(model.dtype)) for i in pack],
            batch_first=True, padding_value=0)
        target_hidden = torch.nn.utils.rnn.pad_sequence(
            [rollout_tensor_for_training(target_rows[i].to(model.dtype)) for i in pack],
            batch_first=True, padding_value=0)
        if profile:
            timing['packing_cpu_ms'] += (time.perf_counter() - packing_start) * 1000
        if profile and torch.cuda.is_available() and model.device.type == 'cuda':
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_start.record()
        feature_losses, distribution_losses = persistent_loss(
            model,ids,features,target_hidden,attention_mask,loss_mask.squeeze(-1),target_head)
        feature_loss=feature_losses.sum()
        distribution_loss=distribution_losses.sum()
        loss=feature_loss+distribution_loss
        if profile and torch.cuda.is_available() and model.device.type == 'cuda':
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_end.record()
            cuda_events.append(('total_forward_ms', forward_start, forward_end))
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_start.record()
        (loss / nrows / accumulation_steps).backward()
        if profile and torch.cuda.is_available() and model.device.type == 'cuda':
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_end.record()
            cuda_events.append(('backward_ms', backward_start, backward_end))
        loss_total += feature_loss.detach()
        distribution_total += distribution_loss.detach()
    total_feature, total_distribution = torch.stack((loss_total, distribution_total)).cpu().tolist()
    if profile:
        for label, start, end in cuda_events:
            timing[label] = timing.get(label, 0.0) + start.elapsed_time(end)
        model.last_draft_train_profile = timing
    denom = nrows
    # Legacy arity; now the two upstream weighted draft loss components.
    return total_feature / denom, total_distribution / denom, 0.0, 0.0, valid_tokens

        
optimizer_target = torch.optim.AdamW(model.target_model.parameters(), lr=target_lr)
optimizer_draft = draft_optimizer(model.draft_model,draft_lr,args.opd_projector_lr if method=='opd_reflex' else None)

log_mode = "a" if append_log else "w"
with open(log_file, log_mode, encoding='utf-8') as f:
    pass

step=0
used_items=0
draft_step=0
draft_accumulated_step=0 
batch_logs=[]
batch_data={
    'messages':[],
    'rewards':[],
    'std_rewards':[],
    'generate_time_cost':0,
    'last_generate_time_cost':[],
    'train_time_cost':0,
    'last_train_time_cost':[],
    'generate_length':0,
    'last_generate_length':[],
    'total_rollout_tokens':0,
    'total_acc_length':0,
    'last_acc_length':[],
    'total_decoded_token_num':0,
    'last_decoded_token_num':[],
    'total_accepted_draft_tokens':0,
    'total_proposed_draft_tokens':0,
    'last_accepted_draft_tokens':[],
    'last_proposed_draft_tokens':[],
    'prefill_time_cost':0,
    'target_time_cost':0,
    'draft_time_cost':0,
    'check_time_cost':0,
    'ignore_due_correct':0,
    'ignore_due_incorrect':0,
    'mean_rewards':0,
    'last_mean_rewards':[],
    'draft_train_time_cost':0,
    'last_draft_loss1':[],
    'last_draft_loss2':[] ,
    'generate_length_list':[],
    'draft_sparse_tv_sum':0.0,
    'draft_sparse_kl_sum':0.0,
    'draft_sparse_count':0,
    'trace_rollout_count':0,
    'opd_updates':0,
    'opd_profile_time_ms':0.0,
    **{name:0. for name in OPD_COUNTER_NAMES+GENERATION_COUNTER_NAMES},
    'reward_sum':0.0,
    'reward_count':0,
    'target_loss_sum':0.0,
    'target_loss_count':0,
    'draft_loss1_sum':0.0,
    'draft_loss2_sum':0.0,
    'draft_loss_count':0,
}

optimizer_target.zero_grad(set_to_none=True)
optimizer_draft.zero_grad(set_to_none=True)
session_start_time=process_started_at
cumulative_elapsed_before_resume=0.0


def _cumulative_wall_time():
    return cumulative_elapsed_before_resume + time.perf_counter() - session_start_time


batch=[]
start_epoch = 0
start_batch = 0
last_checkpoint_step = -1

if resume_checkpoint:
    checkpoint = load_training_checkpoint(
        resume_checkpoint,
        model=model,
        optimizer_target=optimizer_target,
        optimizer_draft=optimizer_draft,
    )
    step = int(checkpoint.get("step", 0))
    used_items = int(checkpoint.get("used_items", 0))
    draft_step = int(checkpoint.get("draft_step", 0))
    draft_accumulated_step = int(checkpoint.get("draft_accumulated_step", 0))
    saved_batch_data = checkpoint.get("batch_data", {})
    if isinstance(saved_batch_data, dict):
        batch_data.update(saved_batch_data)
    start_epoch = int(checkpoint.get("epoch", 0))
    start_batch = int(checkpoint.get("next_batch", 0))
    last_checkpoint_step = step
    cumulative_elapsed_before_resume = float(
        checkpoint.get("cumulative_elapsed_time_s", 0.0)
    )
    print(
        f"Resumed FastGRPO checkpoint {resume_checkpoint}: "
        f"epoch={start_epoch + 1}, next_batch={start_batch}, "
        f"step={step}, used_items={used_items}, draft_step={draft_step}"
    )
    if reset_rng_on_resume:
        _seed_everything(trace_seed)
        print(f"Reset continuation RNG state to paired trace seed={trace_seed}")

# A continuation trace uses local counters so --max_grpo_steps=100 means 100
# newly completed labels even when the restored checkpoint is already at 310.
trace_start_step = int(step)
trace_start_draft_step = int(draft_step)
trace_rollout_count = 0
stop_requested = False
batch_data['draft_sparse_tv_sum'] = 0.0
batch_data['draft_sparse_kl_sum'] = 0.0
batch_data['draft_sparse_count'] = 0
batch_data['trace_rollout_count'] = 0
for param_group in optimizer_draft.param_groups:
    param_group['lr'] = float(param_group['lr']) * draft_lr_multiplier
    if param_group.get('name')=='opd_projector' and args.opd_projector_lr is not None:
        param_group['lr']=args.opd_projector_lr
effective_draft_lrs = [float(group['lr']) for group in optimizer_draft.param_groups]

run_config_log = {
    **opd_kwargs,
    "phase": "run_config",
    "run_name": version_name,
    "method": method,
    "opd_profile": _as_bool(args.opd_profile),
    "opd_diagnostics": _as_bool(args.opd_diagnostics),
    "opd_backend_requested": args.opd_backend,
    "resume_checkpoint": str(resume_checkpoint),
    "append_log": bool(append_log),
    "source_grpo_step": int(trace_start_step),
    "source_draft_step": int(trace_start_draft_step),
    "max_grpo_steps": int(max_grpo_steps),
    "train_option": str(args.train_option),
    "dataset_path": str(args.dataset_path),
    "eval_dataset_path": str(args.eval_dataset_path),
    "train_data_fraction": float(train_data_fraction),
    "train_subset_seed": int(train_subset_seed),
    "seed": int(trace_seed),
    "reset_rng_on_resume": bool(reset_rng_on_resume),
    "drift_metric": "bidirectional_topk_union_with_tail_sparse_tv",
    "drift_kl_direction": "forward_kl_target_to_draft",
    "drift_topk": int(drift_topk),
    "drift_temperature": float(drift_temperature),
    "drift_normalization": "mean_over_valid_response_token_rows",
    "draft_lr_multiplier": float(draft_lr_multiplier),
    "draft_train_mode": args.draft_train_mode,
    "draft_train_max_batch_size": int(args.draft_train_max_batch_size),
    "draft_train_max_tokens": int(args.draft_train_max_tokens),
    "draft_train_max_padding_ratio": float(args.draft_train_max_padding_ratio),
    "effective_draft_lrs": effective_draft_lrs,
    "fastgrpo_ablation": bool(fastgrpo_ablation),
    "step_metric_definition": "distinct inherited GRPO labels; all updates with the same label are accumulated",
    "log_interval": log_interval,
    "log_interval_applies_to": "progress display; authoritative step telemetry is always recorded",
}
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(run_config_log) + '\n')
phase_timings = PhaseTimings(
    model.target_model.device,
    target_s=batch_data.get('_phase_target_time_s', batch_data['train_time_cost']),
    draft_s=batch_data.get('_phase_draft_time_s', batch_data['draft_train_time_cost']),
)
step_metrics_state = batch_data.get('_step_metrics_state') if append_log else None
step_metrics_baseline = {}
if resume_checkpoint and step_metrics_state is None:
    baseline_metrics = _aggregate_job_metrics(
        batch_data, model.target_model.device, cumulative_elapsed_before_resume)
    step_metrics_baseline = completed_step_snapshot(
        baseline_metrics, batch_data, phase_timings, model.target_model.device,
        cumulative_elapsed_before_resume)
step_metrics = StepMetricsWriter(
    log_file, timing_file, enabled=is_main_process, append=append_log,
    baseline=step_metrics_baseline, state=step_metrics_state)

class TrainDataCollator:
    def __init__(self, tokenizer, max_prompt_length):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
    
    def __call__(self, batch):
        system_prompt = "You are a math problem assistant." 
        user_prompt =  '''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
        messages = []
        answers = []

        for example in batch:
            messages.append([
                {"role" : "system" , "content": system_prompt} , 
                {"role" : "user" , "content": user_prompt.format_map({"instruction" : example['question']}) }
            ])
            answers.append(example['answer'])
        tokenized_inputs = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            return_tensors='pt',padding='longest',truncation=True,max_length=self.max_prompt_length,padding_side='left'
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'messages': messages,        
            'answers': answers,           
        }


def _analysis_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_analysis_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _analysis_boundary(step_value):
    return step_value in analysis_boundaries or (
        analysis_interval > 0 and step_value > 0 and step_value % analysis_interval == 0
    )


def _teacher_prefix_hidden(eval_batch):
    base = _get_base_causal_lm(model.target_model)
    with torch.inference_mode():
        result = base.model(
            input_ids=eval_batch['input_ids'].to(model.target_model.device),
            attention_mask=eval_batch['attention_mask'].to(model.target_model.device),
            use_cache=False,
            return_dict=True,
        )
        hidden = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result[0]
        return hidden.float().cpu()


def _teacher_tv_from_hidden(old_hidden, new_hidden, valid_mask):
    base = _get_base_causal_lm(model.target_model)
    total = 0.0
    count = 0
    for start in range(0, old_hidden.shape[1], drift_row_chunk_size):
        mask = valid_mask[:, start:start + drift_row_chunk_size].bool()
        if not mask.any():
            continue
        with torch.inference_mode():
            old_logits = base.lm_head(
                old_hidden[:, start:start + drift_row_chunk_size].to(
                    device=model.target_model.device, dtype=base.lm_head.weight.dtype
                )
            )
            new_logits = base.lm_head(
                new_hidden[:, start:start + drift_row_chunk_size].to(
                    device=model.target_model.device, dtype=base.lm_head.weight.dtype
                )
            )
            value = teacher_shift_tv(old_logits, new_logits, mask, row_chunk_size=drift_row_chunk_size)
        cur_count = int(mask.sum().item())
        total += value * cur_count
        count += cur_count
        del old_logits, new_logits
    return total / max(count, 1)


def _evaluate_analysis_branch(branch_name, draft_state, eval_batch, policy_step):
    model.draft_model.load_state_dict(draft_state, strict=True)
    model.eval()
    rows = []
    prompt_count = eval_batch['input_ids'].shape[0]
    for sampling_seed in analysis_seeds:
        _seed_everything(sampling_seed)
        with torch.inference_mode():
            result = speculative_generate(
                model=model,
                input_ids=eval_batch['input_ids'].to('cuda'),
                attention_mask=eval_batch['attention_mask'].to('cuda'),
                tokenizer=tokenizer,
                do_sample=True,
                max_length=max_length,
                repeated_generate_nums=repeated_generate_nums,
                temperature=temperature,
                top_p=top_p,
                verification_capacity=verification_capacity,
                max_draft_token_length=max_draft_token_length,
                max_draft_k=max_draft_k,
                max_verification_num=max_verification_num,
                min_draft_token_length=min_draft_token_length,
                draft_token_length_c=draft_token_length_c,
                return_all_draft_input=False,
                statistical_time=False,
                **opd_eval_kwargs,
            )
        for response_index, (accepted, rounds, generated) in enumerate(zip(
            result['response_accepted_length_sum'],
            result['response_verification_rounds'],
            result['response_generated_tokens'],
        )):
            rows.append({
                'policy_step': int(policy_step),
                'seed': int(sampling_seed),
                'branch': branch_name,
                'prompt_id': int(response_index // repeated_generate_nums),
                'response_index': int(response_index % repeated_generate_nums),
                'accepted_length_sum': int(accepted),
                'verification_rounds': int(rounds),
                'generated_tokens': int(generated),
            })
    model.train()
    model.target_model.eval()
    return rows


eval_batch_for_analysis = None
if analysis_enabled:
    analysis_eval_path = args.eval_dataset_path or args.dataset_path
    eval_qas = (
        get_QAs_from_path(analysis_eval_path, args.eval_split)
        if analysis_eval_path else get_test_QAs(args.train_option)
    )
    eval_qas = eval_qas[:max(1, int(args.analysis_eval_prompts))]
    train_prompt_ids = {str(row['question']).strip() for row in QAs}
    eval_prompt_ids = {str(row['question']).strip() for row in eval_qas}
    overlap = train_prompt_ids & eval_prompt_ids
    if overlap:
        raise RuntimeError(
            'policy-lag evaluation prompts overlap the GRPO training pool; '
            f'found {len(overlap)} duplicate prompt(s). Configure a disjoint '
            '--eval_dataset_path.'
        )
    # This held-out split is never inserted into the GRPO buffer.
    eval_batch_for_analysis = TrainDataCollator(tokenizer, max_prompt_length)(eval_qas)
    Path(policy_lag_output_dir).mkdir(parents=True, exist_ok=True)

train_sampler = DistributedSampler(
    QAs, num_replicas=world_size, rank=rank, shuffle=True, seed=trace_seed,
)
dataloader=DataLoader(
    QAs,
    collate_fn=TrainDataCollator(tokenizer=tokenizer, max_prompt_length=max_prompt_length),
    num_workers=num_workers,
    persistent_workers=persistent_workers and num_workers > 0,
    batch_size=batch_size,
    shuffle=False,
    sampler=train_sampler,
    drop_last=False,
)

progress_disabled = (
    os.environ.get("TQDM_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}
    or not is_main_process
)
epoch_bar = tqdm(
    range(start_epoch, num_epochs), total=num_epochs, initial=start_epoch,
    desc="GRPO epoch", unit="epoch", dynamic_ncols=True,
    mininterval=1.0, disable=progress_disabled, position=0,
)
rollout_timing_path=Path(args.log_file).parent / ('rollout_timing.csv' if rank==0 else f'rollout_timing.rank{rank}.csv')
rollout_resume_state=batch_data.get('_rollout_metrics_state')
if rollout_resume_state is None and resume_checkpoint:
    rollout_resume_state=dict(global_iter=start_epoch*len(dataloader)+start_batch,
        accepted=batch_data['total_acc_length'],rounds=batch_data['total_decoded_token_num'],
        tokens=batch_data['total_rollout_tokens'],generation=batch_data['generate_time_cost'],
        accepted_draft=batch_data['total_accepted_draft_tokens'],proposed=batch_data['total_proposed_draft_tokens'])
rollout_metrics=RolloutMetricsWriter(rollout_timing_path,method,state=rollout_resume_state,
    flush_interval=args.rollout_log_flush_interval)

def finish_rollout_iteration():
    rollout_metrics.finish(iter_outputs,grpo_step=step,used_items=used_items,wall_time_s=_cumulative_wall_time())
    batch_data['_rollout_metrics_state']=rollout_metrics.state.copy()

# Flush host telemetry on torchrun termination too; never touch CUDA here.
import signal
_previous_sigterm_handler=signal.getsignal(signal.SIGTERM)
def _flush_rollout_sigterm(signum,frame):
    rollout_metrics.flush()
    if callable(_previous_sigterm_handler):
        _previous_sigterm_handler(signum,frame)
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,_flush_rollout_sigterm)

for epoch in epoch_bar:
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    batch_data['ignore_due_correct']=0
    batch_data['ignore_due_incorrect']=0
    batch_data['length_stdev'] = []
    batch_data['length_range'] = []
    batch_data['length_cv'] = []
    
    batch_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"GRPO epoch {epoch + 1}/{num_epochs}",
        dynamic_ncols=True,
        unit="batch", mininterval=1.0, position=1,
        leave=False,
        disable=progress_disabled,
    )
    for i,batch in enumerate(batch_bar):
        if epoch == start_epoch and i < start_batch:
            batch_bar.set_postfix(step=step, phase="resume_skip", refresh=False)
            continue

        iter_outputs=None
        iteration_checkpoint=False
        rollout_metrics.begin(epoch+1,i,len(batch['answers']),used_items)
        try:
            if batch['input_ids'].shape[-1]>=max_length:
                batch=[]
                batch_bar.set_postfix(phase="skip_prompt_len", step=step, refresh=False)
                continue

            if None in batch['answers']:
                batch=[]
                batch_bar.set_postfix(phase="skip_none_answer", step=step, refresh=False)
                continue

            input_ids=batch['input_ids'].to('cuda')
            attention_mask=batch['attention_mask'].to('cuda')
            analysis_train_input_ids = input_ids.detach().cpu() if analysis_enabled else None
            analysis_train_attention_mask = attention_mask.detach().cpu() if analysis_enabled else None
            analysis_base_draft = None
            analysis_base_optimizer = None
            analysis_old_teacher_logits = None
            analysis_old_policy_id = None
            analysis_old_policy_state = None
            if analysis_enabled:
                analysis_base_draft = {
                    key: value.detach().cpu().clone()
                    for key, value in model.draft_model.state_dict().items()
                }
                analysis_base_optimizer = deepcopy(optimizer_draft.state_dict())
                analysis_old_policy_state = {
                    key: value.detach().cpu().clone()
                    for key, value in _target_lora_state_dict(model.target_model).items()
                }
                analysis_old_policy_id = state_digest(analysis_old_policy_state)
                analysis_old_teacher_logits = _teacher_prefix_hidden(eval_batch_for_analysis)
            messages=batch['messages']
            answers=batch['answers']

            with torch.inference_mode():
                outputs=speculative_generate(model=model,input_ids=input_ids,attention_mask=attention_mask,tokenizer=tokenizer,
                do_sample=True,max_length=max_length,repeated_generate_nums=repeated_generate_nums,temperature=temperature,top_p=top_p,
                verification_capacity=verification_capacity,
                max_draft_token_length=max_draft_token_length,
                max_draft_k=max_draft_k,
                max_verification_num=max_verification_num,
                min_draft_token_length=min_draft_token_length,
                draft_token_length_c=draft_token_length_c,
                return_all_draft_input=True,statistical_time=statistical_time,
                **opd_kwargs)
            iter_outputs=rollout_metrics.capture(outputs)
            effective_opd_backend = outputs.get('opd_backend', 'off')
            if _as_bool(args.opd_diagnostics) and is_main_process:
                with open(log_file,'a',encoding='utf-8') as f:
                    f.write(json.dumps({'phase':'opd_diagnostics','step':int(step),
                        **{k:v for k,v in outputs.items() if k.startswith('opd_final_')}})+'\n')
            if _as_bool(args.opd_profile) and is_main_process:
                with open(log_file, 'a', encoding='utf-8') as profile_stream:
                    profile_stream.write(json.dumps({
                        'phase': 'opd_profile', 'step': int(step),
                        'sections_ms': outputs.get('opd_profile_sections_ms'),
                    }) + '\n')


            prompt_length=input_ids.shape[-1]
            outputs['prompt_length']=prompt_length

            outputs['decoded_sequences']=[tokenizer.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]
            token_ids_length = [len(item) for item in outputs['generated_token_ids'] ]
            total_rollout_tokens = int(sum(token_ids_length))
            length_stdev = stdev(token_ids_length)
            length_range = max(token_ids_length) - min(token_ids_length)
            length_cv = length_stdev / mean(token_ids_length)
            length_ave = mean(token_ids_length)
            batch_data['generate_length_list'].extend(token_ids_length)
            batch_data['total_rollout_tokens']+=total_rollout_tokens

            draft_sparse_tv = None
            draft_sparse_kl = None
            draft_sparse_count = 0
            draft_update_committed = False
            if is_train_draft:
                if statistical_time:
                    torch.cuda.synchronize()
                draft_train_time_start=time.time()
                draft_phase_ticket = phase_timings.begin('draft')
                draft_loss1,draft_loss2,draft_sparse_tv,draft_sparse_kl,draft_sparse_count=training_draft_model(model,outputs,attention_mask)
                iter_outputs.update(iter_draft_feature_loss=float(draft_loss1),
                    iter_draft_distribution_loss=float(draft_loss2),iter_draft_total_loss=float(draft_loss1+draft_loss2))
                if _as_bool(args.draft_train_profile) and is_main_process:
                    with open(log_file, 'a', encoding='utf-8') as profile_stream:
                        profile_stream.write(json.dumps({
                            'phase': 'draft_profile', 'step': int(step),
                            **getattr(model, 'last_draft_train_profile', {}),
                        }) + '\n')
                if statistical_time:
                    torch.cuda.synchronize()
                batch_data['draft_train_time_cost']+=time.time()-draft_train_time_start
                batch_data['last_draft_loss1'].append(draft_loss1)
                batch_data['last_draft_loss2'].append(draft_loss2)
                draft_loss_weight = max(int(draft_sparse_count), 1)
                batch_data['draft_loss1_sum'] += float(draft_loss1) * draft_loss_weight
                batch_data['draft_loss2_sum'] += float(draft_loss2) * draft_loss_weight
                batch_data['draft_loss_count'] += draft_loss_weight
                batch_data['draft_sparse_tv_sum'] += float(draft_sparse_tv) * int(draft_sparse_count)
                batch_data['draft_sparse_kl_sum'] += float(draft_sparse_kl) * int(draft_sparse_count)
                batch_data['draft_sparse_count'] += int(draft_sparse_count)
                draft_accumulated_step += 1
                if is_train_draft and draft_accumulated_step % draft_accumulation_steps == 0:
                    if method=='opd_reflex' and _as_bool(args.opd_train_projector):
                        model.apply_opd_projector_gradient()
                    _sync_gradients(model.draft_model)
                    if _as_bool(args.draft_train_profile):
                        optimizer_profile_start = torch.cuda.Event(enable_timing=True)
                        optimizer_profile_end = torch.cuda.Event(enable_timing=True)
                        optimizer_profile_start.record()
                    optimizer_draft.step()
                    if _as_bool(args.draft_train_profile):
                        optimizer_profile_end.record()
                        optimizer_profile_end.synchronize()
                        if is_main_process:
                            with open(log_file, 'a', encoding='utf-8') as profile_stream:
                                profile_stream.write(json.dumps({
                                    'phase': 'draft_optimizer_profile', 'step': int(step),
                                    'optimizer_ms': optimizer_profile_start.elapsed_time(optimizer_profile_end),
                                }) + '\n')
                    optimizer_draft.zero_grad(set_to_none=True)
                    draft_step += 1
                    draft_update_committed = True
                phase_timings.end(draft_phase_ticket)

            if draft_step % 1024 == 0 and step > 0 and is_train_draft:
                with open(f"{saved_statistics_dir}/{step}.pkl","wb") as f:
                    pickle.dump(batch_data['generate_length_list'],f)

            generate_length=0
            for idx_batch in range(len(answers)):
                generate_length += outputs['max_sequence_length']
                rewards=[]
                new_messages=[]
                for idx_k in range(repeated_generate_nums):
                    idx_sequence=idx_batch*repeated_generate_nums+idx_k
                    decoded_sequence=outputs['decoded_sequences'][idx_sequence]
                    ground_truth=answers[idx_batch]

                    new_message=deepcopy(messages[idx_batch])
                    new_message.append({
                        "role": "assistant",
                        "content":decoded_sequence
                    })

                    format_reward=format_reward_func([decoded_sequence])
                    answer_reward=accuracy_reward_func([decoded_sequence],[ground_truth])
                    reward=0.2*format_reward[0]+answer_reward[0]

                    rewards.append(reward)
                    new_messages.append(new_message)


                rewards=np.array(rewards)
                if rewards.std()==0:

                    if rewards[0]>=1.0:
                        batch_data['ignore_due_correct']+=1
                    else:
                        batch_data['ignore_due_incorrect']+=1

                    continue

                std_rewards=(rewards-rewards.mean())/rewards.std()
                batch_data['messages']+=new_messages
                batch_data['rewards']+=rewards.tolist()
                batch_data['std_rewards']+=std_rewards.tolist()
                used_items+=1

            generate_length /= len(answers)

            batch_data['length_stdev'].append(length_stdev)
            batch_data['length_range'].append(length_range)
            batch_data['length_cv'].append(length_cv)
            batch_data['last_generate_time_cost'].append(outputs['total_time_cost'])
            batch_data['last_acc_length'].append(outputs['total_acc_length'])
            batch_data['last_decoded_token_num'].append(outputs['total_decoded_token_num'])
            accepted_draft_tokens = int(outputs.get('total_accepted_draft_tokens', 0))
            proposed_draft_tokens = int(outputs.get('total_proposed_draft_tokens', 0))
            batch_data['last_accepted_draft_tokens'].append(accepted_draft_tokens)
            batch_data['last_proposed_draft_tokens'].append(proposed_draft_tokens)
            batch_data['last_generate_length'].append(generate_length)
            batch_data['prefill_time_cost']+=outputs['prefill_time_cost']
            batch_data['target_time_cost']+=outputs['target_time_cost']
            batch_data['draft_time_cost']+=outputs['draft_time_cost']
            batch_data['check_time_cost']+=outputs['check_time_cost']

            batch_data['generate_time_cost']+=outputs['total_time_cost']
            batch_data['total_acc_length']+=outputs['total_acc_length']
            batch_data['total_decoded_token_num']+=outputs['total_decoded_token_num']
            batch_data['total_accepted_draft_tokens']+=accepted_draft_tokens
            batch_data['total_proposed_draft_tokens']+=proposed_draft_tokens
            for name in OPD_COUNTER_NAMES+GENERATION_COUNTER_NAMES:
                value=float(outputs.get(name,0.))
                if name=='opd_active_rows_max':
                    batch_data[name]=max(batch_data.get(name,0.),value)
                    batch_data['opd_interval_active_rows_max']=max(batch_data.get('opd_interval_active_rows_max',0.),value)
                else:batch_data[name]=batch_data.get(name,0.)+value
            batch_data['opd_profile_time_ms']+=float(outputs.get('opd_profile_time_ms',0.))
            batch_data['generate_length']+=generate_length
            trace_rollout_count += 1
            batch_data['trace_rollout_count'] = int(batch_data.get('trace_rollout_count', 0)) + 1
            batch_data['used_items'] = int(used_items)
            source_grpo_step = used_items // max(1, batch_size * accumulation_steps)
            local_grpo_step = max(0, int(source_grpo_step - trace_start_step))
            rollout_log = {
                "phase": "rollout",
                "epoch": int(epoch + 1),
                "batch": int(i),
                "grpo_step": int(local_grpo_step),
                "source_grpo_step": int(source_grpo_step),
                "rollout_count": int(trace_rollout_count),
                "used_items": int(used_items),
                "draft_sparse_tv": None if draft_sparse_tv is None else float(draft_sparse_tv),
                "draft_sparse_kl": None if draft_sparse_kl is None else float(draft_sparse_kl),
                "draft_sparse_count": int(draft_sparse_count),
                "draft_update_committed": bool(draft_update_committed),
                "draft_updates_cumulative": int(draft_step - trace_start_draft_step),
                "drift_topk": int(drift_topk),
                "drift_temperature": float(drift_temperature),
                "draft_lr_multiplier": float(draft_lr_multiplier),
                "fastgrpo_ablation": bool(fastgrpo_ablation),
            }
            batch=[]

            cur_acc_length = (
                batch_data['total_acc_length'] / max(batch_data['total_decoded_token_num'], 1)
            )
            cur_draft_acceptance_rate = (
                batch_data['total_accepted_draft_tokens'] /
                max(batch_data['total_proposed_draft_tokens'], 1)
            )
            batch_bar.set_postfix(
                step=step,
                acc=f"{cur_acc_length:.3f}",
                macc=f"{cur_draft_acceptance_rate:.3f}",
                gen=f"{outputs['total_time_cost'] / 60:.2f}m",
                pending=f"{len(batch_data['messages'])}/{batch_size * accumulation_steps}",
                phase="rollout",
                refresh=False,
            )

            all_ranks_ready = len(batch_data['messages']) > 0
            if dist.is_initialized():
                ready_tensor = torch.tensor(
                    int(all_ranks_ready), device=model.target_model.device, dtype=torch.int32
                )
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                all_ranks_ready = bool(ready_tensor.item())
            if not all_ranks_ready:
                continue

            text=tokenizer.apply_chat_template(batch_data['messages'],tokenize=False,add_generation_prompt=False)
            text=tokenizer(text,padding=False)
            loss_mask=[]

            for idx_message, message in enumerate(batch_data['messages']):
                prompt_text=tokenizer.apply_chat_template(message[:-1],tokenize=False,add_generation_prompt=True)
                prompt_text=tokenizer.encode(prompt_text)
                cur_loss_mask=[0]*(len(prompt_text)-1)+[1]*(len(text.input_ids[idx_message])-len(prompt_text)+1)
                loss_mask.append(cur_loss_mask)

            input_ids=text.input_ids
            attention_mask=text.attention_mask

            sorted_pairs = sorted(
                zip(input_ids, attention_mask, loss_mask),
                key=lambda x: len(x[0]),
                reverse=False
            )

            input_ids_sorted, attention_mask_sorted, loss_mask_sorted = zip(*sorted_pairs)

            input_ids, attention_mask, loss_mask = list(input_ids_sorted), list(attention_mask_sorted), list(loss_mask_sorted)

            synchronized_used_items = int(used_items)
            if dist.is_initialized():
                used_tensor = torch.tensor(
                    synchronized_used_items, device=model.target_model.device, dtype=torch.long
                )
                dist.all_reduce(used_tensor, op=dist.ReduceOp.MIN)
                synchronized_used_items = int(used_tensor.item())
            step = synchronized_used_items // (batch_size * accumulation_steps)
            step_metrics.advance(step)
            batch_old_logps=[]
            batch_ref_logps=[]
            batch_data['reward_sum'] += float(sum(batch_data['rewards']))
            batch_data['reward_count'] += int(len(batch_data['rewards']))

            for grpo_iteration in range(grpo_iteration_num):
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_time_start=time.time()
                target_phase_ticket = phase_timings.begin('target')

                cur_max_length=0
                device=model.target_model.device
                microbatch_index=0

                cur_input_ids=[]
                cur_attention_mask=[]
                cur_loss_mask=[]
                cur_rewards=[]

                for j in range(len(batch_data['messages'])):

                    if ((max(cur_max_length, len(input_ids[j])) * (len(cur_input_ids)+1)<=max_training_token and
                        (len(input_ids[j])-cur_max_length)*len(cur_input_ids)<=max_training_padding_gap) or
                        len(cur_input_ids)==0):
                        cur_max_length=max(cur_max_length, len(input_ids[j]))

                        cur_input_ids.append(input_ids[j])
                        cur_attention_mask.append(attention_mask[j])
                        cur_loss_mask.append(loss_mask[j])
                        cur_rewards.append(batch_data['std_rewards'][j])

                    else:

                        cur_batch=len(cur_input_ids)
                        for idx_seq in range(cur_batch):

                            cur_len=len(cur_input_ids[idx_seq])
                            padding_len=cur_max_length-cur_len

                            if padding_len>0:

                                cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                                cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                                cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len

                        cur_input_ids=torch.tensor(cur_input_ids, device=device)
                        cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
                        cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
                        cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                        old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                        ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                        loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                            model,
                            cur_input_ids,
                            cur_attention_mask,
                            cur_loss_mask,
                            cur_rewards,
                            epsilon,
                            beta,
                            grpo_iteration,
                            old_logps=old_logps,
                            ref_logps=ref_logps,
                            chunk_size=logps_chunk_size,
                            loss_scale=1.0 / max(len(batch_data['messages']), 1),
                        )
                        batch_data['target_loss_sum'] += float(loss)
                        batch_data['target_loss_count'] += int(cur_batch)

                        if grpo_iteration==0:
                            batch_old_logps.append(old_logps)
                            batch_ref_logps.append(ref_logps)
                        microbatch_index += 1
                        del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards

                        cur_input_ids=[input_ids[j]]
                        cur_attention_mask=[attention_mask[j]]
                        cur_loss_mask=[loss_mask[j]]
                        cur_rewards=[batch_data['std_rewards'][j]]

                        cur_max_length=len(input_ids[j])

                cur_batch=len(cur_input_ids)
                for idx_seq in range(cur_batch):

                    cur_len=len(cur_input_ids[idx_seq])
                    padding_len=cur_max_length-cur_len

                    if padding_len>0:

                        cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                        cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                        cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len

                cur_input_ids=torch.tensor(cur_input_ids, device=device)
                cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
                cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
                cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                    model,
                    cur_input_ids,
                    cur_attention_mask,
                    cur_loss_mask,
                    cur_rewards,
                    epsilon,
                    beta,
                    grpo_iteration,
                    old_logps=old_logps,
                    ref_logps=ref_logps,
                    chunk_size=logps_chunk_size,
                    loss_scale=1.0 / max(len(batch_data['messages']), 1),
                )
                batch_data['target_loss_sum'] += float(loss)
                batch_data['target_loss_count'] += int(cur_batch)

                if grpo_iteration==0:
                    batch_old_logps.append(old_logps)
                    batch_ref_logps.append(ref_logps)
                microbatch_index += 1
                del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards


                _sync_gradients(model.target_model)
                optimizer_target.step()
                optimizer_target.zero_grad(set_to_none=True)
                phase_timings.end(target_phase_ticket)

                if (
                    analysis_enabled
                    and _analysis_boundary(int(step))
                    and int(step) not in analysis_completed
                ):
                    analysis_main_rng = _analysis_rng_state()
                    target_was_training = model.target_model.training
                    draft_was_training = model.draft_model.training
                    boundary_dir = Path(policy_lag_output_dir) / 'boundaries' / f'step_{int(step)}'
                    boundary_dir.mkdir(parents=True, exist_ok=True)
                    current_policy_id = state_digest(_target_lora_state_dict(model.target_model))
                    current_policy_state = {
                        key: value.detach().cpu().clone()
                        for key, value in _target_lora_state_dict(model.target_model).items()
                    }
                    target_digest_before_analysis = current_policy_id
                    new_teacher_logits = _teacher_prefix_hidden(eval_batch_for_analysis)
                    teacher_tv = _teacher_tv_from_hidden(
                        analysis_old_teacher_logits,
                        new_teacher_logits,
                        eval_batch_for_analysis['attention_mask'],
                    )
                    # Fresh R_{t+1}: same training prompts, current target, separate RNG;
                    # it is never added to batch_data/the GRPO replay buffer.
                    _seed_everything(trace_seed + int(step) * 1009)
                    with torch.inference_mode():
                        fresh_outputs = speculative_generate(
                            model=model,
                            input_ids=analysis_train_input_ids.to('cuda'),
                            attention_mask=analysis_train_attention_mask.to('cuda'),
                            tokenizer=tokenizer,
                            do_sample=True,
                            max_length=max_length,
                            repeated_generate_nums=repeated_generate_nums,
                            temperature=temperature,
                            top_p=top_p,
                            verification_capacity=verification_capacity,
                            max_draft_token_length=max_draft_token_length,
                            max_draft_k=max_draft_k,
                            max_verification_num=max_verification_num,
                            min_draft_token_length=min_draft_token_length,
                            draft_token_length_c=draft_token_length_c,
                            return_all_draft_input=True,
                            statistical_time=False,
                            **opd_eval_kwargs,
                        )
                    old_available = sum(
                        max(int(ids.shape[-1]) - int(analysis_train_attention_mask[idx // repeated_generate_nums].sum()) - 1, 0)
                        for idx, ids in enumerate(outputs['all_draft_input_ids'])
                    )
                    fresh_available = sum(
                        max(int(ids.shape[-1]) - int(analysis_train_attention_mask[idx // repeated_generate_nums].sum()) - 1, 0)
                        for idx, ids in enumerate(fresh_outputs['all_draft_input_ids'])
                    )
                    configured_budget = int(args.analysis_training_token_budget)
                    common_budget = min(old_available, fresh_available)
                    if configured_budget > 0:
                        common_budget = min(common_budget, configured_budget)
                    if common_budget <= 0:
                        raise RuntimeError(f'boundary {step} has no common valid supervised-token budget')
                    branch_update_steps = int(args.analysis_draft_update_steps)
                    if common_budget < branch_update_steps:
                        raise RuntimeError(
                            f'training token budget {common_budget} is smaller than '
                            f'optimizer steps {branch_update_steps}'
                        )
                    torch.save({
                        'format': 'fastgrpo_policy_lag_base_v1',
                        'policy_step': int(step),
                        'policy_t_id': analysis_old_policy_id,
                        'policy_t_plus_1_id': current_policy_id,
                        'policy_t_lora_state_dict': analysis_old_policy_state,
                        'policy_t_plus_1_lora_state_dict': current_policy_state,
                        'draft_state_dict': analysis_base_draft,
                        'optimizer_state_dict': analysis_base_optimizer,
                        'scheduler_state_dict': None,
                        'feature_layers': list(model.feature_layers),
                        'specforge': model.checkpoint_metadata(),
                    }, boundary_dir / 'phi_base.pt')

                    branch_states = {}
                    branch_optimizer_states = {}
                    branch_losses = {}
                    for branch_name, branch_outputs, feature_policy in (
                        ('stale', outputs, analysis_old_policy_id),
                        ('fresh', fresh_outputs, current_policy_id),
                    ):
                        model.draft_model.load_state_dict(analysis_base_draft, strict=True)
                        optimizer_draft.load_state_dict(deepcopy(analysis_base_optimizer))
                        if state_digest(model.draft_model.state_dict()) != state_digest(analysis_base_draft):
                            raise RuntimeError(f'{branch_name} did not start from phi_base')
                        if state_digest(optimizer_draft.state_dict()) != state_digest(analysis_base_optimizer):
                            raise RuntimeError(f'{branch_name} optimizer did not start from phi_base')
                        optimizer_draft.zero_grad(set_to_none=True)
                        consumed_tokens = 0
                        loss_totals = [0.0, 0.0]
                        for update_index in range(branch_update_steps):
                            step_budget = common_budget // branch_update_steps
                            if update_index < common_budget % branch_update_steps:
                                step_budget += 1
                            losses = training_draft_model(
                                model,
                                branch_outputs,
                                analysis_train_attention_mask.to('cuda'),
                                token_budget=step_budget,
                            )
                            consumed_tokens += int(losses[4])
                            loss_totals[0] += float(losses[0])
                            loss_totals[1] += float(losses[1])
                            optimizer_draft.step()
                            optimizer_draft.zero_grad(set_to_none=True)
                        if consumed_tokens != common_budget:
                            raise RuntimeError(
                                f'{branch_name} consumed {consumed_tokens} tokens, expected {common_budget}'
                            )
                        branch_states[branch_name] = {
                            key: value.detach().cpu().clone()
                            for key, value in model.draft_model.state_dict().items()
                        }
                        branch_optimizer_states[branch_name] = deepcopy(optimizer_draft.state_dict())
                        branch_losses[branch_name] = loss_totals
                        torch.save({
                            'format': 'fastgrpo_policy_lag_branch_v1',
                            'branch': branch_name,
                            'policy_step': int(step),
                            'feature_policy_version': feature_policy,
                            'base_digest': state_digest(analysis_base_draft),
                            'draft_state_dict': branch_states[branch_name],
                            'optimizer_state_dict': branch_optimizer_states[branch_name],
                            'scheduler_state_dict': None,
                            'actual_training_token_count': int(common_budget),
                            'optimizer_steps': branch_update_steps,
                            'losses': branch_losses[branch_name],
                        }, boundary_dir / f'draft_{branch_name}.pt')

                    stale_rows = _evaluate_analysis_branch('stale', branch_states['stale'], eval_batch_for_analysis, step)
                    fresh_rows = _evaluate_analysis_branch('fresh', branch_states['fresh'], eval_batch_for_analysis, step)
                    analysis_per_response.extend(stale_rows + fresh_rows)
                    for sampling_seed in analysis_seeds:
                        seed_stale = [row for row in stale_rows if row['seed'] == sampling_seed]
                        seed_fresh = [row for row in fresh_rows if row['seed'] == sampling_seed]
                        boot = bootstrap_delta_by_prompt(
                            seed_stale,
                            seed_fresh,
                            seed=trace_seed + sampling_seed + int(step),
                            samples=int(args.analysis_bootstrap_samples),
                        )
                        for branch_name, records in (('stale', seed_stale), ('fresh', seed_fresh)):
                            aal, _, rounds, generated = weighted_aal(records)
                            analysis_summaries.append(BranchSummary(
                                policy_step=int(step),
                                seed=int(sampling_seed),
                                branch=branch_name,
                                aal=aal,
                                delta_aal=boot['delta_aal'] if branch_name == 'fresh' else None,
                                verification_rounds=rounds,
                                generated_tokens=generated,
                                actual_training_token_count=int(common_budget),
                                optimizer_steps=branch_update_steps,
                                policy_checkpoint_id=current_policy_id,
                                draft_checkpoint_id=state_digest(branch_states[branch_name]),
                                feature_policy_version=(analysis_old_policy_id if branch_name == 'stale' else current_policy_id),
                                teacher_shift_tv=teacher_tv,
                                ci_low=boot['delta_aal_ci_low'] if branch_name == 'fresh' else None,
                                ci_high=boot['delta_aal_ci_high'] if branch_name == 'fresh' else None,
                            ))
                    write_results(Path(policy_lag_output_dir), analysis_per_response, analysis_summaries)
                    # Continue the real trajectory with stale, not fresh. Restore all
                    # stochastic/module state so analysis cannot perturb GRPO.
                    model.draft_model.load_state_dict(branch_states['stale'], strict=True)
                    optimizer_draft.load_state_dict(branch_optimizer_states['stale'])
                    draft_step += branch_update_steps - 1
                    _restore_analysis_rng(analysis_main_rng)
                    model.draft_model.train(draft_was_training)
                    model.target_model.train(target_was_training)
                    if state_digest(_target_lora_state_dict(model.target_model)) != target_digest_before_analysis:
                        raise RuntimeError('target policy changed during policy-lag evaluation')
                    analysis_completed.add(int(step))
                    with (boundary_dir / 'complete.json').open('w', encoding='utf-8') as stream:
                        json.dump({
                            'policy_step': int(step),
                            'base_digest': state_digest(analysis_base_draft),
                            'common_training_token_budget': int(common_budget),
                            'optimizer_steps_per_branch': branch_update_steps,
                            'teacher_shift_tv': teacher_tv,
                            'main_branch': 'stale',
                        }, stream, indent=2, sort_keys=True)

                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_time_elapsed=time.time()-train_time_start
                batch_data['last_train_time_cost'].append(train_time_elapsed)
                batch_data['train_time_cost']+=train_time_elapsed
                batch_data['last_mean_rewards'].append(sum(batch_data['rewards'])/len(batch_data['rewards']))
                batch_data['mean_rewards']+=sum(batch_data['rewards'])/len(batch_data['rewards'])

                real_sample_num=sample_num*accumulation_steps
                last_accepted_draft_tokens=sum(batch_data['last_accepted_draft_tokens'][-real_sample_num:])
                last_proposed_draft_tokens=sum(batch_data['last_proposed_draft_tokens'][-real_sample_num:])
                draft_acceptance_rate=(
                    batch_data['total_accepted_draft_tokens'] /
                    max(batch_data['total_proposed_draft_tokens'], 1)
                )
                last_draft_acceptance_rate=last_accepted_draft_tokens / max(last_proposed_draft_tokens, 1)
                average_accept_length=(
                    batch_data['total_acc_length'] /
                    max(batch_data['total_decoded_token_num'], 1)
                )
                accepted_tokens_per_medusa_step=(
                    batch_data['total_accepted_draft_tokens'] /
                    max(batch_data['total_decoded_token_num'], 1)
                )

                avg_logs = {
                    "phase":"target_train",
                    "epoch":epoch+1,
                    "step": step,
                    "grpo_step": int(max(0, step - trace_start_step)),
                    "source_grpo_step": int(step),
                    "rollout_count": int(trace_rollout_count),
                    "used_items" : used_items ,
                    "train_dataset_full_size": full_train_samples,
                    "train_dataset_selected_size": selected_train_samples,
                    "train_data_fraction": train_data_fraction,
                    "train_subset_seed": train_subset_seed,
                    "max_train_samples": max_train_samples,
                    "logps_chunk_size": logps_chunk_size,
                    f"length_range" : round(mean(batch_data['length_range']),4),
                    f"length_cv" : round(mean(batch_data['length_cv']),4) ,
                    f"length_stdev" : round(mean(batch_data['length_stdev']),4) ,
                    "grpo_iteration":grpo_iteration+1,
                    "used_time": round(_cumulative_wall_time()/60, 3),
                    f"last_{sample_num}_generate_time_cost":round(sum(batch_data['last_generate_time_cost'][-real_sample_num:])/60,3),
                    f"last_{sample_num}_train_time_cost": round(sum(batch_data['last_train_time_cost'][-real_sample_num:]) / 60, 3),
                    f"last_{sample_num}_acc_length":round(sum(batch_data['last_acc_length'][-real_sample_num:]) / sum(batch_data['last_decoded_token_num'][-real_sample_num:]),4),
                    f"last_{sample_num}_draft_acceptance_rate":round(last_draft_acceptance_rate,4),
                    f"last_{sample_num}_medusa_acceptance_rate":round(last_draft_acceptance_rate,4),
                    f"last_{sample_num}_mean_rewards": round(sum(batch_data['last_mean_rewards'][-real_sample_num:]) / len(batch_data['last_mean_rewards'][-real_sample_num:]), 3),
                    f"last_{sample_num}_mean_length": round(sum(batch_data['last_generate_length'][-real_sample_num:]) / len(batch_data['last_generate_length'][-real_sample_num:]), 3),

                    "ignore_due_correct_cur_epoch":batch_data['ignore_due_correct'],
                    "ignore_due_incorrect_cur_epoch":batch_data['ignore_due_incorrect'],
                    "generate_time_cost":round(batch_data['generate_time_cost']/60,3),
                    "average_acc_length":round(average_accept_length,4),
                    "average_accept_length":round(average_accept_length,4),
                    "accepted_tokens_per_medusa_step":round(accepted_tokens_per_medusa_step,4),
                    "total_rollout_tokens":int(batch_data['total_rollout_tokens']),
                    "total_accepted_draft_tokens":int(batch_data['total_accepted_draft_tokens']),
                    "total_proposed_draft_tokens":int(batch_data['total_proposed_draft_tokens']),
                    "total_accepted_medusa_tokens":int(batch_data['total_accepted_draft_tokens']),
                    "total_proposed_medusa_tokens":int(batch_data['total_proposed_draft_tokens']),
                    "draft_acceptance_rate":round(draft_acceptance_rate,4),
                    "medusa_acceptance_rate":round(draft_acceptance_rate,4),
                    "prefill_time_cost":round(batch_data['prefill_time_cost']/60,3),
                    "target_time_cost":round(batch_data['target_time_cost']/60,3),
                    "draft_time_cost":round(batch_data['draft_time_cost']/60,3),
                    "train_time_cost":round(batch_data['train_time_cost']/60,3),
                    "check_time_cost":round(batch_data['check_time_cost']/60,3),
                    "mean_reward":round(batch_data['mean_rewards']/used_items,4),
                    "draft_sparse_tv": None if draft_sparse_tv is None else float(draft_sparse_tv),
                    "draft_sparse_kl": None if draft_sparse_kl is None else float(draft_sparse_kl),
                    "draft_sparse_count": int(draft_sparse_count),
                    "draft_update_committed": bool(draft_update_committed),
                    "draft_updates_cumulative": int(draft_step - trace_start_draft_step),
                    "draft_lr_multiplier": float(draft_lr_multiplier),
                    "fastgrpo_ablation": bool(fastgrpo_ablation),
                    "method": method,
                    "opd_updates": int(batch_data['opd_updates']),

                    "draft_train_time_cost":round(batch_data['draft_train_time_cost']/60,3) if is_train_draft else 0,
                    f"last_{sample_num}_draft_loss1":round(sum(batch_data['last_draft_loss1'][-real_sample_num:])/len(batch_data['last_draft_loss1'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0,
                    f"last_{sample_num}_draft_loss2":round(sum(batch_data['last_draft_loss2'][-real_sample_num:])/len(batch_data['last_draft_loss2'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0
                }

                if grpo_iteration == grpo_iteration_num - 1:
                    wall_elapsed = _cumulative_wall_time()
                    global_metrics = _aggregate_job_metrics(
                        batch_data, model.target_model.device, wall_elapsed,
                        include_opd_profile=_as_bool(args.opd_profile),
                    )
                    step_snapshot = completed_step_snapshot(
                        global_metrics, batch_data, phase_timings, model.target_model.device,
                        _cumulative_wall_time())
                    wall_elapsed = step_snapshot['cumulative_wall_time_s']
                    avg_logs.update({
                        "used_time": round(wall_elapsed / 60.0, 3),
                        "generate_time_cost": round(global_metrics['generate_time_cost'] / 60.0, 3),
                        "train_time_cost": round(global_metrics['train_time_cost'] / 60.0, 3),
                        "draft_train_time_cost": round(global_metrics['draft_train_time_cost'] / 60.0, 3),
                        "average_acc_length": round(global_metrics['average_accept_length'], 4),
                        "average_accept_length": round(global_metrics['average_accept_length'], 4),
                        "accepted_tokens_per_medusa_step": round(global_metrics['accepted_tokens_per_medusa_step'], 4),
                        "draft_acceptance_rate": round(global_metrics['draft_acceptance_rate'], 4),
                        "medusa_acceptance_rate": round(global_metrics['draft_acceptance_rate'], 4),
                        "total_rollout_tokens": int(global_metrics['total_rollout_tokens']),
                        "total_acc_length": int(global_metrics['total_acc_length']),
                        "total_decoded_token_num": int(global_metrics['total_decoded_token_num']),
                        "total_accepted_draft_tokens": int(global_metrics['total_accepted_draft_tokens']),
                        "total_proposed_draft_tokens": int(global_metrics['total_proposed_draft_tokens']),
                        "total_accepted_medusa_tokens": int(global_metrics['total_accepted_draft_tokens']),
                        "total_proposed_medusa_tokens": int(global_metrics['total_proposed_draft_tokens']),
                        "mean_reward": round(global_metrics['mean_reward'], 4),
                        "target_loss": float(global_metrics['target_loss']),
                        "draft_loss1": float(global_metrics['draft_loss1']),
                        "draft_loss2": float(global_metrics['draft_loss2']),
                        "opd_updates": int(global_metrics['opd_updates']),
                        "tokens_per_s": float(global_metrics['tokens_per_s']),
                    })
                    if _as_bool(args.opd_profile):
                        avg_logs["opd_profile_time_ms"] = float(
                            global_metrics['opd_profile_time_ms']
                        )
                    step_metrics.submit(step, step_snapshot, avg_logs)
                    batch_data['_step_metrics_state'] = step_metrics.state_dict()

                postfix = {
                    "step": step,
                    "acc": avg_logs["average_accept_length"],
                    "macc": avg_logs["medusa_acceptance_rate"],
                    "gen": f"{avg_logs['last_' + str(sample_num) + '_generate_time_cost']:.2f}m",
                    "train": f"{avg_logs['last_' + str(sample_num) + '_train_time_cost']:.2f}m",
                    "reward": avg_logs["mean_reward"],
                    "phase": "GRPO",
                }
                if step % log_interval == 0:
                    batch_bar.set_postfix(postfix, refresh=False)
                    epoch_bar.set_postfix(postfix, refresh=False)

                torch.cuda.empty_cache()

            batch_data['messages'].clear()
            batch_data['rewards'].clear()
            batch_data['std_rewards'].clear()
            batch_old_logps.clear()
            batch_ref_logps.clear()

            if is_main_process and step%500==0 and step!=0:
                model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
                model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')

            if (
                save_checkpoint_steps > 0
                and (step - trace_start_step) > 0
                and (step - trace_start_step) % save_checkpoint_steps == 0
                and step != last_checkpoint_step
            ):
                batch_data['_rollout_metrics_state']=rollout_metrics.next_state(iter_outputs)
                iteration_checkpoint=True
                rollout_metrics.flush()
                save_training_checkpoint(
                    checkpoint_dir,
                    model=model,
                    optimizer_target=optimizer_target,
                    optimizer_draft=optimizer_draft,
                    epoch=epoch,
                    next_batch=i + 1,
                    step=step,
                    used_items=used_items,
                    draft_step=draft_step,
                    draft_accumulated_step=draft_accumulated_step,
                    batch_data=batch_data,
                    keep_last=keep_last_checkpoints,
                    cumulative_elapsed_time_s=_cumulative_wall_time(),
                )
                last_checkpoint_step = step

            completed_grpo_steps = max(0, int(step - trace_start_step))
            if max_grpo_steps > 0 and completed_grpo_steps >= max_grpo_steps:
                stop_requested = True
                if step != last_checkpoint_step:
                    batch_data['_rollout_metrics_state']=rollout_metrics.next_state(iter_outputs)
                    iteration_checkpoint=True
                    rollout_metrics.flush()
                    save_training_checkpoint(
                        checkpoint_dir,
                        model=model,
                        optimizer_target=optimizer_target,
                        optimizer_draft=optimizer_draft,
                        epoch=epoch,
                        next_batch=i + 1,
                        step=step,
                        used_items=used_items,
                        draft_step=draft_step,
                        draft_accumulated_step=draft_accumulated_step,
                        batch_data=batch_data,
                        keep_last=keep_last_checkpoints,
                        cumulative_elapsed_time_s=_cumulative_wall_time(),
                    )
                    last_checkpoint_step = step
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "phase": "trace_stop",
                        "grpo_step": int(completed_grpo_steps),
                        "source_grpo_step": int(step),
                        "rollout_count": int(trace_rollout_count),
                        "max_grpo_steps": int(max_grpo_steps),
                        "reason": "max_grpo_steps",
                    }) + '\n')
                break

        finally:
            finish_rollout_iteration()
            if iteration_checkpoint or sys.exc_info()[0] is not None:
                rollout_metrics.flush()

    if stop_requested:
        break


rollout_metrics.close()
signal.signal(signal.SIGTERM,_previous_sigterm_handler)

# Drain timers and account for trailing rollouts (including reward-filtered
# responses) before finalizing the last inherited GRPO label. The ordinary
# metric transfer completes the main stream; no new CUDA synchronize is used.
batch_data['used_items'] = int(used_items)
training_end_metrics = _aggregate_job_metrics(
    batch_data, model.target_model.device, _cumulative_wall_time(),
    include_opd_profile=_as_bool(args.opd_profile),
)
training_end_snapshot = completed_step_snapshot(
    training_end_metrics, batch_data, phase_timings, model.target_model.device,
    _cumulative_wall_time())
if step_metrics.pending is not None:
    step_metrics.submit(step_metrics.pending['step'], training_end_snapshot,
                        step_metrics.pending['extras'])
step_metrics.flush()
batch_data['_step_metrics_state'] = step_metrics.state_dict()
if is_main_process:
    model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
    model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')
if checkpoint_dir and not stop_requested:
    save_training_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer_target=optimizer_target,
        optimizer_draft=optimizer_draft,
        epoch=num_epochs,
        next_batch=0,
        step=step,
        used_items=used_items,
        draft_step=draft_step,
        draft_accumulated_step=draft_accumulated_step,
        batch_data=batch_data,
        keep_last=keep_last_checkpoints,
        cumulative_elapsed_time_s=_cumulative_wall_time(),
    )

total_wall_time = _cumulative_wall_time()
batch_data['used_items'] = int(used_items)
final_metrics = _aggregate_job_metrics(
    batch_data, model.target_model.device, total_wall_time,
    include_opd_profile=_as_bool(args.opd_profile),
)
final_snapshot = completed_step_snapshot(
    final_metrics, batch_data, phase_timings, model.target_model.device,
    _cumulative_wall_time())
total_wall_time = final_snapshot['cumulative_wall_time_s']
final_average_accept_length = final_metrics['average_accept_length']
final_medusa_acceptance_rate = final_metrics['draft_acceptance_rate']
final_accepted_tokens_per_medusa_step = final_metrics['accepted_tokens_per_medusa_step']
summary = {
    "run_name": version_name,
    "final_step": int(step),
    "used_items": int(final_metrics['used_items']),
    "draft_step": int(draft_step),
    "completed_grpo_steps": int(max(0, step - trace_start_step)),
    "max_grpo_steps": int(max_grpo_steps),
    "stopped_by_max_grpo_steps": bool(stop_requested),
    "rollout_count": int(final_metrics['trace_rollout_count']),
    "draft_updates_committed": int(draft_step - trace_start_draft_step),
    "draft_sparse_tv": float(final_metrics['draft_sparse_tv']),
    "draft_sparse_kl": float(final_metrics['draft_sparse_kl']),
    "draft_sparse_count": int(final_metrics['draft_sparse_count']),
    "draft_lr_multiplier": float(draft_lr_multiplier),
    "draft_train_mode": args.draft_train_mode,
    "draft_train_max_batch_size": int(args.draft_train_max_batch_size),
    "draft_train_max_tokens": int(args.draft_train_max_tokens),
    "draft_train_max_padding_ratio": float(args.draft_train_max_padding_ratio),
    "effective_draft_lrs": effective_draft_lrs,
    "fastgrpo_ablation": bool(fastgrpo_ablation),
    "method": method,
    "opd_rank": int(args.opd_rank),
    "opd_proposal_mode": os.environ.get('OPD_PROPOSAL_MODE','auto'),
    "opd_dense_implementation": os.environ.get('OPD_DENSE_IMPLEMENTATION','auto'),
    "opd_proposal_profile": os.environ.get('OPD_PROPOSAL_PROFILE',''),
    "opd_topk": int(args.opd_topk),
    "opd_update_stream": _as_bool(args.opd_update_stream),
    "opd_visited_weight": args.opd_visited_weight,
    "opd_frontier_weight": args.opd_frontier_weight,
    "opd_fast_lr": float(args.opd_fast_lr),
    "opd_diagnostics": _as_bool(args.opd_diagnostics),
    "opd_backend_requested": args.opd_backend,
    "opd_updates": int(final_metrics['opd_updates']),
    "opd_backend_effective": effective_opd_backend,
    "train_dataset_full_size": int(full_train_samples),
    "train_dataset_selected_size": int(selected_train_samples),
    "dataset_path": str(args.dataset_path),
    "eval_dataset_path": str(args.eval_dataset_path),
    "train_data_fraction": float(train_data_fraction),
    "train_subset_seed": int(train_subset_seed),
    "max_train_samples": int(max_train_samples),
    "logps_chunk_size": int(logps_chunk_size),
    "total_generate_time_s": float(final_metrics['generate_time_cost']),
    "total_train_time_s": float(final_metrics['train_time_cost']),
    "total_draft_train_time_s": float(final_metrics['draft_train_time_cost']) if is_train_draft else 0.0,
    "total_wall_time_s": float(total_wall_time),
    "total_rollout_tokens": int(final_metrics['total_rollout_tokens']),
    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    "generation_tokens_per_s": float(final_metrics['tokens_per_s']),
    "tokens_per_s": float(final_metrics['tokens_per_s']),
    "average_accept_length": float(final_average_accept_length),
    "average_acc_length": float(final_average_accept_length),
    "accepted_tokens_per_medusa_step": float(final_accepted_tokens_per_medusa_step),
    "medusa_acceptance_rate": float(final_medusa_acceptance_rate),
    "draft_acceptance_rate": float(final_medusa_acceptance_rate),
    "total_acc_length": int(final_metrics['total_acc_length']),
    "total_decoded_token_num": int(final_metrics['total_decoded_token_num']),
    "total_verify_rounds": int(final_metrics['total_decoded_token_num']),
    "total_accepted_draft_tokens": int(final_metrics['total_accepted_draft_tokens']),
    "total_proposed_draft_tokens": int(final_metrics['total_proposed_draft_tokens']),
    "total_accepted_medusa_tokens": int(final_metrics['total_accepted_draft_tokens']),
    "total_proposed_medusa_tokens": int(final_metrics['total_proposed_draft_tokens']),
    "mean_reward": float(final_metrics['mean_reward']),
    "target_loss": float(final_metrics['target_loss']),
    "draft_loss1": float(final_metrics['draft_loss1']),
    "draft_loss2": float(final_metrics['draft_loss2']),
    "ignore_due_correct": int(final_metrics['ignore_due_correct']),
    "ignore_due_incorrect": int(final_metrics['ignore_due_incorrect']),
    "metrics_jsonl": str(log_file),
    "summary_json": str(summary_file),
    "saved_model_dir": f"{saved_model_dir}/step{step}",
    "saved_draft_model_dir": f"{saved_draft_model_dir}/step{step}.pth",
    "saved_statistics_dir": str(saved_statistics_dir),
    "checkpoint_dir": str(checkpoint_dir),
    **final_snapshot,
    "cumulative_generation_tokens_per_s": (
        final_metrics['total_rollout_tokens'] / final_metrics['generate_time_cost']
        if final_metrics['generate_time_cost'] > 0 else 0.0
    ),
    "cumulative_aal": float(final_average_accept_length),
    "cumulative_acceptance_rate": float(final_medusa_acceptance_rate),
    "timing_csv": str(timing_file),
}
if _as_bool(args.opd_profile):
    summary["opd_profile_time_ms"] = float(final_metrics['opd_profile_time_ms'])
summary_text = json.dumps(summary, indent=2, ensure_ascii=True)
if is_main_process:
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    summary_txt = os.path.join(os.path.dirname(summary_file), "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(summary_text)
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
