import os
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
BASH='bash'
MODEL_KEYS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')

def command(script,**overrides):
    env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=PYTHON,**overrides)
    out=subprocess.run([BASH,str(ROOT/script)],cwd=ROOT,env=env,check=True,text=True,capture_output=True).stdout
    return shlex.split(next(x.split(':',1)[1] for x in out.splitlines() if x.startswith('Command  :')))

@pytest.mark.parametrize('key',MODEL_KEYS)
def test_paired_model_launchers_only_differ_by_method_and_keep_paths_hyperparameters(key,tmp_path):
    commands=[]
    for suffix,method in (('','opd_reflex'),('_fastgrpo','fastgrpo')):
        script=f'train_{key}{suffix}.sh'
        args=command(script,RUN_NAME='paired',RUN_DIR=str(tmp_path/'paired'))
        flags=dict(zip(args[args.index('--method')::2],args[args.index('--method')+1::2]))
        assert flags.pop('--method')==method
        for flag,value in (('--target_lr','1e-6'),('--draft_lr','1e-4'),('--draft_accumulation_steps','1'),('--batch_size','8'),
                           ('--accumulation_steps','4'),('--repeated_generate_nums','8'),
                           ('--opd_rank','8'),('--opd_topk','16'),('--opd_fast_lr','0.01'),
                           ('--opd_update_stream','1'),('--opd_profile','0'),('--opd_diagnostics','0'),
                           ('--train_option','simplelr_abel_level3to5'),('--log_interval','1'),
                           ('--attn_implementation','sdpa')):
            assert flags[flag]==value
        assert flags['--adapter_path'].endswith(f'/pretrain/{key}/latest_checkpoint')
        assert flags['--dataset_path']=='/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet'
        source=(ROOT/script).read_text()
        for name in ('CUDA_VISIBLE_DEVICES','DATASET','OPD_FAST_LR','TARGET_LR','DRAFT_LR','BATCH_SIZE','RESPONSES_PER_PROMPT'):
            assert 'export '+name+'=' in source
        commands.append(flags)
    assert commands[0]==commands[1]

def test_all_shells_syntax_and_pretrain_dry_run():
    for script in ROOT.rglob('*.sh'):
        if 'third_party' not in script.parts:subprocess.run([BASH,'-n',str(script)],check=True)
    for key in MODEL_KEYS:
        subprocess.run([BASH,str(ROOT/f'pretrain_{key}.sh')],cwd=ROOT,
            env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=PYTHON),check=True,capture_output=True)

def test_opd_hyperparameter_overrides_reach_cli(tmp_path):
    args=command('train_qwen25_3b.sh',CUDA_VISIBLE_DEVICES='2,3',NPROC_PER_NODE='2',
        DATASET='dapo',TARGET_LR='2e-5',DRAFT_LR='3e-5',BATCH_SIZE='4',ACCUMULATION_STEPS='8',
        RESPONSES_PER_PROMPT='3',OPD_FAST_LR='0.05',OPD_UPDATE_STREAM='1',OPD_RANK='16',OPD_TOPK='32')
    for flag,value in (('--train_option','DAPO-math'),('--target_lr','2e-5'),('--draft_lr','3e-5'),
                       ('--batch_size','4'),('--accumulation_steps','8'),('--repeated_generate_nums','3'),
                       ('--opd_fast_lr','0.05'),('--opd_update_stream','1'),('--opd_rank','16'),('--opd_topk','32')):
        assert args[args.index(flag)+1]==value
    assert '--nproc_per_node=2' in args


def test_stream_zero_still_overrides_new_async_default():
    args=command('train_qwen25_3b.sh',OPD_UPDATE_STREAM='0')
    assert args[args.index('--opd_update_stream')+1]=='0'


@pytest.mark.parametrize('key',MODEL_KEYS)
@pytest.mark.parametrize('suffix',['','_fastgrpo'])
def test_launcher_defaults_new_options_even_if_sourced_config_leaves_them_unset(key,suffix,tmp_path):
    # Reproduce an older/custom server config without new launcher variables.
    # Clear them AFTER wrapper/common exports: the launcher must be self-contained.
    stale=tmp_path/'stale_common.env'
    stale.write_text('source '+shlex.quote(str(ROOT/'configs/_shared/b200_common.env'))+'\n'
        'unset ROLLOUT_LOG_FLUSH_INTERVAL OPD_PROJECTOR_LR OPD_PROPOSAL_PROFILE '
        'OPD_PROPOSAL_MODE OPD_DENSE_IMPLEMENTATION\n')
    args=command(f'train_{key}{suffix}.sh',COMMON_ENV=str(stale),OUTPUT_ROOT=str(tmp_path/'outputs'))
    assert args[args.index('--rollout_log_flush_interval')+1]=='1'
    assert '--opd_projector_lr' not in args
    assert args[args.index('--batch_size')+1]=='8'


@pytest.mark.parametrize('suffix',['','_fastgrpo'])
def test_launcher_preserves_logging_and_projector_overrides(suffix,tmp_path):
    args=command(f'train_qwen3_1p7b{suffix}.sh',ROLLOUT_LOG_FLUSH_INTERVAL='7',
                 OPD_PROJECTOR_LR='2e-5',OUTPUT_ROOT=str(tmp_path/'outputs'))
    assert args[args.index('--rollout_log_flush_interval')+1]=='7'
    if suffix:
        assert '--opd_projector_lr' not in args
    else:
        assert args[args.index('--opd_projector_lr')+1]=='2e-5'


@pytest.mark.parametrize('value',['0','-1','abc'])
def test_invalid_rollout_flush_interval_fails_before_python_training(value):
    result=subprocess.run([BASH,str(ROOT/'train_qwen3_1p7b.sh')],cwd=ROOT,
        env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=PYTHON,ROLLOUT_LOG_FLUSH_INTERVAL=value),
        capture_output=True,text=True)
    assert result.returncode==2
    assert 'ROLLOUT_LOG_FLUSH_INTERVAL must be a positive integer' in result.stderr


def test_tuner_defaults_all_six_even_when_training_model_key_is_exported(tmp_path):
    env=dict(os.environ,PYTHON_BIN=PYTHON,MODEL_KEY='qwen3_1p7b',OUTPUT_ROOT=str(tmp_path/'empty'))
    for name in ('OPD_TUNE_MODELS','OPD_TUNE_OUTPUT','DRAFT_CONFIG','DRAFT_CHECKPOINT','VOCAB_MAPPING'):
        env.pop(name,None)
    result=subprocess.run([BASH,str(ROOT/'scripts/tune_opd_proposals.sh'),'--inspect-only'],
        cwd=ROOT,env=env,check=True,capture_output=True,text=True)
    import json
    payload=json.loads(result.stdout)
    assert [item['model'] for item in payload['skipped']]==list(MODEL_KEYS[:-1])
    assert not (tmp_path/'empty').exists()

def test_real_sweep_dry_run_no_writes_and_quick_settings(tmp_path):
    destination=tmp_path/'not-created'
    out=subprocess.run([BASH,str(ROOT/'scripts/sweep_opd_reflex.sh')],cwd=ROOT,
        env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=PYTHON,BENCH_OUTPUT=str(destination)),
        check=True,capture_output=True,text=True).stdout
    assert '--fast-lrs' in out and '--streams 0\\,1' in out
    assert '--max-length 512' in out and '--responses 8' in out
    assert not destination.exists()

def test_only_two_methods_and_old_lk_arguments_are_gone():
    from helper.method_config import resolve_method
    assert resolve_method('fastgrpo')[0]=='fastgrpo'
    assert resolve_method('opd_reflex')[0]=='opd_reflex'
    with pytest.raises(ValueError):resolve_method('specnaacl')
    source=(ROOT/'grpo_speculative.py').read_text()
    assert '--reflex_lr' not in source and '--reflex_mode' not in source


@pytest.mark.parametrize('suffix', ['', '_fastgrpo'])
@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_target_attention_override_reaches_both_methods(suffix, backend):
    args=command(f'train_qwen3_1p7b{suffix}.sh', ATTENTION_IMPLEMENTATION=backend)
    assert args[args.index('--attn_implementation')+1] == backend

def test_pretrain_wrapper_reports_explicit_backend_topology_and_batch():
    env = dict(os.environ, DRY_RUN="true", PYTHON_BIN=PYTHON,
               PRETRAIN_ATTENTION_BACKEND="sdpa", PRETRAIN_DISTRIBUTED_MODE="auto",
               PRETRAIN_LENGTH_BUCKETING="true", PRETRAIN_DATALOADER_WORKERS="8",
               PRETRAIN_BATCH_SIZE="4", PRETRAIN_ACCUMULATION_STEPS="2",
               NPROC_PER_NODE="4", CUDA_VISIBLE_DEVICES="0,1,2,3")
    result = subprocess.run([BASH, str(ROOT / "pretrain_qwen25_3b.sh")], env=env,
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "attention=sdpa distributed=auto buckets=true workers=8 effective_batch=32" in result.stdout
    wrapper = (ROOT / "scripts/launch/pretrain_model.sh").read_text(encoding="utf-8")
    for name in ("PRETRAIN_ATTENTION_BACKEND", "PRETRAIN_DISTRIBUTED_MODE",
                 "PRETRAIN_LENGTH_BUCKETING", "PRETRAIN_LENGTH_BUCKET_BOUNDARIES",
                 "PRETRAIN_DATALOADER_WORKERS", "PRETRAIN_MAX_LENGTH",
                 "PRETRAIN_COMPACT_TEACHER", "PRETRAIN_OPTIMIZER_CPU_OFFLOAD"):
        assert f'{name}="${name}"' in wrapper
