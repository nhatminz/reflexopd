"""Exact per-step telemetry and checkpointed grouping of inherited labels."""
import ast
import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from helper.opd_reflex import OPD_COUNTER_NAMES,GENERATION_COUNTER_NAMES

from helper.step_metrics import (
    PhaseTimings, StepMetricsWriter, completed_step_snapshot, gpu_memory_stats, step_record,
)


def snapshot(wall, generation, target, draft, tokens, accepted, rounds, draft_accepted, proposed):
    return dict(zip((
        'cumulative_wall_time_s', 'cumulative_generation_time_s', 'cumulative_target_train_time_s',
        'cumulative_draft_train_time_s', 'cumulative_rollout_tokens', 'cumulative_accepted_tokens',
        'cumulative_verification_rounds', 'cumulative_accepted_draft_tokens',
        'cumulative_proposed_draft_tokens',
    ), (wall, generation, target, draft, tokens, accepted, rounds, draft_accepted, proposed)))


A = snapshot(10, 4, 3, 1, 40, 25, 5, 10, 20)
B = snapshot(18, 7, 5, 2, 64, 31, 8, 12, 30)
C = snapshot(30, 11, 9, 4, 100, 41, 13, 18, 45)


def read_rows(directory):
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()
            if json.loads(line).get('phase') == 'target_train']
    with (directory / 'timing.csv').open(newline='') as stream:
        timing = list(csv.DictReader(stream))
    return rows, timing


def test_step_aal_uses_counter_delta_and_generation_throughput_uses_phase_time():
    result = step_record(2, B, A)
    assert result['step_accepted_tokens'] == 6
    assert result['step_verification_rounds'] == 3
    assert result['step_aal'] == 2.0
    assert result['cumulative_aal'] == 31 / 8
    assert result['step_acceptance_rate'] == 2 / 10
    assert result['cumulative_acceptance_rate'] == 12 / 30
    assert result['step_rollout_tokens'] == 24
    assert result['step_generation_tokens_per_s'] == 24 / 3
    assert result['cumulative_generation_tokens_per_s'] == 64 / 7
    assert result['step_wall_time_s'] == 8


def test_repeated_label_has_one_row_after_all_updates(tmp_path):
    writer = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv')
    writer.submit(0, A, {'phase': 'target_train', 'method': 'fastgrpo'})
    writer.submit(0, B, {'phase': 'target_train', 'method': 'fastgrpo'})
    writer.submit(1, C, {'phase': 'target_train', 'method': 'fastgrpo'})
    writer.flush()
    writer.flush()
    rows, csv_rows = read_rows(tmp_path)
    assert [row['step'] for row in rows] == [0, 1]
    assert [row['step'] for row in csv_rows] == ['0', '1']
    assert rows[0]['step_accepted_tokens'] == 31
    assert rows[1]['step_accepted_tokens'] == 10
    for row, csv_row in zip(rows, csv_rows):
        for key in ('step_aal', 'cumulative_aal', 'step_rollout_tokens', 'step_wall_time_s'):
            assert float(csv_row[key]) == row[key]


def test_resume_rewinds_future_rows_and_restores_pending_label(tmp_path):
    writer = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv')
    writer.submit(0, A, {'phase': 'target_train'})
    writer.submit(1, B, {'phase': 'target_train'})
    state = deepcopy(writer.state_dict())
    writer.submit(2, C, {'phase': 'target_train'})
    writer.flush()
    resumed = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv', append=True, state=state)
    resumed.submit(1, B, {'phase': 'target_train'})
    resumed.submit(2, C, {'phase': 'target_train'})
    resumed.flush()
    rows, csv_rows = read_rows(tmp_path)
    assert [row['step'] for row in rows] == [0, 1, 2]
    assert [row['step'] for row in csv_rows] == ['0', '1', '2']
    assert rows[1]['step_aal'] == 2.0
    assert (tmp_path / 'timing.pre_resume.csv').exists()
    assert (tmp_path / 'metrics.pre_resume.jsonl').exists()


def test_resume_can_extend_last_flushed_label_without_duplicate(tmp_path):
    writer = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv')
    writer.submit(0, A, {'phase': 'target_train'})
    writer.flush()
    resumed = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv',
                                append=True, state=deepcopy(writer.state_dict()))
    resumed.submit(0, B, {'phase': 'target_train'})
    resumed.flush()
    rows, csv_rows = read_rows(tmp_path)
    assert len(rows) == len(csv_rows) == 1
    assert rows[0]['step_rollout_tokens'] == B['cumulative_rollout_tokens']


def test_final_filtered_rollouts_and_timers_are_accounted_before_last_flush(tmp_path):
    # Execute the production finalization, without loading model weights or
    # importing the GPU-only training entrypoint. The second rollout is reward
    # filtered, so it contributed draft work/counters but no target update.
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'grpo_speculative.py').read_text())
    aggregate = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_aggregate_job_metrics')
    begin = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'training_end_metrics'
                         for target in node.targets))
    end = next(i for i, node in enumerate(tree.body[begin:], begin)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and ast.unparse(node.value.func) == 'step_metrics.flush')
    writer = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv')
    writer.submit(0, A, {'phase': 'target_train', 'method': 'opd_reflex'})
    timers = PhaseTimings('cpu', target_s=A['cumulative_target_train_time_s'],
                          draft_s=A['cumulative_draft_train_time_s'])
    timers.pending = [('draft', 1.0, None, None)]
    data = dict(generate_time_cost=7.0, train_time_cost=3.0, draft_train_time_cost=2.0,
                total_rollout_tokens=64, total_acc_length=31, total_decoded_token_num=8,
                total_accepted_draft_tokens=12, total_proposed_draft_tokens=30)
    scope = dict(torch=torch, dist=dist,OPD_COUNTER_NAMES=OPD_COUNTER_NAMES,GENERATION_COUNTER_NAMES=GENERATION_COUNTER_NAMES, model=SimpleNamespace(target_model=SimpleNamespace(device='cpu')),
                 args=SimpleNamespace(opd_diagnostics='0', opd_profile='0'),
                 _as_bool=lambda value: value == '1', batch_data=data, phase_timings=timers,
                 step_metrics=writer, completed_step_snapshot=completed_step_snapshot,
                 _cumulative_wall_time=lambda: 18.0)
    exec(compile(ast.Module(body=[aggregate, *tree.body[begin:end + 1]], type_ignores=[]),
                 'grpo_speculative.py', 'exec'), scope)
    rows, csv_rows = read_rows(tmp_path)
    assert len(rows) == len(csv_rows) == 1
    assert rows[0]['step_rollout_tokens'] == 64
    assert rows[0]['step_aal'] == 31 / 8
    assert rows[0]['step_target_train_time_s'] == 3
    assert rows[0]['step_draft_train_time_s'] == 2
    assert timers.pending == []
    assert data['_phase_draft_time_s'] == 2


@pytest.mark.parametrize('method', ['fastgrpo', 'opd_reflex'])
def test_real_training_logging_block_writes_every_unique_step_with_same_schema(tmp_path, method):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'grpo_speculative.py').read_text())
    aggregate = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_aggregate_job_metrics')
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'grpo_iteration == grpo_iteration_num - 1')
    scope = {'torch':torch,'dist':dist,'OPD_COUNTER_NAMES':OPD_COUNTER_NAMES,'GENERATION_COUNTER_NAMES':GENERATION_COUNTER_NAMES}
    exec(compile(ast.Module(body=[aggregate], type_ignores=[]), 'grpo_speculative.py', 'exec'), scope)
    writer = StepMetricsWriter(tmp_path / 'metrics.jsonl', tmp_path / 'timing.csv')
    timers = PhaseTimings('cpu')
    scope.update(
        model=SimpleNamespace(target_model=SimpleNamespace(device='cpu')),
        args=SimpleNamespace(opd_diagnostics='0', opd_profile='0'),
        _as_bool=lambda value: value == '1', step_metrics=writer, phase_timings=timers,
        completed_step_snapshot=completed_step_snapshot, grpo_iteration_num=2,
    )
    code = compile(ast.Module(body=[block], type_ignores=[]), 'grpo_speculative.py', 'exec')
    for label, current in ((0, A), (0, B), (1, C)):
        data = {
            'generate_time_cost': current['cumulative_generation_time_s'],
            'train_time_cost': current['cumulative_target_train_time_s'],
            'draft_train_time_cost': current['cumulative_draft_train_time_s'],
            'total_rollout_tokens': current['cumulative_rollout_tokens'],
            'total_acc_length': current['cumulative_accepted_tokens'],
            'total_decoded_token_num': current['cumulative_verification_rounds'],
            'total_accepted_draft_tokens': current['cumulative_accepted_draft_tokens'],
            'total_proposed_draft_tokens': current['cumulative_proposed_draft_tokens'],
        }
        timers.totals = {'target': data['train_time_cost'], 'draft': data['draft_train_time_cost']}
        scope.update(step=label, batch_data=data,
                     _cumulative_wall_time=lambda: current['cumulative_wall_time_s'],
                     avg_logs={'phase': 'target_train', 'method': method})
        for iteration in range(2):
            scope['grpo_iteration'] = iteration
            exec(code, scope)
    writer.flush()
    rows, csv_rows = read_rows(tmp_path)
    assert len(rows) == len(csv_rows) == 2
    assert [row['step'] for row in rows] == [0, 1]
    assert rows[1]['step_aal'] == 10 / 5
    assert rows[1]['step_rollout_tokens'] == 36
    for row in rows:
        assert row['method'] == method
        assert all(key in row for key in (
            'step_wall_time_s', 'step_generation_time_s', 'step_target_train_time_s',
            'step_draft_train_time_s', 'gpu_allocated_gb', 'gpu_reserved_gb',
            'gpu_peak_allocated_gb', 'gpu_free_gb'))


def test_memory_snapshot_does_not_synchronize_or_flush_allocator(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('memory stats must not synchronize, empty_cache or reset peaks')
    for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name, value in (('memory_allocated', 1024 ** 3), ('memory_reserved', 2 * 1024 ** 3),
                        ('max_memory_allocated', 3 * 1024 ** 3)):
        monkeypatch.setattr(torch.cuda, name, lambda *unused, value=value: value)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *unused: (4 * 1024 ** 3, 8 * 1024 ** 3))
    assert gpu_memory_stats('cuda') == dict(gpu_allocated_gb=1, gpu_reserved_gb=2,
                                          gpu_peak_allocated_gb=3, gpu_free_gb=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA event timing after existing scalar transfer')
def test_phase_timing_reads_events_without_new_synchronization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('telemetry must not add a global CUDA synchronize')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    timers = PhaseTimings('cuda')
    ticket = timers.begin('target')
    work = torch.randn(64, 64, device='cuda').square().sum()
    timers.end(ticket)
    work.cpu()  # Existing phase/log scalar transfer completes the same stream.
    elapsed = timers.resolve()
    assert elapsed['target'] > 0
    assert timers.resolve() == elapsed
