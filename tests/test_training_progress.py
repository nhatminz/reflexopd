"""Progress-only checks; no model download, CUDA or full training run."""

import ast
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerController


def controller(**kwargs):
    result = SimpleNamespace(
        optimizer_stepped=True,
        materialize_metrics=mock.Mock(return_value={"loss": 1.25}),
    )
    core = SimpleNamespace(
        strategy=SimpleNamespace(name="eagle3", trainable_module=lambda: mock.Mock()),
        backend=SimpleNamespace(optimizer=SimpleNamespace(get_learning_rate=lambda: 0.001)),
        accumulation_remainder=0,
        train_step=mock.Mock(return_value=result),
    )
    return TrainerController(core, run_id="test", **kwargs)


def test_pretrain_progress_is_visible_through_non_tty(monkeypatch):
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    run = controller(total_steps=4, start_step=2)
    bar = run._make_progress_bar()
    assert not output.isatty()
    assert bar is not None and not bar.disable
    assert bar.n == 2 and bar.total == 4
    bar.update(1)
    bar.close()
    assert "EAGLE3 pretrain" in output.getvalue()
    assert "3/4" in output.getvalue()


@pytest.mark.parametrize("value,disabled", [("1", True), ("true", True), ("0", False), ("false", False)])
def test_progress_disable_switch(monkeypatch, value, disabled):
    monkeypatch.setenv("TQDM_DISABLE", value)
    with mock.patch("tqdm.tqdm") as factory:
        bar = controller(total_steps=4)._make_progress_bar()
    assert (bar is None) == disabled
    assert factory.call_count == (0 if disabled else 1)


def test_pretrain_progress_only_on_rank_zero(monkeypatch):
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    with mock.patch("torch.distributed.is_initialized", return_value=True), \
         mock.patch("torch.distributed.get_rank", return_value=1), \
         mock.patch("tqdm.tqdm") as factory:
        assert controller(total_steps=4)._make_progress_bar() is None
    factory.assert_not_called()


def test_progress_counts_optimizer_steps_and_reuses_logged_metrics(monkeypatch):
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    run = controller(max_steps=6, total_steps=12, start_step=4,
                     logger=mock.Mock(), log_interval=1)
    accumulated = SimpleNamespace(optimizer_stepped=False)
    stepped = run.core.train_step.return_value
    run.core.train_step.side_effect = [accumulated, stepped, accumulated, stepped]
    batches = [TrainBatch(sample_ids=[str(i)], strategy="eagle3", tensors={}) for i in range(4)]
    with mock.patch("tqdm.tqdm") as factory:
        assert run.fit(batches) == 6
    assert factory.call_args.kwargs["initial"] == 4
    assert factory.call_args.kwargs["total"] == 6
    bar = factory.return_value
    assert bar.update.call_args_list == [mock.call(1), mock.call(1)]
    assert stepped.materialize_metrics.call_count == run.logger.call_count == 2
    postfix = bar.set_postfix.call_args.args[0]
    assert postfix["epoch"] == "1/1" and postfix["batch"] == 4
    assert postfix["loss"] == "1.2500" and postfix["lr"] == "1.00e-03"
    bar.close.assert_called_once()


def test_progress_closes_on_training_failure(monkeypatch):
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    run = controller(total_steps=4)
    run._fit = mock.Mock(side_effect=RuntimeError("synthetic failure"))
    with mock.patch("tqdm.tqdm") as factory:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            run.fit([])
    factory.return_value.close.assert_called_once()


@pytest.mark.parametrize("filename", ["grpo_speculative.py", "train_draft.py"])
def test_legacy_train_progress_configuration_without_importing_training(filename, monkeypatch):
    # Execute only the display assignments, not either GPU entrypoint.
    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root / filename).read_text(encoding="utf-8"))
    assignments = {}
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"progress_disabled", "epoch_bar", "batch_bar"}:
                assignments[target.id] = node
    code = compile(ast.Module(body=[assignments[key] for key in (
        "progress_disabled", "epoch_bar", "batch_bar")], type_ignores=[]), filename, "exec")
    for disabled_env, rank, expected_disabled in [("0", "0", False), ("1", "0", True), ("0", "1", True)]:
        monkeypatch.setenv("TQDM_DISABLE", disabled_env)
        monkeypatch.setenv("RANK", rank)
        factory = mock.Mock()
        exec(code, {"os": os, "tqdm": factory, "start_epoch": 2, "num_epochs": 5,
                    "dataloader": [1, 2], "epoch": 2, "is_main_process": rank == "0"})
        epoch, batch = factory.call_args_list
        assert epoch.kwargs["initial"] == 2 and epoch.kwargs["total"] == 5
        assert epoch.kwargs["unit"] == "epoch" and epoch.kwargs["position"] == 0
        assert batch.kwargs["unit"] == "batch" and batch.kwargs["position"] == 1
        for call in (epoch, batch):
            assert call.kwargs["disable"] is expected_disabled
            assert call.kwargs["mininterval"] == 1.0
