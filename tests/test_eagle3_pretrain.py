"""CPU regression tests for execution/data plumbing, not a surrogate EAGLE loss."""

import contextlib
import gzip
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from specforge.config import TrainingConfig, load_config, apply_overrides
from specforge.launch import _shard_offline_refs, _distributed_sampler_indices
from specforge.runtime.contracts import TrainBatch
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig, resolve_distributed_mode
from specforge.training.controller import TrainerCore, TrainerController
from specforge.training.length_bucketing import bucketed_indices, load_length_index
from specforge.training.pretrain_attention import validate_attention_backend
from specforge.training.schedule import resolve_total_steps
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput
from specforge.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
BOUNDS = [512, 768, 1024, 1280, 1536, 1792, 2048]


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = torch.nn.Linear(1, 1, bias=False)
        self.frozen = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)

    def forward(self, x):
        return self.draft_model(x)


class ToyStrategy(DraftTrainStrategy):
    name = "eagle3"
    required_features = {"input_ids"}

    def __init__(self, model, seen=None, **kwargs):
        self.model, self.seen = model, seen

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        if self.seen is not None:
            self.seen.extend(batch.sample_ids)
        return StepOutput(
            loss=self.model(batch.tensors["input_ids"].float()).square().mean(),
            metrics={},
        )

    def checkpoint_state_filter(self, state):
        return {key.removeprefix("draft_model."): value for key, value in state.items()
                if key.startswith("draft_model.")}


def optimizer_factory(model):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(model, lr=0.01, total_steps=12, warmup_ratio=0)


@pytest.mark.parametrize("world,mode,expected", [
    (1, "auto", "plain"), (1, "ddp", "plain"), (1, "fsdp", "plain"),
    (2, "auto", "ddp"), (4, "auto", "ddp"), (8, "auto", "ddp"),
    (2, "fsdp", "fsdp"),
])
def test_topology_selection(world, mode, expected):
    assert resolve_distributed_mode(mode, world) == expected


def test_single_gpu_never_wraps_and_optimizer_targets_draft():
    model = ToyModel()
    backend = FSDPTrainingBackend(ParallelConfig(world_size=1), optimizer_factory=optimizer_factory)
    with mock.patch("torch.nn.parallel.DistributedDataParallel", side_effect=AssertionError("DDP used")), \
         mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", side_effect=AssertionError("FSDP used")):
        assert backend.prepare_model(model, wrap=True, optimizer_target=model.draft_model) is model
    assert backend._wrapper_kind == "none"
    assert {id(p) for p in backend.optimizer.model_params} == {id(p) for p in model.draft_model.parameters()}
    assert not backend.optimizer._reduce_grad_norm_across_ranks


@pytest.mark.parametrize("backend", ["fa", "sdpa", "flex_attention"])
def test_attention_override_reaches_draft_constructor(monkeypatch, backend):
    from specforge.algorithms import model_providers

    cfg = load_config(str(ROOT / "third_party/SpecForge/examples/configs/offline/colocated/qwen2.5-7b-eagle3-offline.yaml"))
    cfg = apply_overrides(cfg, [f"training.attention_backend={backend}", "model.load_target_embedding=false", "model.vocab_mapping_path="])
    constructor = mock.Mock()
    fake = constructor.return_value
    fake.to.return_value = fake
    monkeypatch.setitem(sys.modules, "specforge.modeling.auto", SimpleNamespace(
        AutoDraftModel=SimpleNamespace(from_config=constructor)))
    monkeypatch.setattr(model_providers, "_device", lambda: torch.device("cpu"))
    model_providers.build_eagle3_draft(cfg, SimpleNamespace())
    assert constructor.call_args.kwargs["attention_backend"] == backend
    assert cfg.training.ttt_length == 7 and cfg.data.max_length == 2048


def test_unavailable_flashattention_fails_without_fallback(monkeypatch):
    # Only the backend capability seam is substituted; validate never selects
    # SDPA/flex after this concrete capability error.
    fake = SimpleNamespace(
        _std_flash_attn_varlen_func=None, _std_flash_attn_varlen_backward=None,
        _std_flash_unpad_input=None, _std_flash_pad_input=None,
        _raise_standard_flash_attn_unavailable=mock.Mock(side_effect=RuntimeError("bad binary")),
    )
    monkeypatch.setitem(sys.modules, "specforge.modeling.draft", SimpleNamespace(llama3_eagle=fake))
    with pytest.raises(RuntimeError, match="No fallback was selected"):
        validate_attention_backend("fa")
    assert validate_attention_backend("sdpa") == "sdpa"


def test_flashattention_error_names_the_required_api(monkeypatch):
    fake = SimpleNamespace(
        _std_flash_attn_varlen_func=None, _std_flash_attn_varlen_backward=None,
        _std_flash_unpad_input=None, _std_flash_pad_input=None,
        _raise_standard_flash_attn_unavailable=mock.Mock(side_effect=RuntimeError(
            "cannot import name 'flash_attn_varlen_func' from 'flash_attn' (unknown location)")),
    )
    monkeypatch.setitem(sys.modules, "specforge.modeling.draft", SimpleNamespace(llama3_eagle=fake))
    with pytest.raises(RuntimeError) as error:
        validate_attention_backend("fa", probe=True)
    message = str(error.value)
    assert "flash_attn.flash_attn_varlen_func" in message
    assert "_flash_attn_varlen_backward" in message
    assert "PRETRAIN_ATTENTION_BACKEND=sdpa" in message
    assert "unknown location" in message
    assert "No fallback was selected" in message


def test_sdpa_cached_seven_step_forward_backward_without_flashattention(monkeypatch):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from specforge.modeling.draft import llama3_eagle as eagle

    # Exercise the existing TTT attention implementation, not an alternate loss.
    # Disable only helper compilation for this tiny CPU regression (no MSVC/GPU
    # compiler required); the attention, cache and autograd code are unchanged.
    monkeypatch.setattr(eagle, "apply_rotary_pos_emb",
                        eagle.apply_rotary_pos_emb._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle.LlamaRotaryEmbedding, "forward",
                        eagle.LlamaRotaryEmbedding.forward._torchdynamo_orig_callable)
    monkeypatch.setattr(eagle.LlamaRMSNorm, "forward",
                        eagle.LlamaRMSNorm.forward._torchdynamo_orig_callable)
    for name in ("_std_flash_attn_varlen_func", "_std_flash_attn_varlen_backward",
                 "_std_flash_unpad_input", "_std_flash_pad_input"):
        monkeypatch.setattr(eagle, name, None)
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, max_position_embeddings=2048,
                      pretraining_tp=1)
    layer = eagle.LlamaDecoderLayer(cfg, attention_backend="sdpa")
    assert type(layer.self_attn) is eagle.LlamaAttention
    input_emb = torch.randn(2, 8, 16, requires_grad=True)
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    positions = torch.arange(8).unsqueeze(0).expand(2, -1)
    mask = torch.zeros(2, 1, 8, 8).masked_fill(
        torch.ones(8, 8, dtype=torch.bool).triu(1), float("-inf"))
    cache = [[], []]
    for _ in range(7):
        hidden = layer(input_emb, hidden, cache_hidden=cache,
                       attention_mask=mask, position_ids=positions)
    assert len(cache[0]) == len(cache[1]) == 7
    assert torch.isfinite(hidden).all()
    hidden.square().mean().backward()
    assert torch.isfinite(input_emb.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def plan(lengths, rank=0, world=1, epoch=0, batch=4):
    return bucketed_indices(lengths, batch_size=batch, dp_rank=rank, dp_size=world,
                            seed=42, epoch=epoch, boundaries=BOUNDS)


@pytest.mark.parametrize("size,world", [(53, 1), (53, 2), (53, 4), (53, 8), (3, 8), (64, 4)])
def test_bucketing_covers_all_samples_and_balances_ranks(size, world):
    lengths = [128 + (index * 337) % 1920 for index in range(size)]
    shards = [plan(lengths, rank=rank, world=world) for rank in range(world)]
    flattened = [index for shard in shards for index in shard]
    assert set(flattened) == set(range(size))
    assert len({len(shard) for shard in shards}) == 1
    assert len(flattened) == math.ceil(size / world) * world
    if size % world == 0:
        assert len(flattened) == len(set(flattened))
    for rank in range(world):
        assert shards[rank] == plan(lengths, rank=rank, world=world)


def test_bucketing_shuffles_each_epoch_and_reduces_padding():
    lengths = [256] * 64 + [768] * 64 + [1280] * 64 + [2048] * 64
    bucketed = plan(lengths)
    assert bucketed != plan(lengths, epoch=1)
    random = _distributed_sampler_indices(len(lengths), dp_rank=0, dp_size=1, seed=42, epoch=0)

    def padded_tokens(order):
        return sum(max(lengths[index] for index in order[start:start + 4]) * len(order[start:start + 4])
                   for start in range(0, len(order), 4))

    assert padded_tokens(bucketed) < padded_tokens(random)


def test_length_index_cached_and_max_length_does_not_modify_features(tmp_path, monkeypatch):
    paths = []
    for index, size in enumerate([12, 2200]):
        path = tmp_path / f"{index}.ckpt"
        torch.save({"input_ids": torch.arange(size), "hidden_state": torch.ones(1, size, 3)}, path)
        paths.append(str(path))
    cache = str(tmp_path / "lengths.json")
    lengths, identity = load_length_index(paths, cache, 2048)
    assert lengths == [12, 2048]
    assert torch.load(paths[1], weights_only=True)["input_ids"].numel() == 2200
    with mock.patch("specforge.training.length_bucketing._feature_length", side_effect=AssertionError("rescanned")):
        assert load_length_index(paths, cache, 2048) == (lengths, identity)


def test_compressed_length_index(tmp_path):
    path = tmp_path / "sample.ckpt.gz"
    with gzip.open(path, "wb") as stream:
        torch.save({"input_ids": torch.arange(33)}, stream)
    assert load_length_index([str(path)], str(tmp_path / "index.json"), 2048)[0] == [33]


def toy_refs(tmp_path, count=9):
    features = tmp_path / "features"
    features.mkdir(exist_ok=True)
    for index in range(count):
        torch.save({"input_ids": torch.tensor([index + 1])}, features / f"{index:03d}.ckpt")
    return OfflineManifestReader(str(features), run_id="test", feature_keys=("input_ids",)).read()


def collate(raw):
    return {"input_ids": torch.stack([item["input_ids"] for item in raw])}


def test_loader_resume_order_keeps_tail(tmp_path):
    refs = toy_refs(tmp_path)
    ordered = _shard_offline_refs(refs, use_usp_preprocess=False, dp_rank=0, dp_size=1,
                                  seed=42, epoch=3, lengths=[300] * 9,
                                  batch_size=4, boundaries=BOUNDS)
    loader = FeatureDataLoader(LocalFeatureStore("test"), refs=ordered, batch_size=4,
                               collate_fn=collate, drop_last=False, num_workers=2)
    full = [batch.sample_ids for batch in loader]
    assert [len(batch) for batch in full] == [4, 4, 1]
    loader.seek(1)
    assert [batch.sample_ids for batch in loader] == full[1:]
    loader.seek(3)
    assert list(loader) == []


def test_offline_builder_sampler_contract_and_legacy_resume(tmp_path, monkeypatch):
    import specforge.launch as launch
    from specforge.training.checkpoint import CheckpointManager

    refs = toy_refs(tmp_path)
    provider = SimpleNamespace(
        build_reader=lambda *args, **kwargs: SimpleNamespace(read=lambda: refs),
        build_collator=lambda: collate,
        build_normalizer=lambda *args, **kwargs: None,
    )
    algorithm = SimpleNamespace(name="eagle3", providers=SimpleNamespace(
        offline_for=lambda modality: provider,
        step=SimpleNamespace(uses_external_target_head=True),
    ))
    monkeypatch.setattr(launch, "_assemble_trainer", lambda **kwargs: kwargs)
    kwargs = dict(algorithm=algorithm, hidden_states_path=str(tmp_path / "features"),
                  draft_model=ToyModel(), target_head=None, optimizer_factory=optimizer_factory,
                  run_id="test", output_dir=str(tmp_path / "run"), batch_size=4,
                  length_bucketing=True, distributed_mode="auto", seed=42)
    new = launch.build_offline_runtime(**kwargs)
    assert new["drop_last"] is False
    assert new["checkpoint_extra"]["offline_sampler_version"] == 2
    assert new["checkpoint_extra"]["length_bucketing"] is True
    assert new["distributed_mode"] == "auto"
    assert len(new["ref_source"]["refs_for_epoch"](1)) == len(refs)

    # A real historical run must retain v1 data order after a crash, even when
    # the new launcher's defaults request bucketing.
    old_state = {}  # pre-bucketing checkpoints have no sampler metadata at all
    monkeypatch.setattr(CheckpointManager, "read_resume_state", lambda path: old_state)
    old = launch.build_offline_runtime(**kwargs, resume_from="legacy")
    assert old["drop_last"] is True
    assert old["resume_state"] is old_state
    assert old["checkpoint_extra"] == {
        "offline_sampler_version": 1, "sampler_seed": 42, "source_dataset_size": 9,
    }
    assert old["ref_source"]["refs"] == _shard_offline_refs(
        refs, use_usp_preprocess=False, seed=42, epoch=0)
    assert old_state == old["checkpoint_extra"]


def make_trainer(tmp_path, refs, seen, max_steps, resume_from=None, *,
                 drop_last=False, checkpoint_extra=None, resume_state=None,
                 sampler_version=2, num_epochs=1):
    torch.manual_seed(42)
    model = ToyModel()

    def refs_for_epoch(epoch):
        return _shard_offline_refs(refs, use_usp_preprocess=False, dp_rank=0, dp_size=1,
                                  seed=42, epoch=epoch,
                                  lengths=[300] * len(refs) if sampler_version == 2 else None,
                                  batch_size=4, boundaries=BOUNDS)

    return Trainer(
        algorithm_name="eagle3", make_step_strategy=lambda model, **kwargs: ToyStrategy(model, seen),
        controller=DataFlowController("test", metadata_store=NoOpMetadataStore(), enable_sample_queue=False),
        store=LocalFeatureStore("test"), ref_source={"refs": refs_for_epoch(0), "refs_for_epoch": refs_for_epoch},
        model=model, target_head=None, optimizer_factory=optimizer_factory,
        run_id="test", output_dir=str(tmp_path), batch_size=4, accumulation_steps=1,
        num_epochs=num_epochs, max_steps=max_steps, total_steps=12, save_interval=0,
        logger=None, log_interval=50, collate_fn=collate, durable_ack=False,
        resume_from=resume_from, drop_last=drop_last,
        checkpoint_extra=checkpoint_extra, resume_state=resume_state,
    )


def test_real_trainer_checkpoint_resume_matches_uninterrupted_and_tail(tmp_path):
    refs = toy_refs(tmp_path)
    all_seen = []
    full = make_trainer(tmp_path / "full", refs, all_seen, max_steps=3)
    assert full.backend._wrapper_kind == "none"
    assert full._loader.clone_on_fetch is False
    full.fit()
    first_seen = []
    first = make_trainer(tmp_path / "split", refs, first_seen, max_steps=1)
    first.fit()
    from specforge.training.checkpoint import CheckpointManager
    state = CheckpointManager.read_resume_state(str(tmp_path / "split/test-step1"))
    assert state["backend"]["wrapper_kind"] == "none"
    remaining_seen = []
    resumed = make_trainer(tmp_path / "split", refs, remaining_seen, max_steps=3,
                           resume_from=str(tmp_path / "split/test-step1"))
    resumed.fit()
    assert first_seen + remaining_seen == all_seen
    assert len(set(all_seen)) == len(refs) == 9
    assert torch.equal(full.backend.module.draft_model.weight, resumed.backend.module.draft_model.weight)
    assert full.backend.optimizer.scheduler.state_dict() == resumed.backend.optimizer.scheduler.state_dict()
    # A checkpoint at the final short batch resumes with ceil(samples/batch),
    # rather than rejecting or replaying that batch.
    completed_seen = []
    complete = make_trainer(tmp_path / "split", refs, completed_seen, max_steps=3,
                            resume_from=str(tmp_path / "split/test-step3"))
    complete.fit()
    assert completed_seen == []


def test_legacy_sampler_metadata_migration_does_not_break_resume(tmp_path):
    from specforge.training.checkpoint import CheckpointManager

    refs = toy_refs(tmp_path)
    seen_full, seen_first, seen_resume = [], [], []
    full = make_trainer(tmp_path / "full", refs, seen_full, 2,
                        drop_last=True, sampler_version=1)
    full.fit()
    first = make_trainer(tmp_path / "split", refs, seen_first, 1,
                         drop_last=True, sampler_version=1)
    first.fit()
    checkpoint = str(tmp_path / "split/test-step1")
    state = CheckpointManager.read_resume_state(checkpoint)
    assert "offline_sampler_version" not in state
    contract = {"offline_sampler_version": 1, "sampler_seed": 42,
                "source_dataset_size": len(refs)}
    # Same migration performed by build_offline_runtime for historical runs.
    for key, value in contract.items():
        state.setdefault(key, value)
    resumed = make_trainer(tmp_path / "split", refs, seen_resume, 2,
                           resume_from=checkpoint, drop_last=True, sampler_version=1,
                           resume_state=state, checkpoint_extra=contract)
    resumed.fit()
    assert seen_first + seen_resume == seen_full
    assert torch.equal(full.backend.module.draft_model.weight, resumed.backend.module.draft_model.weight)


def test_resume_inside_second_shuffled_epoch(tmp_path):
    refs = toy_refs(tmp_path)
    seen_full, seen_first, seen_resume = [], [], []
    full = make_trainer(tmp_path / "full", refs, seen_full, 6, num_epochs=2)
    full.fit()
    first = make_trainer(tmp_path / "split", refs, seen_first, 4, num_epochs=2)
    first.fit()
    resumed = make_trainer(tmp_path / "split", refs, seen_resume, 6, num_epochs=2,
                           resume_from=str(tmp_path / "split/test-step4"))
    resumed.fit()
    assert seen_full[:9] != seen_full[9:]
    assert seen_first + seen_resume == seen_full
    assert len(seen_full) == 18
    assert torch.equal(full.backend.module.draft_model.weight, resumed.backend.module.draft_model.weight)


@pytest.mark.parametrize("world", [1, 2])
def test_trainer_auto_prepares_plain_or_ddp(tmp_path, world):
    refs = toy_refs(tmp_path)

    class DummyDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module

    with mock.patch.object(ParallelConfig, "from_distributed", return_value=ParallelConfig(world_size=world)), \
         mock.patch("torch.nn.parallel.DistributedDataParallel", side_effect=DummyDDP) as ddp, \
         mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", side_effect=AssertionError("FSDP used")):
        trainer = make_trainer(tmp_path / "run", refs, [], 1)
    assert ddp.call_count == (0 if world == 1 else 1)
    assert trainer.backend._wrapper_kind == ("none" if world == 1 else "ddp")
    composite = trainer.core.strategy.model
    if world > 1:
        composite = composite.module
    assert {id(p) for p in trainer.backend.optimizer.model_params} == {
        id(p) for p in composite.draft_model.parameters()
    }


def test_accumulation_context_covers_forward_and_backward():
    model = ToyModel()
    backend = FSDPTrainingBackend(ParallelConfig(), optimizer_factory=optimizer_factory)
    backend.prepare_model(model, wrap=False, optimizer_target=model.draft_model)
    active = []
    entered = []

    @contextlib.contextmanager
    def microstep_context(*, is_boundary):
        entered.append(is_boundary)
        active.append(True)
        yield
        active.pop()

    backend.microstep_context = microstep_context
    strategy = ToyStrategy(model)
    original = strategy.forward_loss

    def forward(*args, **kwargs):
        assert active
        return original(*args, **kwargs)

    strategy.forward_loss = forward
    model.draft_model.weight.register_hook(lambda gradient: gradient if active else pytest.fail("backward outside context"))
    core = TrainerCore(strategy, backend, accumulation_steps=2)
    batch = TrainBatch(sample_ids=["sample"], strategy="eagle3", tensors={"input_ids": torch.ones(1, 1)})
    assert not core.train_step(batch).optimizer_stepped
    assert core.train_step(batch).optimizer_stepped
    assert entered == [False, True]


def test_no_slow_memory_options_enabled_by_default():
    cfg = TrainingConfig()
    assert cfg.compact_teacher is False
    assert cfg.optimizer_cpu_offload is False
    assert cfg.trim_loss_positions is False
    assert resolve_total_steps(total_steps=None, max_steps=None, num_samples=9,
                               batch_size=4, accumulation_steps=1, num_epochs=1, drop_last=False) == 3


def test_benchmark_report_uses_global_sample_count():
    from scripts.benchmark_pretrain import summarize_benchmark

    report = summarize_benchmark(elapsed_s=10, samples=200, steps=25, gpu_count=2,
                                attention_backend="sdpa", distributed_mode="ddp",
                                warmup_steps=5, effective_batch=8)
    assert report["optimizer_step_time_s"] == 0.4
    assert report["samples_per_second"] == 20


def test_benchmark_refuses_existing_run_or_redirected_output(tmp_path):
    from scripts.benchmark_pretrain import validate_benchmark_output

    fresh = tmp_path / "benchmark"
    validate_benchmark_output(fresh, fresh)
    fresh.mkdir()
    (fresh / "console.log").touch()
    validate_benchmark_output(fresh, fresh)
    with pytest.raises(ValueError, match="redirect"):
        validate_benchmark_output(tmp_path / "production", fresh)
    (fresh / "training_state.pt").touch()
    with pytest.raises(ValueError, match="existing run"):
        validate_benchmark_output(fresh, fresh)


def _ddp_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = ToyModel()
        pc = ParallelConfig(world_size=2, sharding_strategy="NO_SHARD",
                            fsdp_process_group=dist.group.WORLD)
        backend = FSDPTrainingBackend(pc, optimizer_factory=optimizer_factory)
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        assert backend._wrapper_kind == "ddp"
        assert not backend.optimizer._reduce_grad_norm_across_ranks
        calls = []

        def reduce_hook(state, bucket):
            calls.append(True)
            return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(
                lambda future: future.value()[0].div_(2))

        wrapped.register_comm_hook(None, reduce_hook)
        core = TrainerCore(ToyStrategy(wrapped), backend, accumulation_steps=2)
        for micro in range(2):
            batch = TrainBatch(sample_ids=[f"{rank}:{micro}"], strategy="eagle3",
                               tensors={"input_ids": torch.tensor([[rank * 2 + micro + 1.]])})
            core.train_step(batch)
        assert len(calls) == 1  # DDP did not reduce on the first forward/backward
        from specforge.training.checkpoint import CheckpointManager

        controller = TrainerController(core, run_id="ddp", output_dir=output,
                                       start_step=1, start_batch=2, start_samples=2,
                                       total_steps=12)
        controller.save_checkpoint(1)
        restored = CheckpointManager.read_resume_state(str(Path(output) / "ddp-step1"))
        assert restored["backend"]["wrapper_kind"] == "ddp"
        assert "replicated_optimizer_state" in restored
        backend.load_state_dict(restored["backend"])
        torch.save({"weight": model.draft_model.weight.detach(),
                    "optimizer": backend.optimizer.state_dict()}, Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch has no CPU Gloo backend")
def test_two_rank_ddp_accumulation_matches_global_batch(tmp_path):
    import torch.multiprocessing as mp

    try:
        dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    except RuntimeError as exc:
        if "unsupported gloo device" in str(exc):
            pytest.skip(f"This PyTorch build has no usable Gloo transport: {exc}")
        raise
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(42)
    model = ToyModel()
    backend = FSDPTrainingBackend(ParallelConfig(), optimizer_factory=optimizer_factory)
    backend.prepare_model(model, wrap=False, optimizer_target=model.draft_model)
    core = TrainerCore(ToyStrategy(model), backend)
    core.train_step(TrainBatch(sample_ids=["0", "1", "2", "3"], strategy="eagle3",
                              tensors={"input_ids": torch.tensor([[1.], [2.], [3.], [4.]])}))
    for rank in range(2):
        saved = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False)
        torch.testing.assert_close(saved["weight"], model.draft_model.weight, atol=1e-7, rtol=1e-6)
        for expected, actual in zip(backend.optimizer.fp32_params, saved["optimizer"]["fp32_params"]):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
