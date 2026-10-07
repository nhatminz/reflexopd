"""Storage ownership and full-rollout parity against the former concat history."""
import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from helper.rollout_history import RolloutHistory
from opd_fixtures import TinyModel, load_rollout


class ConcatHistoryReference:
    """Test-only former batched concat / finished-row compaction semantics."""
    def __init__(self, initial, *, repeats=1, **unused):
        self.tensors = {name: tensor.repeat_interleave(repeats, dim=0)
                        for name, tensor in initial.items()}
        self.active = list(range(next(iter(initial.values())).shape[0] * repeats))
        self.lengths = {name: tensor.shape[1] for name, tensor in self.tensors.items()}

    def append(self, original_rows, chunks):
        assert original_rows == self.active
        for name, chunk in chunks.items():
            self.tensors[name] = torch.cat((self.tensors[name], chunk), dim=1)
            self.lengths[name] = self.tensors[name].shape[1]

    def finish(self, original):
        row = self.active.index(original)
        result = {name: tensor[row].clone() for name, tensor in self.tensors.items()}
        keep = [i for i in range(len(self.active)) if i != row]
        self.active.pop(row)
        for name, tensor in self.tensors.items():
            self.tensors[name] = tensor[keep]
        return result


def test_finished_row_has_independent_storage_and_releases_its_buffer():
    initial = torch.randn(3, 4, 6)
    original = weakref.ref(initial)
    history = RolloutHistory({'features': initial}, max_length=20, reserve_tokens=4)
    del initial
    gc.collect()
    assert original() is None
    owner = history.buffers['features'][1]
    owner_ref = weakref.ref(owner)
    expected = owner[:4].clone()
    owned_storage = owner.untyped_storage().data_ptr()
    result = history.finish(1)['features']
    del owner
    gc.collect()
    assert owner_ref() is None
    assert history.buffers['features'][1] is None
    assert result._base is None
    assert result.untyped_storage().data_ptr() != owned_storage
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_append_preserves_data_across_capacity_growth_and_row_finish():
    source = torch.arange(24).reshape(3, 4, 2)
    history = RolloutHistory({'features': source}, repeats=2, max_length=8, reserve_tokens=1)
    expected = [source[i // 2].clone() for i in range(6)]
    active = list(range(6))
    for round_index in range(8):
        chunk = torch.full((len(active), 3, 2), 100 + round_index)
        old_pointers = [history.buffers['features'][i].data_ptr() for i in active]
        old_capacity = history.capacities['features']
        old_length = history.lengths['features']
        history.append(active, {'features': chunk})
        if old_length + 3 <= old_capacity:
            assert old_pointers == [history.buffers['features'][i].data_ptr() for i in active]
        for row, original in enumerate(active):
            expected[original] = torch.cat((expected[original], chunk[row]), dim=0)
        if round_index == 2:
            finished = history.finish(1)['features']
            torch.testing.assert_close(finished, expected[1], rtol=0, atol=0)
            active.remove(1)
    for original in active:
        torch.testing.assert_close(history.finish(original)['features'], expected[original], rtol=0, atol=0)
    assert all(buffer is None for buffer in history.buffers['features'])


@pytest.mark.parametrize('mode', ['fastgrpo', 'opd_reflex'])
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('collect', [False, True])
def test_full_rollout_matches_concat_history(mode, sample, collect, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'Stream', lambda device: object())
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    ids = torch.tensor([[0, 0, 4, 5], [3, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    records = []
    for storage in (ConcatHistoryReference, RolloutHistory):
        torch.manual_seed(411)
        model = TinyModel()
        with torch.inference_mode():
            result = load_rollout(history_type=storage)(
                model, ids, mask, SimpleNamespace(eos_token_id=16),
                do_sample=sample, repeated_generate_nums=8, max_length=22,
                verification_capacity=80, max_verification_num=16, max_draft_k=3,
                max_draft_token_length=3, min_draft_token_length=2,
                method=mode, opd_backend='auto', opd_fast_lr=.05,
                return_all_draft_input=collect)
        records.append((result, model.calls, model.masks))
    old, new = records
    assert old[1] == new[1]
    for a, b in zip(old[2], new[2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ('generated_token_ids', 'response_accepted_length_sum', 'response_verification_rounds',
                'total_acc_length', 'total_decoded_token_num', 'total_accepted_draft_tokens',
                'total_proposed_draft_tokens', 'response_generated_tokens'):
        assert old[0][key] == new[0][key]
    for key in ('all_draft_input_states', 'all_target_hidden_states', 'all_draft_input_ids'):
        if collect:
            assert len(new[0][key]) == 16
            for a, b in zip(old[0][key], new[0][key]):
                assert a.shape == b.shape
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert new[0][key] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocator retention check')
def test_cuda_history_releases_storage_and_has_no_cross_rollout_growth():
    baseline = torch.cuda.memory_allocated()
    allocated = []
    for _ in range(6):
        initial = torch.randn(8, 32, 128, device='cuda', dtype=torch.bfloat16)
        history = RolloutHistory({'features': initial}, max_length=256)
        del initial
        chunk = torch.randn(8, 8, 128, device='cuda', dtype=torch.bfloat16)
        pointers = [row.data_ptr() for row in history.buffers['features']]
        reserved_history_bytes = torch.cuda.memory_allocated()
        for _ in range(12):
            history.append(list(range(8)), {'features': chunk})
            assert torch.cuda.memory_allocated() == reserved_history_bytes
        assert pointers == [row.data_ptr() for row in history.buffers['features']]
        before_finish = torch.cuda.memory_allocated()
        result = history.finish(0)['features']
        assert torch.cuda.memory_allocated() < before_finish
        del result, history, chunk
        gc.collect()
        # Test-only barrier for the allocator assertion; production has none.
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
    assert max(allocated) <= baseline + 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA streamed rollout history parity')
def test_cuda_triton_root_stream_preserves_collected_history(monkeypatch):
    class CudaModel(TinyModel):
        device = torch.device('cuda:0')

        def __init__(self):
            super().__init__()
            self.embedding = self.embedding.cuda()
            self.target_head.cuda()
            self.draft_head.cuda()

    def forbidden_sync(*args, **kwargs):
        pytest.fail('production history/rollout must not call cuda.synchronize')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden_sync)
    ids = torch.tensor([[0, 0, 4, 5], [3, 7, 8, 9]], device='cuda')
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], device='cuda')
    records = []
    for storage in (ConcatHistoryReference, RolloutHistory):
        torch.manual_seed(111)
        model = CudaModel()
        with torch.inference_mode():
            output = load_rollout(device='cuda:0', history_type=storage)(
                model, ids, mask, SimpleNamespace(eos_token_id=16),
                do_sample=True, repeated_generate_nums=8, max_length=22,
                verification_capacity=80, max_verification_num=16, max_draft_k=3,
                max_draft_token_length=3, min_draft_token_length=2,
                method='opd_reflex', opd_backend='triton', opd_fast_lr=.05,
                opd_update_stream=True,
                return_all_draft_input=True)
        records.append((output, model.calls))
    old, new = records
    assert old[1] == new[1]
    assert new[0]['opd_updates']>0
    for key in ('generated_token_ids', 'response_accepted_length_sum', 'response_verification_rounds'):
        assert old[0][key] == new[0][key]
    for key in ('all_draft_input_states', 'all_target_hidden_states', 'all_draft_input_ids'):
        for a, b in zip(old[0][key], new[0][key]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
