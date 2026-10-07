"""Independent probability snapshots oracle for the REAL OPD tree builder."""
import ast
from pathlib import Path
import time

import pytest
import torch

from helper.opd_reflex import OPDReflex, initialize_projector
from helper.tree_verification import pack_tree, select_confidence_nodes

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


class ChangingDraft:
    """Each depth and row has different logits; no transformer/model download."""
    dtype = torch.bfloat16

    def __init__(self, device, vocab=257, hidden=32):
        self.device = torch.device(device)
        self.vocab, self.hidden, self.level, self.calls = vocab, hidden, 0, 0
        self.draft_head = torch.nn.Linear(hidden, vocab, bias=False, device=device)
        self.opd_projector = initialize_projector(hidden, 8).to(device)

    def compute_compact_logits(self, hidden):
        b, c, _ = hidden.shape
        p = torch.zeros(b, c, self.vocab, device=self.device)
        if self.level == 0:
            p[..., :4] = torch.tensor([.6, .25, .1, .05], device=self.device)
        elif self.level == 1:
            p[..., :4] = torch.tensor([.9, .07, .02, .01], device=self.device)
        else:
            p.fill_(1. / self.vocab)
        # Vary batch/context rows as well as depth: scratch changes its B*C
        # interpretation during expansion and previously mixed these rows too.
        offsets = torch.arange(b * c, device=self.device).view(b, c)
        p[..., 0] += .001 * (offsets % 7)
        p /= p.sum(-1, keepdim=True)
        self.level += 1
        return p.log()

    def __call__(self, hidden_states, input_ids, past_key_values, **kwargs):
        self.calls += 1
        b, k = input_ids.shape
        hidden = torch.ones(b, k, self.hidden, device=self.device, dtype=self.dtype)
        key = torch.cat((past_key_values[0][0], hidden[:, None]), -2)
        return dict(hidden_states=hidden, next_feature_states=hidden,
                    past_key_values=[(key, key.clone())])


def confidence_reference(proposals, k):
    # Deliberately independent owned snapshots, never OPD proposal scratch.
    b = proposals[0].shape[0]
    device = proposals[0].device
    beam = proposals[0][:, 0].clone()
    scores = [beam]
    beam_ids = torch.arange(k, device=device).expand(b, -1)
    parents = [torch.full_like(beam_ids, -1)]
    for depth, conditional in enumerate(proposals[1:], 1):
        parents.append(beam_ids.repeat_interleave(k, dim=1))
        candidates = (beam.unsqueeze(-1) * conditional).reshape(b, -1)
        scores.append(candidates)
        indices = candidates.topk(k, dim=-1).indices
        beam_ids = k + k * k * (depth - 1) + indices
        beam = candidates.gather(1, indices)
    return torch.cat(scores, -1), torch.cat(parents, -1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('batch,depth,k', [(3, 3, 7), (64, 3, 7), (3, 5, 8), (2, 2, 3)])
def test_tree_scores_survive_changed_depth_batch_layout_and_reused_proposal_scratch(device, batch, depth, k):
    model = ChangingDraft(device)
    mapping = torch.arange(model.vocab, device=device)
    engine = OPDReflex(8, 16, backend='auto')
    engine.start(model, batch, mapping, model.hidden,
        max_contexts=1 + k * (depth - 1), max_nodes=batch * 32,
        max_path=depth + 1, max_proposal_contexts=k)
    root_storage = engine.tree_root_confidences.data_ptr()
    engine.draft_mask=torch.empty(batch*k*(2+k*depth),device=device,dtype=model.dtype)
    proposals = []
    original_propose = engine.propose
    def capture_propose(*args, **kwargs):
        values, ids, target_ids = original_propose(*args, **kwargs)
        proposals.append(values.clone())
        return values, ids, target_ids
    engine.propose = capture_propose
    captured = {}
    def check_selection(confidences, count, **kwargs):
        expected, parents = confidence_reference(proposals, k)
        # This assertion failed BEFORE the fix, before the CUDA device assert.
        assert torch.equal(confidences, expected), 'retained root confidence overwritten by expansion scratch'
        prefix = torch.cat((torch.ones(batch, 1, device=device), confidences), 1)
        assert (confidences <= prefix.gather(1, parents + 1)).all()
        captured['parents'] = parents
        chosen = select_confidence_nodes(confidences, count, **kwargs)
        oracle = expected.argsort(dim=-1, descending=True, stable=True)[:, :count].sort(-1).values
        assert torch.equal(chosen, oracle)
        return chosen
    def check_pack(parents, *args, **kwargs):
        assert torch.equal(parents, captured['parents'])
        return pack_tree(parents, *args, **kwargs)
    source = ast.parse((Path(__file__).parents[1] / 'helper/specualtive_generate.py').read_text())
    function = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'draft_generate')
    scope = dict(torch=torch, time=time, total_check_time=0., enabled=True, statistical_time=False,
                 opd=engine, compact_to_target=mapping, select_confidence_nodes=check_selection, pack_tree=check_pack)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual-draft-tree', 'exec'), scope)
    key = torch.ones(batch, 1, 2, model.hidden, device=device, dtype=model.dtype)
    hidden = torch.ones(batch, 1, model.hidden, device=device, dtype=model.dtype)
    for _ in range(2):
        model.level = 0
        proposals.clear()
        result = scope['draft_generate'](model, hidden, hidden, [(key, key.clone())], depth,
            torch.ones(batch, device=device, dtype=torch.long), [], draft_k=k, draft_total_token=k)
        assert engine.tree_root_confidences.data_ptr() == root_storage
        assert engine.tree_root_confidences.untyped_storage().data_ptr() != engine.proposal_q.untyped_storage().data_ptr()
        assert torch.equal(engine.tree_root_confidences, proposals[0][:, 0])
        assert torch.equal(engine.q_cache[:batch, 0, :k], proposals[0][:, 0])
        assert result['tensor_tree'].parents.shape == (batch, k + 1)
    assert model.calls == 2 * (depth - 1)  # No extra draft transformer calls.


@pytest.mark.parametrize('device', DEVICES)
def test_proposal_views_are_transient_but_context_caches_are_owned(device):
    model = ChangingDraft(device)
    mapping = torch.arange(model.vocab, device=device)
    state = OPDReflex(8, 16)
    state.start(model, 3, mapping, 32, max_contexts=15, max_nodes=64, max_path=4, max_proposal_contexts=7)
    root_hidden = torch.ones(3, 1, 32, device=device)
    values, ids, target_ids = state.propose(model.compute_compact_logits(root_hidden), root_hidden, 7, mapping, root=True)
    snapshot = values.clone()
    targets = target_ids.clone()
    assert values.untyped_storage().data_ptr() == state.proposal_q.untyped_storage().data_ptr()
    child_hidden = torch.ones(3, 7, 32, device=device)
    state.propose(model.compute_compact_logits(child_hidden), child_hidden, 7, mapping, context_offset=1)
    assert not torch.equal(values, snapshot)  # Makes the lifetime hazard explicit.
    assert torch.equal(state.q_cache[:3, 0, :7], snapshot[:, 0])
    assert torch.equal(target_ids, targets)  # Mapped IDs do not alias scratch.


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('support', [1, 3, 8])
@pytest.mark.parametrize('vocab', [257, 521])
def test_masked_logit_tile_has_zero_mass_not_nan_in_proposal(device, support, vocab):
    model = ChangingDraft(device, vocab=vocab)
    mapping = torch.arange(model.vocab, device=device)
    state = OPDReflex(8, 16)
    state.start(model, 3, mapping, 32, max_contexts=15, max_nodes=64, max_path=4, max_proposal_contexts=7)
    hidden = torch.ones(3, 1, 32, device=device)
    raw = torch.full((3, 1, vocab), -float('inf'), device=device)
    raw[..., :support] = -torch.arange(support, device=device, dtype=torch.float32)
    q, ids, _ = state.propose(raw, hidden, 16, mapping, root=True)
    expected = raw.softmax(-1).gather(-1, ids)
    assert torch.isfinite(q).all()
    torch.testing.assert_close(q, expected, rtol=2e-5, atol=2e-7)
    assert (q[..., support:] == 0).all()
    assert torch.equal(ids, torch.arange(16, device=device).expand(3, 1, -1))
