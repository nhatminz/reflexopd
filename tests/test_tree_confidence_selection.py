"""Regression: FP32 path-confidence ties must never orphan a selected node."""
from types import SimpleNamespace

import pytest
import torch

from helper.tree_verification import pack_tree, select_confidence_nodes, trace_verified_path
from opd_fixtures import CountModel, load_rollout

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def backend(device):
    if device == 'cpu':
        return None
    from helper import tree_kernels
    return tree_kernels


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['random', 'ones', 'zero', 'signed_zero', 'ulps', 'underflow'])
@pytest.mark.parametrize('count', [1, 7, 31, 105])
def test_exact_confidence_selection_matches_stable_fp32_reference(device, kind, count):
    torch.manual_seed(739)
    scores = torch.rand(4, 105, device=device)
    if kind == 'ones':
        scores.fill_(1.)
    elif kind in ('zero', 'signed_zero'):
        scores.zero_()
        if kind == 'signed_zero':
            scores[:, ::2] = -0.
    elif kind == 'ulps':
        scores.fill_(.5)
        scores[:, ::3] = torch.nextafter(scores[:, ::3], torch.ones_like(scores[:, ::3]))
        scores[:, 1::3] = torch.nextafter(scores[:, 1::3], torch.zeros_like(scores[:, 1::3]))
    elif kind == 'underflow':
        scores = torch.cumprod(torch.full_like(scores, .01), -1)
    before = scores.clone()
    # A view with nontrivial stride also exercises the fused encoder's addressing.
    holder = torch.empty(4, 210, device=device)
    holder[:, ::2] = scores
    view = holder[:, ::2]
    workspace = torch.empty(4 * 105, device=device, dtype=torch.long)
    actual = select_confidence_nodes(view, count, kernels=backend(device), workspace=workspace)
    expected = scores.argsort(dim=-1, descending=True, stable=True)[:, :count].sort(-1).values
    assert torch.equal(actual, expected)
    assert torch.equal(view, before)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('probability', [1., .01, 0.])
def test_selected_tied_tree_is_parent_closed_mask_and_verifier_work(device, probability):
    # Breadth-first binary tree, FP32 child<=parent; unit and underflow ties.
    nodes = 126
    ids = torch.arange(nodes, device=device)
    parents = ((ids - 2) // 2).clamp_min(-1).expand(3, -1).clone()
    probabilities = torch.full((3, nodes), probability, device=device)
    scores = torch.empty_like(probabilities)
    for j in range(nodes):
        p = int((j - 2) // 2)
        scores[:, j] = probabilities[:, j] if p < 0 else scores[:, p] * probabilities[:, j]
    contexts = ids.expand(3, -1).clone()
    tokens = (ids + 1).expand(3, -1).clone()
    for count in (1, 7, 15, 63, nodes):
        chosen = select_confidence_nodes(scores, count, kernels=backend(device))
        selected = torch.zeros(3, nodes, device=device, dtype=torch.bool).scatter_(1, chosen, True)
        original_parents = parents.gather(1, chosen)
        assert ((original_parents < 0) | selected.gather(1, original_parents.clamp_min(0))).all()
        tree = pack_tree(parents, contexts, chosen, tokens, max_depth=6)
        assert ((tree.parents[:, 1:] >= 0) &
                (tree.parents[:, 1:] < torch.arange(1, count + 1, device=device))).all()
        mask = tree.attention_mask(4, torch.bfloat16, kernels=backend(device))
        reference_mask = tree.attention_mask(4, torch.bfloat16)
        assert torch.equal(mask, reference_mask)
        samples = torch.ones_like(tree.parents)
        path = trace_verified_path(tree, samples, 1000, kernels=backend(device))
        reference = trace_verified_path(tree, samples, 1000)
        for field in ('tokens', 'packed_indices', 'feedback_contexts', 'lengths'):
            assert torch.equal(getattr(path, field), getattr(reference, field))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('streamed', [False, True])
def test_saturated_draft_rollout_previously_orphaned_nodes_now_runs(device, streamed):
    if device == 'cpu' and streamed:
        pytest.skip('CUDA side-stream exercised on GPU')
    torch.manual_seed(739)
    model = CountModel(device)
    head = torch.nn.Linear(8, 17, bias=True).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-1000.)
        head.bias[0] = 1000.
    model.draft_head = head
    generate = load_rollout(device)
    captured = []
    original_pack = generate._test_scope['pack_tree']
    def checked_pack(*args, **kwargs):
        tree = original_pack(*args, **kwargs)
        captured.append(tree.parents.clone())
        return tree
    generate._test_scope['pack_tree'] = checked_pack
    result = generate(model, torch.tensor([[4, 5], [8, 9]], device=device),
        torch.ones(2, 2, device=device, dtype=torch.long), SimpleNamespace(eos_token_id=16),
        method='opd_reflex', do_sample=True, repeated_generate_nums=8, max_length=20,
        verification_capacity=128, max_verification_num=32, max_draft_k=7,
        max_draft_token_length=5, min_draft_token_length=3, opd_fast_lr=0.,
        opd_update_stream=streamed, return_all_draft_input=True)
    assert len(captured) > 1
    for parents in captured:
        assert ((parents[:, 1:] >= 0) &
                (parents[:, 1:] < torch.arange(1, parents.shape[1], device=device))).all()
    assert len(result['generated_token_ids']) == 16
    assert model.calls == result['verification_batches'] + 1
    assert result['total_decoded_token_num'] == sum(result['response_verification_rounds'])
    assert len(result['all_draft_input_states']) == 16


def test_invalid_tree_is_not_silently_accepted_on_cpu():
    parents = torch.tensor([[-1, 0]])
    with pytest.raises(RuntimeError, match='not parent-closed'):
        pack_tree(parents, parents.clone(), torch.tensor([[1]]), parents.clone(), 2)


@pytest.mark.parametrize('device', DEVICES)
def test_unique_confidences_preserve_original_native_selection_and_do_not_use_rng(device):
    torch.manual_seed(739)
    scores = torch.stack([torch.randperm(264, device=device).float() / 512 for _ in range(4)])
    random_state = torch.cuda.get_rng_state() if device != 'cpu' else torch.random.get_rng_state()
    workspace = torch.empty(4 * 264, device=device, dtype=torch.long)
    for count in (1, 7, 31, 128, 264):
        original = torch.topk(scores, k=count, dim=-1).indices.sort(-1).values
        actual = select_confidence_nodes(scores, count, kernels=backend(device), workspace=workspace)
        assert torch.equal(actual, original)
    after = torch.cuda.get_rng_state() if device != 'cpu' else torch.random.get_rng_state()
    assert torch.equal(random_state, after)
