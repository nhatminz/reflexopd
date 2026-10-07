import torch

from helper.sampling import (
    build_sampling_probs,
    sample_from_probs,
    sample_target_from_logits,
)


def test_sampling_distribution_applies_temperature_top_p_and_top_k_once():
    logits = torch.tensor([[[3.0, 2.0, 1.0, -1.0]]])
    actual = build_sampling_probs(logits, temperature=2.0, top_p=0.8, top_k=2)
    base = torch.softmax(logits.float() / 2.0, dim=-1)
    # top-p=0.8 retains the first two sorted tokens; top-k=2 retains the same
    # support, followed by a single normalization.
    expected = torch.zeros_like(base)
    expected[..., :2] = base[..., :2]
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, 1))


def test_sample_from_probs_uses_the_given_distribution_without_logits():
    probs = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])
    samples = sample_from_probs(probs)
    assert samples.tolist() == [[1], [0]]


def test_sampler_returns_existing_probs_and_preserves_rng_consumption():
    logits=torch.randn(3,4,17)
    torch.manual_seed(18)
    p=build_sampling_probs(logits,1.,.95,5,0)
    expected=sample_from_probs(p);rng=torch.random.get_rng_state()
    torch.manual_seed(18)
    actual,reused=sample_target_from_logits(logits,do_sample=True,temperature=1.,top_p=.95,top_k=5,eos_token_id=0)
    assert torch.equal(actual,expected) and torch.equal(torch.random.get_rng_state(),rng)
    assert torch.equal(p,reused)


def test_greedy_sampler_does_not_construct_teacher_softmax():
    tokens,probs=sample_target_from_logits(torch.tensor([[[0.,3.,1.,2.]]]),
        do_sample=False,temperature=1.,top_p=.95,top_k=None,eos_token_id=0)
    assert tokens.tolist()==[[1]] and probs is None
