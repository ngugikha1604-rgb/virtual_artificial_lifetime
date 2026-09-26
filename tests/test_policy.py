from policy import EpsilonGreedyPolicy


def test_decay_default_rate_uses_epsilon_decay():
    p = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9)
    p.decay()
    assert abs(p.epsilon - 0.9) < 1e-9


def test_decay_clamps_to_epsilon_min():
    p = EpsilonGreedyPolicy(epsilon=0.06, epsilon_min=0.05, epsilon_decay=0.5)
    p.decay()  # 0.06*0.5=0.03, below floor -> clamps up to 0.05
    assert p.epsilon == 0.05


def test_decay_explicit_rate_overrides_without_changing_default():
    """Regression test for the 2026-09 change adding an optional `rate`
    override (used by population.py's INLIFE_EPSILON_DECAY) — a one-off
    override must NOT permanently change self.epsilon_decay."""
    p = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.9)
    p.decay(rate=0.5)
    assert abs(p.epsilon - 0.5) < 1e-9
    p.decay()  # no rate given -> must use the ORIGINAL epsilon_decay (0.9)
    assert abs(p.epsilon - 0.45) < 1e-9


def test_choose_action_always_greedy_when_epsilon_zero():
    p = EpsilonGreedyPolicy(epsilon=0.0)
    q_values = [0.1, 0.9, -0.2, 0.05, 0.3]
    for _ in range(20):
        assert p.choose_action(q_values, 5) == 1
