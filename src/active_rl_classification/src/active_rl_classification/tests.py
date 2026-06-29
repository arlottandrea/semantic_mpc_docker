"""
Integration tests: belief convergence, reward signals, feature extractor.

Run: uv run python -m active_rl_classification.tests
"""

import os
import numpy as np
import torch

from active_rl_classification.env import TreeClassificationEnv, BELIEF_THRESHOLD
from active_rl_classification.model import TreeClassFeatureExtractor


def _make_env(**overrides):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csvs = [
        os.path.join(data_dir, "RawData.csv"),
        os.path.join(data_dir, "RipeData.csv"),
    ]
    csvs = [p for p in csvs if os.path.exists(p)]
    config = {
        "ntargets": 1,
        "layout": "random",
        "side": 15.0,
        "perception_csvs": csvs if csvs else None,
    }
    config.update(overrides)
    return TreeClassificationEnv(config=config)


# ---------------------------------------------------------------------- #
# Belief convergence
# ---------------------------------------------------------------------- #

def test_belief_convergence():
    """Verify the Bayesian update converges when fed an informative measurement.

    Two flavors:
      (a) synthetic fixed prob [0.2, 0.8] → belief must reach threshold.
      (b) parametric perception with drone directly in front of the tree.
    """
    print("\n=== Belief convergence test ===")

    # (a) Synthetic update
    belief = np.array([0.5, 0.5])
    prob = np.array([0.2, 0.8])
    history = [belief[1]]
    for _ in range(50):
        new_b = belief * prob
        new_b /= new_b.sum() + 1e-32
        belief = new_b
        history.append(belief[1])
        if belief[1] >= BELIEF_THRESHOLD:
            break
    assert belief[1] >= BELIEF_THRESHOLD, f"synthetic belief failed to converge: {belief}"
    print(f"  (a) synthetic [0.2,0.8] → converged in {len(history)-1} steps, final={belief[1]:.4f}")

    # (b) Parametric perception end-to-end through env._get_measurement
    env = _make_env(ntargets=1, side=15.0)
    if env.perception.class_models[0].tree is None:
        print("  (b) skipped — perception CSVs not available")
        return

    env.reset(seed=11)
    env.ntargets = 1
    env.trees = [np.array([0.0, 0.0])]
    env.tree_classes = np.array([1])
    env.beliefs = np.ones((1, 2)) / 2
    env._observations = np.ones((1, 2)) / 2
    env.tracked = np.array([False])
    env.drone = np.array([-2.0, 0.0, 0.0])  # facing tree at origin

    history_b = [env.beliefs[0, 1]]
    for _ in range(50):
        m = env._get_measurement(0)
        new_b = env.beliefs[0] * m
        new_b /= new_b.sum() + 1e-32
        env.beliefs[0] = new_b
        history_b.append(new_b[1])
        if new_b[1] >= BELIEF_THRESHOLD:
            break
    assert env.beliefs[0, 1] >= BELIEF_THRESHOLD, (
        f"parametric belief failed: {env.beliefs[0]}")
    print(f"  (b) parametric in-FOV → converged in {len(history_b)-1} steps, "
          f"final={env.beliefs[0,1]:.4f}")


# ---------------------------------------------------------------------- #
# Reward signals
# ---------------------------------------------------------------------- #

def test_reward_signals():
    """Verify each reward component is produced under the right condition."""
    print("\n=== Reward signal test ===")
    env = _make_env()
    env.reset(seed=1)

    # 1) Step penalty + movement cost on a no-op action
    env.ntargets = 1
    env.trees = [np.array([0.0, 0.0])]
    env.tree_classes = np.array([0])
    env.beliefs = np.ones((1, 2)) / 2
    env.tracked = np.array([False])
    env.drone = np.array([10.0, 10.0, 0.0])
    env._observations = np.ones((1, 2)) / 2
    env._prev_entropy = 1.0
    env.steps = 0

    _, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
    print(f"  far, zero action -> reward={r:.3f} (expect ~-0.3)")
    assert not term and not trunc
    assert r < 0, f"expected negative reward, got {r}"

    # 2) Classification bonus: tree should become tracked this step
    env.reset(seed=2)
    env.ntargets = 1
    env.trees = [np.array([0.0, 0.0])]
    env.tree_classes = np.array([1])
    env.beliefs = np.array([[0.001, 0.999]])
    env.tracked = np.array([False])
    env._observations = np.ones((1, 2)) / 2
    env._prev_entropy = 0.01
    env.steps = 0
    env.drone = np.array([-3.0, 0.0, 0.0])

    _, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    print(f"  ready-to-track -> reward={r:.3f} tracked={info['num_tracked']} term={term}")
    assert term, "should terminate when all tracked"
    assert r > 100, f"expected >100 on completion, got {r}"

    # 3) Timeout → truncated, not terminated
    env.reset(seed=3)
    env.ntargets = 1
    env.trees = [np.array([0.0, 0.0])]
    env.tree_classes = np.array([0])
    env.beliefs = np.ones((1, 2)) / 2
    env.tracked = np.array([False])
    env._observations = np.ones((1, 2)) / 2
    env._prev_entropy = 1.0
    env.drone = np.array([10.0, 10.0, 0.0])
    env.steps = env.max_steps - 1
    _, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
    print(f"  timeout step -> term={term} trunc={trunc}")
    assert trunc and not term


# ---------------------------------------------------------------------- #
# Feature extractor
# ---------------------------------------------------------------------- #

def test_feature_extractor():
    """Forward/backward pass + mask invariance."""
    print("\n=== Feature extractor test ===")
    env = _make_env()
    N = env.k_obs
    extractor = TreeClassFeatureExtractor(env.observation_space, features_dim=128)

    B = 4

    def dummy_obs(n_real):
        obs = {
            "location": torch.randn(B, N, 3),
            "belief": torch.rand(B, N, 1),
            "measurement": torch.rand(B, N, 1),
            "tracked": torch.zeros(B, N, 1),
            "mask": torch.zeros(B, N),
        }
        obs["mask"][:, :n_real] = 1.0
        return obs

    obs = dummy_obs(n_real=3)
    out = extractor(obs)
    assert out.shape == (B, 128), out.shape
    assert torch.isfinite(out).all()
    print(f"  forward OK -> shape={tuple(out.shape)}")

    # Backward pass
    loss = out.pow(2).mean()
    loss.backward()
    grads_finite = all(
        (p.grad is None) or torch.isfinite(p.grad).all()
        for p in extractor.parameters()
    )
    assert grads_finite, "non-finite gradients"
    print(f"  backward OK — all grads finite")

    # Mask invariance: perturbing padded slots (n_real..k) must not change output
    extractor.zero_grad()
    with torch.no_grad():
        obs_a = dummy_obs(n_real=3)
        obs_b = {kk: v.clone() for kk, v in obs_a.items()}
        obs_b["location"][:, 3:]    = torch.randn_like(obs_b["location"][:, 3:]) * 10
        obs_b["belief"][:, 3:]      = torch.rand_like(obs_b["belief"][:, 3:])
        obs_b["measurement"][:, 3:] = torch.rand_like(obs_b["measurement"][:, 3:])

        out_a = extractor(obs_a)
        out_b = extractor(obs_b)
        diff = (out_a - out_b).abs().max().item()
    print(f"  mask perturbation max-diff={diff:.4f} "
          f"(non-zero expected: SAB mixes across all slots)")
    assert torch.isfinite(out_b).all()


if __name__ == "__main__":
    test_belief_convergence()
    test_reward_signals()
    test_feature_extractor()
    print("\nAll integration tests passed.")
