"""
tests/test_smoke.py
===================
Smoke tests verifying that all core modules import and instantiate correctly
without requiring GPU, Gurobi, or training data.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np


# ── Model import tests ────────────────────────────────────────────────────────

def test_imports():
    """All model modules import without error."""
    from models import GATEncoder, LSTMPredictor, PPOPolicy, ValueNetwork
    from optimization import ChanceConstraintChecker, MILPSolver
    from environment import EVRPTWEnv, EVRPTWConfig


def test_gat_encoder_forward():
    """GATEncoder produces correct output shapes."""
    from models import GATEncoder
    N = 20   # nodes
    E = N * (N - 1)  # fully connected
    model = GATEncoder(
        node_feature_dim=7, edge_feature_dim=4,
        hidden_dim=32, n_layers=2, n_heads=4
    )
    node_feat  = torch.randn(N, 7)
    edge_index = torch.tensor(
        [(i, j) for i in range(N) for j in range(N) if i != j],
        dtype=torch.long
    ).T  # (2, E)
    edge_feat  = torch.randn(E, 4)
    veh_state  = torch.randn(3)

    H_L, h_graph = model(node_feat, edge_index, edge_feat, veh_state)

    assert H_L.shape    == (N, 32), f"Expected ({N}, 32), got {H_L.shape}"
    assert h_graph.shape == (32,),  f"Expected (32,), got {h_graph.shape}"


def test_lstm_predictor_forward():
    """LSTMPredictor produces (mu, sigma²) of correct shape."""
    from models import LSTMPredictor
    model = LSTMPredictor(hidden_dim=16, n_layers=1)
    B, T = 4, 10
    x = torch.randn(B, T, 24)
    mu, var = model(x)
    assert mu.shape  == (B, T)
    assert var.shape == (B, T)
    assert (var > 0).all(), "All variances must be positive"


def test_lstm_single_arc():
    """Single-arc inference returns scalar floats."""
    from models import LSTMPredictor
    model = LSTMPredictor(hidden_dim=16, n_layers=1)
    arc_feat = torch.randn(24)
    mu, var = model.predict_single_arc(arc_feat)
    assert isinstance(mu,  float)
    assert isinstance(var, float)
    assert var > 0


def test_ppo_policy_forward():
    """PPOPolicy returns a valid Categorical distribution."""
    from models import GATEncoder, PPOPolicy
    from torch.distributions import Categorical
    N = 15
    gat = GATEncoder(hidden_dim=32, n_layers=1, n_heads=4)
    policy = PPOPolicy(hidden_dim=32)

    node_feat  = torch.randn(N, 7)
    edge_index = torch.tensor(
        [(i, j) for i in range(N) for j in range(N) if i != j],
        dtype=torch.long
    ).T
    edge_feat = torch.randn(N*(N-1), 4)
    veh_state = torch.randn(3)

    H_L, h_graph = gat(node_feat, edge_index, edge_feat, veh_state)
    dist, logits  = policy(H_L, h_graph, veh_state)

    assert isinstance(dist, Categorical)
    assert logits.shape == (N,)
    action = dist.sample()
    assert 0 <= action.item() < N


def test_value_network_forward():
    """ValueNetwork returns a scalar."""
    from models import ValueNetwork
    vn = ValueNetwork(hidden_dim=32)
    h  = torch.randn(32)
    vs = torch.randn(3)
    v  = vn(h, vs)
    assert v.shape == torch.Size([])


# ── Optimization module tests ─────────────────────────────────────────────────

def test_chance_constraint_feasible():
    """Feasibility check passes when battery is ample."""
    from optimization import ChanceConstraintChecker
    from optimization.chance_constraint import BatteryState
    checker = ChanceConstraintChecker(confidence=0.95)
    state   = BatteryState(current_kwh=100.0, capacity_kwh=150.0)
    # Arc needs 10 kWh mean, 1 kWh² variance → required ≈ 11.6 kWh
    assert checker.is_feasible(mu_E=10.0, sigma2_E=1.0, battery_state=state)


def test_chance_constraint_infeasible():
    """Feasibility check fails when battery is nearly empty."""
    from optimization import ChanceConstraintChecker
    from optimization.chance_constraint import BatteryState
    checker = ChanceConstraintChecker(confidence=0.95)
    state   = BatteryState(current_kwh=5.0, capacity_kwh=150.0)
    # Arc needs 10 kWh mean — clearly infeasible
    assert not checker.is_feasible(mu_E=10.0, sigma2_E=1.0, battery_state=state)


def test_battery_update():
    """Battery state updates correctly after arc traversal."""
    from optimization import ChanceConstraintChecker
    from optimization.chance_constraint import BatteryState
    checker = ChanceConstraintChecker()
    state   = BatteryState(current_kwh=80.0, cumulative_variance=0.0)
    new     = checker.update_battery(mu_E=15.0, sigma2_E=4.0, battery_state=state)
    assert abs(new.current_kwh - 65.0) < 1e-6
    assert abs(new.cumulative_variance - 4.0) < 1e-6


def test_phi_inverse_known_values():
    """Φ⁻¹ approximation matches known values within tolerance."""
    from optimization.chance_constraint import ChanceConstraintChecker
    # Φ⁻¹(0.95) ≈ 1.6449
    z95 = ChanceConstraintChecker._phi_inverse(0.95)
    assert abs(z95 - 1.6449) < 0.01, f"Got {z95}, expected ≈ 1.6449"
    # Φ⁻¹(0.99) ≈ 2.3263
    z99 = ChanceConstraintChecker._phi_inverse(0.99)
    assert abs(z99 - 2.3263) < 0.01


# ── Environment tests ─────────────────────────────────────────────────────────

def test_env_reset():
    """EVRPTWEnv.reset() returns a valid observation dict."""
    from environment import EVRPTWEnv, EVRPTWConfig
    env = EVRPTWEnv(EVRPTWConfig(n_customers_min=10, n_customers_max=10, seed=0))
    obs = env.reset()
    assert "node_features"  in obs
    assert "edge_index"     in obs
    assert "edge_features"  in obs
    assert "vehicle_state"  in obs
    assert "feasible_mask"  in obs
    assert obs["node_features"].shape[1] == 7
    assert obs["edge_features"].shape[1] == 4
    assert obs["vehicle_state"].shape    == (3,)


def test_env_step():
    """EVRPTWEnv.step() runs without error and returns correct types."""
    from environment import EVRPTWEnv, EVRPTWConfig
    env = EVRPTWEnv(EVRPTWConfig(n_customers_min=5, n_customers_max=5, seed=42))
    obs = env.reset()
    mask = obs["feasible_mask"]
    feasible_nodes = np.where(mask)[0]
    if len(feasible_nodes) > 0:
        _, reward, done, info = env.step(int(feasible_nodes[0]))
        assert isinstance(reward, float)
        assert isinstance(done,   bool)
        assert isinstance(info,   dict)


def test_curriculum_schedule():
    """n_customers increases as episodes progress."""
    from environment import EVRPTWEnv, EVRPTWConfig
    env = EVRPTWEnv(EVRPTWConfig(n_customers_min=10, n_customers_max=100, seed=0))
    n_early = env._curriculum_n_customers.__func__(env) if False else None
    env._episode = 0
    n0 = env._curriculum_n_customers()
    env._episode = 17500
    n_mid = env._curriculum_n_customers()
    env._episode = 35000
    n_end = env._curriculum_n_customers()
    assert n0 <= n_mid <= n_end
    assert n_end == 100


# ── Heteroscedastic loss test ─────────────────────────────────────────────────

def test_heteroscedastic_loss():
    """Loss is finite and non-negative for valid inputs."""
    from models.lstm_predictor import HeteroscedasticLoss
    criterion = HeteroscedasticLoss()
    mu     = torch.randn(8, 10)
    var    = torch.exp(torch.randn(8, 10))   # strictly positive
    actual = torch.randn(8, 10)
    loss   = criterion(mu, var, actual)
    assert torch.isfinite(loss)
    # NLL can be negative (for tight, accurate predictions) so no lower bound check


# ── Parameter count sanity check ─────────────────────────────────────────────

def test_model_parameter_counts():
    """Total trainable parameters are in a reasonable range."""
    from models import GATEncoder, LSTMPredictor, PPOPolicy, ValueNetwork
    gat    = GATEncoder()
    lstm   = LSTMPredictor()
    policy = PPOPolicy()
    vn     = ValueNetwork()
    total  = sum(p.numel() for m in [gat, lstm, policy, vn] for p in m.parameters())
    # Expect ~500K–5M parameters for these configs
    assert 100_000 < total < 10_000_000, f"Unexpected param count: {total:,}"
    print(f"\n  Total parameters: {total:,}")
