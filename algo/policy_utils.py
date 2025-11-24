"""
Shared helpers for loading trained policies from disk for evaluation scripts.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from algo.PPO_VariableActionGNN import PPOVariableActionGNN
from algo.PPO_actionGNN import PPOActionGNN
from truck_env.baselines.heuristic_policy import HeuristicPolicy
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.optimization import HAS_GUROBI, GurobiOptimalPolicy

# Importing only for type checking / documentation purposes.
from truck_env.state.gnn_state_space import GNNStateSpace  # noqa: F401


def _normalize_policy_type(label: Optional[str]) -> Optional[str]:
    """Map different user-provided labels to canonical policy identifiers."""
    if label is None:
        return None
    cleaned = str(label).strip().lower()
    mapping = {
        "variable-ppo": "ppo-variable",
        "variable_ppo": "ppo-variable",
        "ppo-variable": "ppo-variable",
        "ppo_variable": "ppo-variable",
        "variable": "ppo-variable",
        "var-ppo": "ppo-variable",
        "varppo": "ppo-variable",
        "ppo-var": "ppo-variable",
        "ppo": "ppo",
        "standard-ppo": "ppo",
        "ppo-standard": "ppo",
        "ppo_standard": "ppo",
        "standard": "ppo",
        "heuristic": "heuristic",
        "rule-based": "heuristic",
        "rulebased": "heuristic",
        "optimal": "optimal",
        "gurobi": "optimal",
    }
    return mapping.get(cleaned, cleaned)


def _build_node_feature_dims(
    env: EventDrivenTruckEnv,
    gnn_state_space: "GNNStateSpace",
) -> Dict[str, int]:
    """Infer node feature dimensions from a temporary environment instance."""
    gnn_state = gnn_state_space.get_state_GNN(env)
    dims: Dict[str, int] = {}
    for node_type in gnn_state.node_types:
        features = gnn_state[node_type].x
        feat_dim = int(features.numel()) if features.dim() == 1 else int(features.shape[-1])
        dims[node_type] = feat_dim
    return dims


def load_policy(
    policy_path: str,
    requested_algo: Optional[str],
    gnn_state_space: "GNNStateSpace",
    config: dict,
    device: str = "cpu",
) -> Tuple[object, str]:
    """
    Load a trained policy along with the resolved policy type.

    Automatically infers the correct network architecture from the checkpoint's
    metadata, so mismatched user input (e.g., requesting PPO while pointing to a
    variable-action checkpoint) does not cause hard failures.
    """
    normalized_requested = _normalize_policy_type(requested_algo)
    if isinstance(policy_path, str):
        lowered_path = policy_path.lower()
        if lowered_path == "heuristic":
            normalized_requested = "heuristic"
        elif lowered_path == "optimal":
            normalized_requested = "optimal"

    if normalized_requested == "heuristic":
        return HeuristicPolicy(), "heuristic"

    if normalized_requested == "optimal":
        if not HAS_GUROBI:
            raise RuntimeError(
                "The optimal policy requires gurobipy, but it is not available."
            )
        policy = GurobiOptimalPolicy(config=config, seed=0)
        return policy, "optimal"

    config_file = os.path.join(policy_path, "ppo_network_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Could not find network config at {config_file}")

    with open(config_file, "r") as f:
        net_config = json.load(f)

    saved_algo = _normalize_policy_type(net_config.get("algo"))
    resolved_algo = saved_algo or normalized_requested or "ppo"
    if normalized_requested and resolved_algo != normalized_requested:
        print(
            f"[policy_utils] Requested '{normalized_requested}' but checkpoint is '{resolved_algo}'. "
            "Using checkpoint definition."
        )

    env_temp = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    try:
        env_temp.reset(seed=0)
        node_feature_dims = _build_node_feature_dims(env_temp, gnn_state_space)
    finally:
        env_temp.close()

    policy_kwargs = {
        "action_dim": int(net_config["action_dim"]),
        "node_feature_dims": node_feature_dims,
        "hidden_dim": net_config["hidden_dim"],
        "num_layers": net_config["num_layers"],
        "mlp_dim": net_config["mlp_dim"],
        "lr": net_config["lr"],
        "gamma": net_config["gamma"],
        "gae_lambda": net_config["gae_lambda"],
        "clip_coef": net_config["clip_coef"],
        "value_coef": net_config["value_coef"],
        "entropy_coef": net_config["entropy_coef"],
        "max_grad_norm": net_config["max_grad_norm"],
        "ppo_epochs": net_config["ppo_epochs"],
        "minibatch_size": net_config["minibatch_size"],
        "device": device,
    }

    if resolved_algo == "ppo-variable" and "charge_durations" in net_config:
        policy_kwargs["charge_durations"] = net_config["charge_durations"]

    PolicyCls = PPOVariableActionGNN if resolved_algo == "ppo-variable" else PPOActionGNN
    policy = PolicyCls(**policy_kwargs)

    model_path = os.path.join(policy_path, "ppo_model_best")
    actor_path = f"{model_path}_actor.pt"
    if os.path.exists(actor_path):
        policy.load(model_path)
        print(f"Loaded model from {actor_path}")
    else:
        print(f"Warning: Model file {actor_path} not found. Using random policy.")

    return policy, resolved_algo
