"""Small Seattle problems, one per component, for the generalist GraphPPO tests."""

from __future__ import annotations

import copy
import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from algo.generalist_observation import GeneralistEnv, ObservationSpec
from EVRoutingEnv.baselines.heuristics.canonical_baselines import RandomFeasiblePolicy
from EVRoutingEnv.utils.utils import load_config


SEATTLE = "EVRoutingEnv/config_files/config_joint_seattle.yaml"
BLOCKS = {
    "static": {},
    "public": {"public_charging": {"enabled": True, "exogenous_traffic": {"utilization": 0.65, "service_preset": "keskin_truck"}}},
    "hos": {"hours_of_service": {"enabled": True, "regime": "us_fmcsa_property"}},
    "same_day": {
        "requests": {"mode": "same_day", "max_requests_per_epoch": 2, "max_requests": 40, "static_context_size": 20},
        "depot_charging": {"enabled": True},
    },
}


def small_config(component: str = "static", *, trucks: int = 2, stops: int = 4) -> dict:
    config = copy.deepcopy(load_config(SEATTLE))
    config["environment"].update(num_trucks=trucks, num_stops=stops)
    blocks = BLOCKS[component]
    if blocks:
        config["environment"]["max_time"] = 24.0 if "requests" in blocks else 72.0
    config.update(copy.deepcopy(blocks))
    return config


def small_env(component: str = "static", *, spec: ObservationSpec | None = None, **sizes) -> GeneralistEnv:
    return GeneralistEnv(small_config(component, **sizes), spec or ObservationSpec(charger_budget=4))


def random_observations(env: GeneralistEnv, seed: int, steps: int) -> list:
    """The first ``steps`` observations of a random feasible rollout."""
    observation, _ = env.reset(seed=seed)
    policy = RandomFeasiblePolicy(seed)
    observations = []
    for _ in range(steps):
        observations.append(observation)
        observation, _, terminated, truncated, _ = env.step(policy(env))
        if terminated or truncated:
            break
    return observations
