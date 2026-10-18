"""Public heuristic entrypoint backed by the canonical fleet dispatcher."""

import numpy as np

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    GreedyHeuristicPolicy,
    HeuristicParameters,
)


class HeuristicPolicy:
    """Greedy routing and charging under the environment's shared hard mask."""

    def __init__(self, verbose=False, buffer_frac=0.05, min_target_soc=0.6, multi_delivery_depth=2):
        self.verbose = verbose
        self.buffer_frac = max(0.0, float(buffer_frac))
        self.min_target_soc = min(1.0, max(1e-9, float(min_target_soc)))
        self.multi_delivery_depth = max(1, int(multi_delivery_depth))
        self.decision_history = []
        self.navigator = GreedyHeuristicPolicy(HeuristicParameters(
            energy_safety_factor=1.0 + self.buffer_frac,
            target_soc=self.min_target_soc,
        ))

    def get_action(self, env):
        action, _ = self.get_action_with_explanations(env)
        return action

    def get_action_with_explanations(self, env):
        if isinstance(env, np.ndarray) or not hasattr(env, "active_truck_id"):
            raise TypeError("HeuristicPolicy expects the environment object, not an observation")
        if env.active_truck_id is None:
            raise RuntimeError("no active truck is available for a heuristic decision")
        action = self.navigator(env)
        explanation = env._action_to_string(action)
        self.log_decision(env.active_truck_id, action, explanation)
        if self.verbose:
            print(explanation)
        return action, explanation

    def log_decision(self, truck_id, action, explanation):
        self.decision_history.append({
            "truck_id": truck_id,
            "action": action,
            "explanation": explanation,
        })
