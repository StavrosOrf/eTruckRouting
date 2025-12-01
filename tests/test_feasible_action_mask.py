"""Integration test to ensure the GNN feasible_action_mask is never empty when a decision is required."""

import unittest

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.baselines.heuristic_policy import HeuristicPolicy
from truck_env.utils.utils import load_config


class FeasibleActionMaskTests(unittest.TestCase):
    def setUp(self):
        config = load_config("truck_env/config_files/config.yaml")
        # Keep runs short for CI and debugging
        config["environment"]["max_time"] = 50.0
        config["environment"]["verbose"] = False
        self.env = EventDrivenTruckEnv(
            config=config, verbose=False, enable_plotting=False
        )
        self.gnn = GNNStateSpace(
            num_trucks=self.env.num_trucks,
            num_stops=self.env.num_stops,
            max_time=self.env.max_time,
            num_charging_nodes=self.env.num_charging_nodes,
        )
        self.policy = HeuristicPolicy(verbose=False)

    def tearDown(self):
        if hasattr(self, "env"):
            self.env.close()

    def test_feasible_actions_never_empty(self):
        obs, info = self.env.reset(seed=0)
        for step in range(100):
            state = self.gnn.get_state_GNN(self.env)
            mask = state.feasible_action_mask
            self.assertTrue(
                bool(mask.any()),
                f"Empty feasible_action_mask at step {step}, node={self.env.trucks[self.env.active_truck_id].current_node if self.env.active_truck_id is not None else 'NA'}, state={self.env.truck_states.get(self.env.active_truck_id)}",
            )
            action = self.policy.get_action(self.env)
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                break

    def test_full_battery_at_charger_still_has_actions(self):
        # Construct state where truck is at a charger with full battery and hasn't charged yet
        obs, info = self.env.reset(seed=1)
        truck = self.env.trucks[0]
        charger_node = self.env.charging_nodes[0]
        truck.current_node = charger_node
        truck.current_battery = truck.battery_capacity  # full
        truck.has_charged_this_stop = False
        truck.is_charging = False
        self.env.truck_states[truck.truck_id] = "ready"
        self.env.active_truck_id = truck.truck_id

        state = self.gnn.get_state_GNN(self.env)
        mask = state.feasible_action_mask
        self.assertTrue(
            bool(mask.any()),
            f"Empty feasible_action_mask when full battery at charger node {charger_node}",
        )

    def test_no_empty_masks_with_many_trucks_many_scenarios(self):
        """Stress check with 10 trucks, 3 stops over 100 seeds."""
        config = load_config("truck_env/config_files/config.yaml")
        config["environment"]["num_trucks"] = 10
        config["environment"]["num_stops"] = 3
        config["environment"]["verbose"] = False
        config["environment"]["max_time"] = 50.0
        env = EventDrivenTruckEnv(
            config=config, verbose=False, enable_plotting=False
        )
        gnn = GNNStateSpace(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=env.num_charging_nodes,
        )
        policy = HeuristicPolicy(verbose=False)

        try:
            for seed in range(100):
                env.reset(seed=seed)
                for step in range(60):  # cap steps to keep runtime bounded
                    state = gnn.get_state_GNN(env)
                    mask = state.feasible_action_mask
                    self.assertTrue(
                        bool(mask.any()),
                        f"Empty feasible_action_mask at seed={seed}, step={step}, "
                        f"truck={env.active_truck_id}, node="
                        f"{env.trucks[env.active_truck_id].current_node if env.active_truck_id is not None else 'NA'}, "
                        f"state={env.truck_states.get(env.active_truck_id)}",
                    )
                    action = policy.get_action(env)
                    _, _, done, truncated, _ = env.step(action)
                    if done or truncated:
                        break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
