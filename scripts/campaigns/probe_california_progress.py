"""Check whether observed charging cycles arise from missing feasible progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.features import ACTION_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    config = yaml.safe_load((root / "configs/california_10t30d.yaml").read_text())
    config["environment"]["completion_shield"] = True
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=1_000_000_003)
        graph = env.transport_graph
        factor = 1.2
        safe_nodes = [env.joint_instance.depot_node, *env.charging_nodes]
        energies = []
        for customer in (71, 72):
            escape = min(graph.get_path_energy(customer, node) for node in safe_nodes)
            entries = [
                {
                    "station": node,
                    "leg_kwh": graph.get_path_energy(node, customer) * factor,
                    "leg_plus_escape_kwh": (
                        graph.get_path_energy(node, customer) + escape
                    )
                    * factor,
                }
                for node in env.charging_nodes
            ]
            entries.sort(key=lambda row: row["leg_plus_escape_kwh"])
            energies.append(
                {
                    "customer": customer,
                    "escape_nominal_kwh": escape,
                    "depot_leg_plus_escape_kwh": factor
                    * (
                        graph.get_path_energy(env.joint_instance.depot_node, customer)
                        + escape
                    ),
                    "best_station_options": entries[:5],
                }
            )
        witness = list(env.feasibility_witness)
        removed = []
        for step, (truck, action) in enumerate(witness):
            if env.active_truck_id != truck:
                raise RuntimeError("certificate event order does not match replay")
            mask = policy_action_mask(env)
            if not mask[action]:
                row = env.get_canonical_features().action_features[action]
                removed.append(
                    {
                        "step": step,
                        "truck": truck,
                        "action": action,
                        "target": int(row[ACTION_FEATURES.index("target_node")]),
                        "battery": env.trucks[truck].current_battery,
                    }
                )
            _, _, terminated, truncated, info = env.step(action)
        result = {
            "energy_geometry": energies,
            "shield_rejected_witness_actions": removed,
            "witness_replay_successful": bool(info["successful"]),
            "terminated": terminated,
            "truncated": truncated,
        }
        path = root / "diagnostics/progress_geometry.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
