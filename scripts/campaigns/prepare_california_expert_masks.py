"""Derive conservative exact-packing masks from train-only recorded states."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algo.canonical_policy import CanonicalPolicyConfig
from algo.canonical_state import flat_observation_slices
from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.state.action_mask import remaining_payload_can_fit
from EVRoutingEnv.state.features import ACTION_FEATURES, CUSTOMER_FEATURES, GLOBAL_FEATURES, TRUCK_FEATURES
from EVRoutingEnv.state.feasibility import ActionKind


OBSERVATIONS = ACTIONS = MASKS = SHAPE = LAYOUT = None


def process_chunk(bounds):
    start, stop = bounds
    values = OBSERVATIONS[start:stop]
    trucks = values[:, LAYOUT["truck_features"]].reshape(-1, SHAPE.max_trucks, len(TRUCK_FEATURES))
    customers = values[:, LAYOUT["customer_features"]].reshape(-1, SHAPE.max_customers, len(CUSTOMER_FEATURES))
    actions = values[:, LAYOUT["action_features"]].reshape(-1, SHAPE.max_actions, len(ACTION_FEATURES))
    global_values = values[:, LAYOUT["global_features"]]
    result = MASKS[start:stop].copy()
    tf = {name: index for index, name in enumerate(TRUCK_FEATURES)}
    cf = {name: index for index, name in enumerate(CUSTOMER_FEATURES)}
    af = {name: index for index, name in enumerate(ACTION_FEATURES)}
    customer_kind = list(ActionKind).index(ActionKind.CUSTOMER)
    depot_kind = list(ActionKind).index(ActionKind.DEPOT)
    unassigned = list(TaskStatus).index(TaskStatus.UNASSIGNED)
    served = list(TaskStatus).index(TaskStatus.SERVED)
    for row in range(stop - start):
        active = int(global_values[row, GLOBAL_FEATURES.index("active_truck_id")])
        if not 0 <= active < SHAPE.max_trucks or trucks[row, active, tf["is_active"]] != 1:
            raise ValueError("recorded active truck does not match canonical row order")
        depot_slots = np.flatnonzero(actions[row, :, af["kind_code"]] == depot_kind)
        if len(depot_slots) != 1:
            raise ValueError("recorded depot identity is ambiguous")
        depot = int(actions[row, depot_slots[0], af["target_node"]])
        customer_rows = customers[row]
        # Bounds include the original float64 quantities rounded to float32.
        # Upper capacities / lower demands can only retain extra possibilities;
        # they cannot create a spurious infeasibility proof from rounding.
        demand_lower = np.nextafter(customer_rows[:, cf["demand"]], np.float32(-np.inf)).astype(float)
        capacity_upper = np.nextafter(trucks[row, :, tf["remaining_payload"]], np.float32(np.inf)).astype(float)
        statuses = customer_rows[:, cf["status_code"]].astype(int)
        pending = np.flatnonzero(statuses == unassigned)
        tasks = [SimpleNamespace(node_id=int(customer_rows[index, cf["node_id"]]), demand=float(demand_lower[index]), preassigned_to=None if customer_rows[index, cf["preassigned_to"]] < 0 else int(customer_rows[index, cf["preassigned_to"]])) for index in pending]
        by_node = {task.node_id: task for task in tasks}
        others = []
        for index, truck in enumerate(trucks[row]):
            if index == active or truck[tf["failed"]] or truck[tf["is_complete"]] or truck[tf["return_to_depot_pending"]] or int(truck[tf["route_destination"]]) == depot:
                continue
            committed = (statuses != unassigned) & (statuses != served) & (customer_rows[:, cf["claimed_by"]] == index)
            others.append((index, max(0.0, capacity_upper[index] - demand_lower[committed].sum())))
        for slot in np.flatnonzero(result[row]):
            kind = int(actions[row, slot, af["kind_code"]])
            if kind == customer_kind:
                node = int(actions[row, slot, af["target_node"]])
                selected = by_node[node]
                remaining = [task for task in tasks if task.node_id != node]
                payloads = [(active, max(0.0, capacity_upper[active] - selected.demand)), *others]
                if not remaining_payload_can_fit(remaining, payloads, exact=True):
                    result[row, slot] = False
            elif kind == depot_kind and tasks and not remaining_payload_can_fit(tasks, others, exact=True):
                result[row, slot] = False
        if not result[row, ACTIONS[start + row]]:
            raise ValueError(f"packing restriction pruned a successful expert action at transition {start + row}")
    return start, result, int((MASKS[start:stop] & ~result).sum())


def save_array(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    global OBSERVATIONS, ACTIONS, MASKS, SHAPE, LAYOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    root = args.archive_directory.resolve()
    audit = json.loads((root / "dataset_audit.json").read_text())
    if audit["status"] != "completed" or audit["split"] != "train":
        raise ValueError("expert archive has not passed its train-only integrity audit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = CanonicalPolicyConfig.load(args.policy_config)
    SHAPE, LAYOUT = config.shape, flat_observation_slices(config.shape)
    with np.load(root / "alns_planned.npz") as archive:
        OBSERVATIONS, ACTIONS, MASKS = archive["observations"], archive["actions"], archive["masks"]
    if OBSERVATIONS.shape != (audit["transitions"], SHAPE.flat_size):
        raise ValueError("expert and policy schemas differ")
    episode_steps = OBSERVATIONS[:, LAYOUT["global_features"]][:, GLOBAL_FEATURES.index("episode_steps")]
    starts = np.flatnonzero(episode_steps == 0)
    if len(starts) != audit["successful_episodes"] or starts[0] != 0:
        raise ValueError("expert trajectory boundaries do not match episode accounting")
    chunks = [(start, min(start + 1024, len(ACTIONS))) for start in range(0, len(ACTIONS), 1024)]
    remasked = MASKS.copy()
    pruned = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for start, masks, count in pool.map(process_chunk, chunks):
            remasked[start:start + len(masks)] = masks
            pruned += count
    save_array(output / "exact_masks.npy", remasked)
    save_array(output / "episode_starts.npy", starts)
    report = {"status": "completed", "split": "train", "source_archive_sha256": audit["archive_sha256"], "transitions": len(ACTIONS), "successful_episodes": len(starts), "pruned_candidate_rows": pruned, "pruned_expert_actions": 0, "precision_policy": "upper float32 interval capacities and lower interval demands retain uncertainty", "masks_sha256": hashlib.sha256((output / "exact_masks.npy").read_bytes()).hexdigest(), "episode_starts_sha256": hashlib.sha256((output / "episode_starts.npy").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
