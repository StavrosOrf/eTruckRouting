"""Query a nominal expert on states visited by a frozen student/teacher mixture.

Each training case has one initial ALNS plan shared across all mixture arms.
Labels are expert queries; behavior actions are recorded separately. Only
successful, contiguous episodes whose labels obey the policy mask are retained.
No model parameters are updated during collection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch

from algo.canonical_policy import CanonicalActorCritic
from EVRoutingEnv.baselines.heuristics.canonical_baselines import decode_feasible_actions
from EVRoutingEnv.baselines.optimization.exact_optimization import PlanSolution
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.feasibility import ActionKind
from EVRoutingEnv.utils.utils import load_config
from scripts.training.collect_demonstrations import build_demonstrator


_ENV = None
_STUDENT = None


def arm_name(probability):
    return f"teacher_{round(100 * probability):03d}"


def recovery_action(env, navigator, mask):
    """Explicit recovery expert restricted to the learner's allowed actions."""
    candidates = [action for action in decode_feasible_actions(env) if mask[action.index]]
    if not candidates:
        return None
    by_kind = {}
    for action in candidates:
        by_kind.setdefault(action.kind, []).append(action)
    truck = env.trucks[env.active_truck_id]
    goal = navigator._select_goal(
        env, truck, by_kind.get(ActionKind.CUSTOMER, []), by_kind.get(ActionKind.DEPOT, [])
    )
    return int(navigator.navigate_toward(env, truck, goal, candidates, by_kind))


def worker(task):
    global _ENV, _STUDENT
    config_path, checkpoint, prefix, probabilities, seeds, directory, initial_plans, recover_labels = task
    torch.set_num_threads(1)
    if _ENV is None:
        config = load_config(config_path)
        _ENV = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        _STUDENT = CanonicalActorCritic.load(checkpoint, prefix=prefix).eval()
    env = _ENV
    result = {arm_name(probability): {"episodes": [], "arrays": []} for probability in probabilities}
    for seed in seeds:
        env.reset(seed=seed)
        anchor = deepcopy(env.scenario_descriptor)
        if initial_plans:
            saved = json.loads((Path(initial_plans) / f"{seed}.json").read_text())
            evidence = saved["diagnostics"]
            plan = PlanSolution(
                routes={int(key): list(value) for key, value in saved["routes"].items()},
                status=evidence["status"], objective_hours=evidence["objective_hours"],
                best_bound_hours=evidence["best_bound_hours"], wall_seconds=evidence["solver_wall_seconds"],
            )
        else:
            initial = build_demonstrator("alns_planned")
            plan = deepcopy(initial._ensure_plan(env))
            evidence = initial.diagnostics()
        (Path(directory) / "plans" / f"{seed}.json").write_text(json.dumps({
            "scenario_seed": seed, "routes": plan.routes,
            "diagnostics": evidence,
        }, indent=2) + "\n")
        for probability in probabilities:
            observation, info = env.reset(seed=seed)
            if env.scenario_descriptor != anchor:
                raise ValueError("paired collection physical scenario changed")
            # Restore the same initial nominal plan for every mixture arm. The
            # planner prices depot-start routes, rather than moving-truck routes.
            teacher = build_demonstrator("alns_planned")
            teacher._plan = deepcopy(plan)
            teacher.last_plan = teacher._plan
            teacher._scenario_key = (id(env), env.scenario_seed)
            teacher._searches = 1
            rng = np.random.default_rng(seed + 310_000_000)
            records = {key: [] for key in ("observations", "actions", "masks", "behavior_actions", "student_actions", "teacher_selected", "nominal_teacher_actions", "recovery_used")}
            terminated = truncated = False
            rejected = None
            disagreements = student_deviations = 0
            nominal_rejections = recovery_queries = 0
            while not (terminated or truncated) and len(records["actions"]) < 10000:
                mask = policy_action_mask(env)
                nominal = int(teacher(env, observation, info))
                label = nominal
                recovery_used = False
                if not 0 <= label < len(mask) or not mask[label]:
                    nominal_rejections += 1
                    if recover_labels:
                        recovered = recovery_action(env, teacher.navigator, mask)
                        if recovered is not None:
                            label = recovered
                            recovery_used = True
                            recovery_queries += 1
                if not 0 <= label < len(mask) or not mask[label]:
                    rejected = {"step": len(records["actions"]), "teacher_action": label}
                    break
                with torch.no_grad():
                    action, _, _ = _STUDENT.act(
                        torch.as_tensor(observation).unsqueeze(0),
                        torch.as_tensor(mask).unsqueeze(0), deterministic=True,
                    )
                proposed = int(action.item())
                use_teacher = bool(rng.random() < probability)
                behavior = label if use_teacher else proposed
                disagreements += int(proposed != label)
                student_deviations += int(not use_teacher and proposed != label)
                records["observations"].append(np.asarray(observation, dtype=np.float32).copy())
                records["actions"].append(label)
                records["masks"].append(mask.copy())
                records["behavior_actions"].append(behavior)
                records["student_actions"].append(proposed)
                records["teacher_selected"].append(use_teacher)
                records["nominal_teacher_actions"].append(nominal)
                records["recovery_used"].append(recovery_used)
                observation, _, terminated, truncated, info = env.step(behavior)
            metrics = extract_operational_metrics(env).as_dict()
            kept = bool(metrics["success"] and rejected is None)
            arm = result[arm_name(probability)]
            arm["episodes"].append({
                "scenario_seed": seed, "teacher_probability": probability,
                "success": bool(metrics["success"]), "retained": kept,
                "decisions": len(records["actions"]),
                "teacher_student_disagreements": disagreements,
                "executed_student_disagreements": student_deviations,
                "teacher_label_rejection": rejected,
                "nominal_label_rejections": nominal_rejections,
                "recovery_expert_queries": recovery_queries,
                "operating_hours": metrics["total_operating_time"],
                "termination_reason": metrics["termination_reason"],
            })
            if kept:
                arrays = {
                    "observations": np.stack(records["observations"]),
                    "actions": np.asarray(records["actions"], dtype=np.int64),
                    "masks": np.stack(records["masks"]),
                    "behavior_actions": np.asarray(records["behavior_actions"], dtype=np.int64),
                    "student_actions": np.asarray(records["student_actions"], dtype=np.int64),
                    "teacher_selected": np.asarray(records["teacher_selected"], dtype=bool),
                    "nominal_teacher_actions": np.asarray(records["nominal_teacher_actions"], dtype=np.int64),
                    "recovery_used": np.asarray(records["recovery_used"], dtype=bool),
                    "scenario_seeds": np.full(len(records["actions"]), seed, dtype=np.int64),
                }
                if not np.isfinite(arrays["observations"]).all() or not arrays["masks"][np.arange(len(arrays["actions"])), arrays["actions"]].all():
                    raise ValueError("retained recovery labels are invalid")
                arm["arrays"].append(arrays)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--student-prefix", default="pretrained")
    parser.add_argument("--teacher-probabilities", nargs="+", type=float, default=[1.0, 0.8, 0.5])
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--seed-offset", type=int, default=700000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-plans", type=Path)
    parser.add_argument("--recover-rejected-labels", action="store_true")
    args = parser.parse_args()
    if not 20000 <= args.seed_offset < args.seed_offset + args.scenarios < 1_000_000_000:
        raise ValueError("collection must use separate train-only seeds")
    if any(not 0 <= value <= 1 for value in args.teacher_probabilities):
        raise ValueError("teacher probabilities must lie in [0,1]")
    if len(set(map(arm_name, args.teacher_probabilities))) != len(args.teacher_probabilities):
        raise ValueError("teacher probability arm names collide")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(exist_ok=True)
    if (root / "summary.json").exists():
        raise FileExistsError("inspect existing recovery collection")
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"evrp_recovery_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["EVRP_SCENARIO_CACHE"] = str(scratch / "scenarios")
    os.environ["MPLCONFIGDIR"] = str(scratch / "matplotlib")
    torch.set_num_threads(1)
    seeds = list(range(args.seed_offset, args.seed_offset + args.scenarios))
    tasks = [(args.config, args.student_checkpoint, args.student_prefix, args.teacher_probabilities, seeds[index:index + 4], str(root), str(args.initial_plans.resolve()) if args.initial_plans else None, args.recover_rejected_labels) for index in range(0, len(seeds), 4)]
    gathered = {arm_name(value): {"episodes": [], "arrays": []} for value in args.teacher_probabilities}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for index, result in enumerate(pool.map(worker, tasks), 1):
            for name, values in result.items():
                gathered[name]["episodes"].extend(values["episodes"])
                gathered[name]["arrays"].extend(values["arrays"])
            print(f"completed {index}/{len(tasks)} paired batches", flush=True)
    summaries = {}
    for name, values in gathered.items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        blocks = values.pop("arrays")
        if not blocks:
            summaries[name] = {"episodes": len(values["episodes"]), "retained_episodes": 0}
            (folder / "metadata.json").write_text(json.dumps(values, indent=2) + "\n")
            continue
        arrays = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        archive = folder / "recovery.npz"
        np.savez_compressed(archive, **arrays)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        retained = [row for row in values["episodes"] if row["retained"]]
        summary = {
            "episodes": len(values["episodes"]), "retained_episodes": len(retained),
            "transitions": len(arrays["actions"]),
            "teacher_label_rejections": sum(row["teacher_label_rejection"] is not None for row in values["episodes"]),
            "executed_student_disagreements_retained": sum(row["executed_student_disagreements"] for row in retained),
            "nominal_label_rejections": sum(row["nominal_label_rejections"] for row in values["episodes"]),
            "recovery_expert_queries_retained": sum(row["recovery_expert_queries"] for row in retained),
            "mean_successful_operating_hours": float(np.mean([row["operating_hours"] for row in retained])),
            "archive_sha256": digest,
        }
        summaries[name] = summary
        metadata = {**summary, "episode_records": values["episodes"], "split": "train", "scenario_seeds": seeds,
                    "operating_hours_role": "mixed collection behavior, not a deployed student-policy result",
                    "label_source": "nominal expert queried on mixed-behavior states",
                    "behavior_actions_recorded_separately": True,
                    "teacher_definition": "initial ALNS plan plus mask-restricted energy-safe recovery" if args.recover_rejected_labels else "initial ALNS plan",
                    "student_checkpoint": args.student_checkpoint, "student_prefix": args.student_prefix}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps({"status": "completed", "optimizer_updates": 0,
        "role": "train-only recovery demonstrations; mixed behavior is not a deployed policy",
        "scenarios": len(seeds), "summaries": summaries}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
