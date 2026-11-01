"""Replay chosen episodes of the passing policy and of the best-known solution, and save what each truck did.

Run from the repository root:
    PYTHONPATH=. python scripts/analysis/policy_vs_bks_traces.py CASES.json OUT_DIR [RUN]

``CASES.json`` lists ``{"problem", "seed", "label"}``. The policy plays with the
settings of the locked test (plan on, mask at margin 0, retirement guard); the
BKS replays its recorded actions. Each trace is checked against the row it
reproduces; ``row_match`` says whether it did. Replays run on the CPU one episode
at a time, so a near-tied decision can differ from the batched GPU evaluation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/generalist_graphppo"
os.environ.update(
    EVRP_PLAN_MODE="on", EVRP_PLAN_DROPOUT="0", EVRP_SAFETY_MASK="on", EVRP_SAFETY_MARGIN="0",
    EVRP_RETIRE_GUARD="on", EVRP_PLAN_CACHE=str(RESULTS / "bks"),
)

import torch  # noqa: E402

from algo.generalist_observation import GeneralistEnv, ObservationSpec, collate  # noqa: E402
from algo.generalist_policy import GeneralistActorCritic  # noqa: E402
from algo.generalist_ppo import episode_summary  # noqa: E402
from EVRoutingEnv.state.feasibility import joint_action_feasibility  # noqa: E402

CAMPAIGN = RESULTS / "suite_v4"


def _requests(env) -> list[dict]:
    engine = getattr(env, "request_engine", None)
    if engine is None or not getattr(env, "dynamic_requests", False):
        return []
    return [
        {
            "id": int(request.request_id), "kind": request.kind.value, "status": request.status.value,
            "pickup": None if request.pickup is None else int(request.pickup.node),
            "delivery": int(request.delivery.node), "release": float(request.release_time),
        }
        for request in engine.registry.requests()
    ]


def _finish(env, decisions: list[dict]) -> dict:
    summary = episode_summary(env)
    trucks = []
    for truck in env.trucks:
        log = truck.export_event_log()
        trucks.append({
            "id": int(truck.truck_id), "failed": bool(truck.failed), "complete": bool(truck.is_complete),
            "failure_reason": str(truck.failure_reason) if truck.failed else None,
            "final_node": int(truck.current_node), "final_battery": float(truck.current_battery),
            "battery_capacity": float(truck.battery_capacity),
            "timeline": [
                {key: row.get(key) for key in ("activity", "start_time", "end_time", "duration", "start_location", "end_location")}
                for row in log["activity_timeline"]
            ],
        })
    return {"summary": summary, "decisions": decisions, "trucks": trucks, "requests": _requests(env), "clock": float(env.global_clock)}


def _record(env, feasibility, action: int) -> dict:
    truck = env.trucks[int(env.active_truck_id)]
    result = feasibility[action]
    return {
        "truck": int(truck.truck_id), "node": int(truck.current_node), "clock": float(env.global_clock),
        "battery": float(truck.current_battery), "action": int(action),
        "kind": result.action_kind.name, "target": None if result.target_node is None else int(result.target_node),
    }


def replay_policy(policy, env, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    decisions = []
    while not env._episode_ended and env.active_truck_id is not None:
        with torch.no_grad():
            position, _, _ = policy.act(collate([observation], "cpu"), True)
        action = int(observation.action_index[int(position[0])])
        decisions.append(_record(env, env.last_action_feasibility, action))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return _finish(env, decisions)


def replay_actions(env, seed: int, actions: list[int]) -> dict:
    env.reset(seed=seed)
    decisions = []
    for action in actions:
        if env._episode_ended or env.active_truck_id is None:
            break
        decisions.append(_record(env, joint_action_feasibility(env), int(action)))
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return _finish(env, decisions)


def main() -> None:
    cases = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    run = sys.argv[3] if len(sys.argv) > 3 else "full_dagger_plan_charge_s2"
    torch.set_num_threads(2)
    policy = GeneralistActorCritic.load(CAMPAIGN / "training" / run, prefix="best", map_location="cpu").eval()
    rows = {(row["problem"], row["seed"]): row for row in json.loads((CAMPAIGN / f"evaluation/test/final/{run}.json").read_text())}
    environments: dict = {}
    for case in cases:
        problem, seed = case["problem"], int(case["seed"])
        target = out / f"{problem}_{seed}_{case['label']}.json"
        if target.exists():
            continue
        config = str(CAMPAIGN / "configs" / f"{problem}.yaml")
        if problem not in environments:
            environments[problem] = (
                GeneralistEnv(config, policy.config.observation_spec, problem=problem),
                GeneralistEnv(config, ObservationSpec(charger_budget=64)),
            )
        played, plain = environments[problem]
        best = json.loads((RESULTS / "bks" / problem / "test.json").read_text())[str(seed)]["best"]
        method_rows = (RESULTS / "bks" / problem / "runs" / f"{best['method']}__test.jsonl").read_text().splitlines()
        actions = next(row["actions"] for row in map(json.loads, method_rows) if row["seed"] == seed)
        trace = {
            "case": case, "run": run, "bks_method": best["method"],
            "policy": replay_policy(policy, played, seed), "bks": replay_actions(plain, seed, actions),
            "bks_row": {key: best[key] for key in ("hours", "served", "success", "failed")},
        }
        row = rows[(problem, seed)]
        trace["policy_row"] = {key: row[key] for key in ("hours", "served", "success", "failed", "termination_reason")}
        trace["policy_row_match"] = abs(trace["policy"]["summary"]["hours"] - row["hours"]) < 1e-3 and trace["policy"]["summary"]["served"] == row["served"]
        trace["bks_row_match"] = abs(trace["bks"]["summary"]["hours"] - best["hours"]) < 1e-3 and trace["bks"]["summary"]["served"] == best["served"]
        target.write_text(json.dumps(trace))
        print(problem, seed, case["label"], "policy", round(trace["policy"]["summary"]["hours"], 1), trace["policy"]["summary"]["termination_reason"],
              "match", trace["policy_row_match"], "| bks", round(trace["bks"]["summary"]["hours"], 1), "match", trace["bks_row_match"], flush=True)


if __name__ == "__main__":
    main()
