"""Check that a case-study config generates many distinct, certified, solvable scenarios.

    python scripts/casestudies/check_scenarios.py EVRoutingEnv/config_files/config_joint_texas.yaml \
        --episodes 20 --plot-episodes 2

For each seed the environment draws a depot and customers from the case's nodes, certifies the
instance is solvable (it raises otherwise), and a heuristic policy plays the episode. Writes
``scenario_check.json`` (and ``figures/scenarios.png`` for generated case studies) next to the data.
Loading a large case computes all-pairs routes once (cached in $TMPDIR): run it under SLURM.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

import numpy as np  # noqa: E402

from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy  # noqa: E402
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv  # noqa: E402
from EVRoutingEnv.utils.utils import load_config, resolve_network_data_dir  # noqa: E402


def original(env, idx):
    return int(env.transport_graph.graph.nodes[int(idx)]["original_id"])


def draw(env, seed):
    env.reset(seed=seed)
    inst = env.joint_instance
    return {
        "seed": seed,
        "depot": original(env, inst.depot_node),
        "customers": sorted(original(env, t.node_id) for t in inst.tasks),
        "demand_total": float(sum(t.demand for t in inst.tasks)),
    }


def play(env, policy, max_steps=600, seed=None):
    """Play the scenario the last reset drew (pass ``seed`` to reset first)."""
    if seed is not None:
        env.reset(seed=seed)
    steps, done, info = 0, False, {}
    while not done and steps < max_steps:
        _, _, terminated, truncated, info = env.step(policy.get_action(env))
        steps += 1
        done = terminated or truncated
    om = dict(info.get("operational_metrics") or {})
    om["steps"] = steps
    om["finished"] = bool(done)
    return om


def summarise(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"mean": round(float(np.mean(v)), 3), "median": round(float(np.median(v)), 3),
            "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--plot-episodes", type=int, default=0, help="also write the environment's episode figures for the first N seeds")
    ap.add_argument("--max-steps", type=int, default=600)
    a = ap.parse_args()

    config = load_config(a.config)
    data_dir = Path(resolve_network_data_dir(config["network"]))
    in_case_folder = "case_studies" in str(data_dir)
    name = data_dir.name if in_case_folder else "california_original_folder"
    t0 = time.time()
    env = EventDrivenTruckEnv(a.config, verbose=False, enable_plotting=False)
    load_s = time.time() - t0
    g = env.transport_graph
    print(f"[{name}] environment ready in {load_s:.0f}s: {g.num_nodes} nodes, {len(g.get_charging_nodes())} charger nodes, "
          f"battery {config['truck']['battery_capacity']} kWh, trucks {env.num_trucks}, stops {env.fixed_num_stops}", flush=True)

    roles = {}
    if (data_dir / "vis_data" / "nodes.json").exists():
        roles = {n["node"]: n for n in json.load(open(data_dir / "vis_data" / "nodes.json"))}

    # 1. determinism: the same seed must replay the same scenario
    first, again = draw(env, a.seed), draw(env, a.seed)
    deterministic = first == again

    # 2. many scenarios: draw, certify, play
    policy = HeuristicPolicy(verbose=False)
    rows, draws, failures = [], [], []
    for k in range(a.episodes):
        seed = a.seed + k
        t1 = time.time()
        try:
            d = draw(env, seed)
            t_draw = time.time() - t1
            m = play(env, policy, a.max_steps)
        except Exception as e:  # noqa: BLE001 - record why a scenario could not be built or played
            failures.append({"seed": seed, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"  seed {seed}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        d["seconds"] = round(time.time() - t1, 1)
        d["seconds_to_generate"] = round(t_draw, 1)
        d["result"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        draws.append(d); rows.append(m)
        print(f"  seed {seed}: depot={roles.get(d['depot'], {}).get('depot_code') or roles.get(d['depot'], {}).get('zone_id') or d['depot']}"
              f" success={m['success']} steps={m['steps']} distance={m['total_distance']:.0f}km "
              f"charges={m['charging_sessions']} queue={m['total_queue_time']:.2f}h makespan={m['fleet_makespan']}", flush=True)

    ok = [r for r in rows if r["success"]]
    all_customers = [c for d in draws for c in d["customers"]]
    summary = {
        "case": name, "config": a.config, "env_load_seconds": round(load_s),
        "nodes": g.num_nodes, "charger_nodes": len(g.get_charging_nodes()),
        "battery_kwh": config["truck"]["battery_capacity"], "trucks": env.num_trucks, "stops": env.fixed_num_stops,
        "episodes_requested": a.episodes, "episodes_played": len(rows), "scenario_failures": failures,
        "same_seed_replays_same_scenario": deterministic,
        "success_rate": round(len(ok) / max(len(rows), 1), 3),
        "termination_reasons": {r: sum(x.get("termination_reason") == r for x in rows) for r in {x.get("termination_reason") for x in rows}},
        "distinct_depots": len({d["depot"] for d in draws}),
        "distinct_customers_used": len(set(all_customers)),
        "distinct_scenarios": len({(d["depot"], tuple(d["customers"])) for d in draws}),
        "episodes_needing_charge": sum(r["charging_sessions"] > 0 for r in rows),
        "metrics": {k: summarise(rows, k) for k in (
            "total_distance", "total_energy_consumed", "total_energy_charged", "charging_sessions", "total_charging_time",
            "total_queue_time", "fleet_makespan", "total_operating_time", "minimum_terminal_soc", "steps")},
        "seconds_per_scenario_median": round(st.median([d["seconds"] for d in draws]), 1) if draws else None,
        "seconds_to_generate_median": round(st.median([d["seconds_to_generate"] for d in draws]), 1) if draws else None,
        "scenarios": draws,
    }
    out = data_dir / "scenario_check.json" if in_case_folder else Path("scenario_check_" + name + ".json")
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios"}, indent=1), flush=True)

    # 3. the environment's own episode figures (on the case's road backdrop)
    if a.plot_episodes:
        penv = EventDrivenTruckEnv(a.config, verbose=False, enable_plotting=True, run_id=f"scenario_check_{name}")
        for k in range(a.plot_episodes):
            play(penv, policy, a.max_steps, seed=a.seed + k)
        penv.close()
        print("episode figures in EVRoutingEnv/results/scenario_check_" + name, flush=True)
    return 0 if not failures and rows else 1


if __name__ == "__main__":
    sys.exit(main())
