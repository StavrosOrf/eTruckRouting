"""Run heuristic-policy episodes on a case-study config to verify it works end to end.

    python scripts/casestudies/smoke_episode.py EVRoutingEnv/config_files/config_joint_seattle.yaml \
        --episodes 3 --plot

With --plot the environment writes its usual episode figures (initial plan, final routes, charger
queues) under EVRoutingEnv/results/<run_id>/, drawn on the case study's own road backdrop.
"""
import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy  # noqa: E402
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    t0 = time.time()
    env = EventDrivenTruckEnv(a.config, verbose=False, enable_plotting=a.plot, run_id=a.run_id)
    print(f"environment ready in {time.time() - t0:.1f}s: {env.transport_graph.num_nodes} nodes, "
          f"{len(env.transport_graph.get_charging_nodes())} charger nodes, battery "
          f"{env.config['truck']['battery_capacity']} kWh")
    policy = HeuristicPolicy(verbose=False)
    for ep in range(a.episodes):
        env.reset(seed=a.seed + ep)
        done, steps, total = False, 0, 0.0
        while not done:
            action = policy.get_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            total += reward; steps += 1
            done = terminated or truncated
        print(f"episode {ep}: steps={steps} reward={total:.1f} terminated={terminated} "
              f"info keys={sorted(k for k in info if k != 'scenario')[:6]}")
        om = info.get("operational_metrics") or {}
        if om:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(om.items())[:10]})
    env.close()


if __name__ == "__main__":
    main()
