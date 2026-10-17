"""Pre-generate certified scenarios for a config so that training and evaluation resets are instant.

    python scripts/casestudies/generate_scenarios.py EVRoutingEnv/config_files/config_joint_texas.yaml \
        --seeds 0:2000 --workers 16 --out /path/to/scenario_cache

A reset on a state-scale case spends seconds to a minute in the solvability certificate (a search
through hundreds of chargers). The result depends only on the scenario seed, the config and the
network, so it is generated once here and reused: point ``environment.scenario_cache_dir`` in the config
(or the ``EVRP_SCENARIO_CACHE`` environment variable) at ``--out``; ``reset(seed=s)`` then loads the
stored instance in milliseconds and the episode is identical to one generated on the spot.

The first load of a large case also computes all-pairs routes once (minutes, cached in $TMPDIR); run
this on a compute node, e.g. through sbatch, for big cases.
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

_ENV = None


def _init(config, out):
    global _ENV
    os.environ["EVRP_SCENARIO_CACHE"] = out
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv

    _ENV = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def _generate(seed):
    t = time.time()
    try:
        _ENV.reset(seed=seed)
        return seed, round(time.time() - t, 2), None
    except Exception as e:  # noqa: BLE001 - report scenarios that cannot be certified
        return seed, round(time.time() - t, 2), f"{type(e).__name__}: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", default="0:100", help="start:stop (stop exclusive)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    start, stop = (int(x) for x in a.seeds.split(":"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0, done, bad = time.time(), 0, []
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.config, a.out)) as pool:
        for seed, sec, err in pool.imap_unordered(_generate, range(start, stop)):
            done += 1
            if err:
                bad.append((seed, err))
            if done % 10 == 0 or done == stop - start:
                print(f"{done}/{stop - start} scenarios, {time.time() - t0:.0f}s elapsed, last {sec}s", flush=True)
    print(f"generated {done - len(bad)} scenarios, {len(bad)} failed")
    for seed, err in bad:
        print("  seed", seed, err)


if __name__ == "__main__":
    main()
