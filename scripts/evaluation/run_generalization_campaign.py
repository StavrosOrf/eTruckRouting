"""Score the frozen methods outside the distribution they were tuned on.

R1.7 asks for generalization evidence that separates three different claims,
which the campaign therefore keeps separate:

``interpolation``
    the same generator, unseen scenarios -- this is the ordinary test split and
    is included as the control every other regime is read against;
``size_transfer``
    the same physics with fewer customers or trucks.  The action envelope is
    held at the trained width and the surplus slots report as empty, so this is
    a genuinely smaller instance rather than a differently shaped observation;
``ood``
    a generator the policy never saw -- different charging infrastructure,
    battery, demand, service, road distances, or uncertainty law.

Every regime scores *every* method, learned and classical, on the same held-out
seeds, so a regime where the policy degrades is reported next to what the
planner and the heuristic do on that same regime rather than in isolation.

Nothing here is selected on: the checkpoints and baseline settings arrive frozen
from validation, and every regime writes the full artifact contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.evaluation.artifacts import collect_run_manifest
from EVRoutingEnv.evaluation.runner import run_evaluation_campaign
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds
from scripts.evaluation.run_canonical_campaign import build_policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# Each regime is a set of config-section overrides applied to the campaign
# config.  The comment on each says what a reviewer should read it as.
REGIMES: dict[str, dict] = {
    # -- control -------------------------------------------------------------
    "in_distribution": {
        "kind": "interpolation",
        "description": "the training generator, unseen seeds",
        "overrides": {},
    },
    # -- size transfer -------------------------------------------------------
    "customers_4": {
        "kind": "size_transfer",
        "description": "4 customers instead of 10",
        "overrides": {"environment": {"instance_num_stops": 4}},
    },
    "customers_6": {
        "kind": "size_transfer",
        "description": "6 customers instead of 10",
        "overrides": {"environment": {"instance_num_stops": 6}},
    },
    "customers_8": {
        "kind": "size_transfer",
        "description": "8 customers instead of 10",
        "overrides": {"environment": {"instance_num_stops": 8}},
    },
    "fleet_1": {
        "kind": "size_transfer",
        "description": "a single truck serving the same customer set",
        "overrides": {
            "environment": {"num_trucks": 1, "canonical_max_trucks": 2},
        },
    },
    # -- scale grid, for the 4-truck / 14-customer envelope policy only ------
    # These require --config config_joint_scale.yaml: the headline policy has a
    # fixed observation width and cannot be evaluated above its own envelope,
    # which is exactly why a separate envelope policy exists.
    "scale_1t4c": {
        "kind": "size_transfer",
        "description": "1 truck, 4 customers",
        "overrides": {
            "environment": {
                "num_trucks": 1,
                "canonical_max_trucks": 4,
                "instance_num_stops": 4,
                "allow_variable_num_stops": False,
            }
        },
    },
    "scale_2t8c": {
        "kind": "size_transfer",
        "description": "2 trucks, 8 customers",
        "overrides": {
            "environment": {
                "num_trucks": 2,
                "canonical_max_trucks": 4,
                "instance_num_stops": 8,
                "allow_variable_num_stops": False,
            }
        },
    },
    "scale_3t11c": {
        "kind": "size_transfer",
        "description": "3 trucks, 11 customers",
        "overrides": {
            "environment": {
                "num_trucks": 3,
                "canonical_max_trucks": 4,
                "instance_num_stops": 11,
                "allow_variable_num_stops": False,
            }
        },
    },
    "scale_4t14c": {
        "kind": "size_transfer",
        "description": "4 trucks, 14 customers -- the full envelope",
        "overrides": {
            "environment": {
                "num_trucks": 4,
                "canonical_max_trucks": 4,
                "instance_num_stops": 14,
                "allow_variable_num_stops": False,
            }
        },
    },
    # The one mechanism the two-truck distribution never exercises: with four
    # trucks and a single port per station, charger contention finally binds.
    "congestion_4t_1port": {
        "kind": "ood",
        "description": "4 trucks, one port per station, so queues bind",
        "overrides": {
            "environment": {
                "num_trucks": 4,
                "canonical_max_trucks": 4,
                "instance_num_stops": 14,
                "allow_variable_num_stops": False,
            },
            "charging": {"port_capacity_scale": 0.02},
        },
    },
    # -- charging infrastructure --------------------------------------------
    "chargers_weak": {
        "kind": "ood",
        "description": "slower stations: 50/150/350 kW classes",
        "overrides": {
            "charging": {"station_power_classes_kw": [50.0, 150.0, 350.0]},
        },
    },
    "chargers_strong": {
        "kind": "ood",
        "description": "faster stations: 350/750/1000 kW classes",
        "overrides": {
            "charging": {"station_power_classes_kw": [350.0, 750.0, 1000.0]},
        },
    },
    "chargers_inefficient": {
        "kind": "ood",
        "description": "0.80 charging efficiency instead of 0.90",
        "overrides": {
            "charging": {
                "level2": {"charge_rate": 150.0, "efficiency": 0.80},
                "dcfast": {
                    "charge_rate": 350.0,
                    "efficiency": 0.80,
                    "taper_start_soc": 0.8,
                    "taper_power_min": 150.0,
                },
            },
        },
    },
    "ports_scarce": {
        "kind": "ood",
        "description": "a third of the charging ports, so queues bind",
        "overrides": {"charging": {"port_capacity_scale": 0.34}},
    },
    "ports_plentiful": {
        "kind": "ood",
        "description": "triple the charging ports, so queues rarely bind",
        "overrides": {"charging": {"port_capacity_scale": 3.0}},
    },
    # -- vehicle -------------------------------------------------------------
    "battery_small": {
        "kind": "ood",
        "description": "300 kWh battery instead of 400",
        "overrides": {"truck": {"battery_capacity": 300.0}},
    },
    "battery_large": {
        "kind": "ood",
        "description": "500 kWh battery instead of 400",
        "overrides": {"truck": {"battery_capacity": 500.0}},
    },
    # truck.base_speed is deliberately absent as a regime: travel times are read
    # from the network tables rather than derived from it, so changing it leaves
    # every leg identical. The road network is perturbed directly instead.
    "network_slow": {
        "kind": "ood",
        "description": "every leg takes 40% longer",
        "overrides": {"network": {"travel_time_scale": 1.4}},
    },
    "network_fast": {
        "kind": "ood",
        "description": "every leg takes 25% less time",
        "overrides": {"network": {"travel_time_scale": 0.75}},
    },
    "energy_hungry": {
        "kind": "ood",
        "description": "every leg draws 30% more energy",
        "overrides": {"network": {"energy_scale": 1.3}},
    },
    "energy_frugal": {
        "kind": "ood",
        "description": "every leg draws 20% less energy",
        "overrides": {"network": {"energy_scale": 0.8}},
    },
    # -- demand --------------------------------------------------------------
    "demand_heavy": {
        "kind": "ood",
        "description": "customer demand up to 6.0, tightening payload",
        "overrides": {"problem": {"max_customer_demand": 6.0}},
    },
    "service_slow": {
        "kind": "ood",
        "description": "0.4 h base service time instead of 0.2",
        "overrides": {"problem": {"base_service_time": 0.4}},
    },
    # environment.min_hop_distance / max_hop_distance are deliberately absent
    # too: the joint instance generator does not read them at all -- they are
    # consumed only by the inherited preassigned-route generator -- so a regime
    # built on them produces the identical instance and would read as spurious
    # robustness.
    # -- uncertainty law -----------------------------------------------------
    "traffic_calm": {
        "kind": "ood",
        "description": "near-deterministic travel: 0.05 variance, no rush hour",
        "overrides": {
            "traffic": {"std_dev_factor": 0.05, "rush_hour_multiplier": 1.0},
        },
    },
    "traffic_severe": {
        "kind": "ood",
        "description": "0.30 travel variance and a 3x rush-hour multiplier",
        "overrides": {
            "traffic": {
                "std_dev_factor": 0.30,
                "max_std_dev_hours": 2.0,
                "rush_hour_multiplier": 3.0,
            },
        },
    },
    "energy_severe": {
        "kind": "ood",
        "description": "energy draw between 0.8x and 1.5x nominal",
        "overrides": {
            "traffic": {
                "energy_uncertainty_factor": 0.15,
                "min_energy_multiplier": 0.80,
                "max_energy_multiplier": 1.50,
            },
        },
    },
}


def _apply(config: dict, overrides: dict) -> dict:
    """Apply one regime's section overrides to a copy of the config."""
    result = deepcopy(config)
    for section, values in overrides.items():
        if section not in result:
            raise ValueError(f"unknown config section {section!r}")
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(result[section].get(key), dict):
                result[section][key].update(value)
            else:
                result[section][key] = value
    return result


def _run_one(job: tuple) -> dict:
    """Score one (regime, method) pair in its own process."""
    import torch

    # See run_canonical_campaign.score_method: one thread per worker, or N
    # processes each spawn a thread per core and the host thrashes.
    torch.set_num_threads(1)

    regime, method, settings, config_path, seeds, split, destination, max_steps = job
    definition = REGIMES[regime]
    config = _apply(load_config(config_path), definition["overrides"])
    # The regime says what changed about the world; the method says what
    # environment it was trained in. Both have to be applied, and the method's
    # comes last: scoring the unmasked arm under the hard mask it never trained
    # with reads as a collapse in generalization that is really an evaluation
    # error. No regime touches those keys, so the two never fight.
    config = _apply(config, settings.get("environment_overrides") or {})
    target = Path(destination) / regime / method
    if target.exists():
        summary_path = target / "summary.json"
        if not summary_path.exists():
            # The artifact contract refuses to overwrite, so an interrupted run
            # leaves its directory behind as evidence. Say so and skip this job
            # rather than reading a summary that was never written or silently
            # discarding the evidence; the rest of the campaign still completes.
            return {
                "regime": regime,
                "method": method,
                "summary": None,
                "cached": True,
                "incomplete": True,
            }
        summary = json.loads(summary_path.read_text())
        return {"regime": regime, "method": method, "summary": summary, "cached": True}

    manifest = collect_run_manifest(
        run_id=f"{method}__{regime}__{split}",
        algorithm=method,
        split=split,
        command=tuple(sys.argv),
        resolved_config=config,
        scenario_seeds=seeds,
        repository_root=REPOSITORY_ROOT,
        checkpoint=settings.get("checkpoint"),
    )

    def factory():
        return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)

    artifacts = run_evaluation_campaign(
        environment_factory=factory,
        policy=build_policy(method, settings),
        manifest=manifest,
        output_directory=target,
        max_policy_steps=max_steps,
    )
    return {
        "regime": regime,
        "method": method,
        "summary": json.loads(artifacts.summary_path.read_text()),
        "cached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", default="results/canonical/generalization")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--max-policy-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Subset of regimes to run; defaults to all of them.",
    )
    arguments = parser.parse_args()

    methods = json.loads(Path(arguments.methods).read_text())
    regimes = arguments.regimes or list(REGIMES)
    unknown = sorted(set(regimes) - set(REGIMES))
    if unknown:
        raise SystemExit(f"unknown regimes: {unknown}")

    seeds = split_seeds(arguments.split, arguments.scenarios, arguments.base_seed)
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            regime,
            method,
            settings,
            arguments.config,
            seeds,
            arguments.split,
            str(destination),
            arguments.max_policy_steps,
        )
        for regime in regimes
        for method, settings in methods.items()
    ]
    print(
        f"{len(jobs)} (regime, method) runs over {len(seeds)} {arguments.split} "
        f"scenarios on {arguments.workers} workers",
        flush=True,
    )

    completed: list[dict] = []
    if arguments.workers > 1:
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            for result in pool.map(_run_one, jobs):
                completed.append(result)
                _report(result)
    else:
        for job in jobs:
            result = _run_one(job)
            completed.append(result)
            _report(result)

    incomplete = [
        f"{result['regime']}/{result['method']}"
        for result in completed
        if result.get("incomplete")
    ]
    if incomplete:
        print(
            "\nincomplete runs left by an earlier interrupted campaign; delete "
            "these directories to retry them:",
            flush=True,
        )
        for name in incomplete:
            print(f"  {destination / name}", flush=True)

    index: dict[str, dict] = {}
    for result in completed:
        if result["summary"] is None:
            continue
        regime = result["regime"]
        entry = index.setdefault(
            regime,
            {
                "kind": REGIMES[regime]["kind"],
                "description": REGIMES[regime]["description"],
                "overrides": REGIMES[regime]["overrides"],
                "methods": {},
            },
        )
        entry["methods"][result["method"]] = result["summary"]

    (destination / "generalization_index.json").write_text(
        json.dumps(
            {
                "split": arguments.split,
                "scenario_seeds": seeds,
                "regimes": index,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"generalization artifacts under {destination}", flush=True)


def _report(result: dict) -> None:
    if result["summary"] is None:
        print(
            f"  {result['regime']:22s} {result['method']:12s} INCOMPLETE "
            "(directory exists without a summary)",
            flush=True,
        )
        return
    aggregate = result["summary"]["aggregate"]
    travel = aggregate["metrics"].get("total_travel_time", {}).get("mean")
    print(
        f"  {result['regime']:22s} {result['method']:12s} "
        f"success={aggregate['success_rate']:.3f} "
        f"travel={'n/a' if travel is None else f'{travel:.1f}'}"
        f"{' (cached)' if result['cached'] else ''}",
        flush=True,
    )


if __name__ == "__main__":
    main()
