"""Compare the simulator's charging curve against the standard alternatives.

Ziyan's review asks for the nonlinear charging equation to be sourced and
validated, and for a comparison against an established three-segment/piecewise
formulation of the Montoya type.  This script produces that comparison as
numbers rather than prose:

* ``cccv``    -- the curve the simulator integrates: a four-phase constant
  current / constant voltage profile (ramp to 10%, ramp to 50%, plateau to the
  taper point, then a tapering tail);
* ``linear``  -- energy over rated power, the model most routing papers assume;
* ``montoya`` -- a concave piecewise-linear charging function interpolating the
  CCCV curve at its phase boundaries, i.e. the approximation Montoya et al.
  (2017) optimise over.

For each station power class the script reports the charging time each model
predicts for a set of (initial SoC, target SoC) pairs, and the error each
approximation makes against the integrated curve.  The point is not that one
model is right: it is to quantify how much the modelling choice moves the
quantity a routing policy trades against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.utils.utils import load_config


def _pairs() -> list[tuple[float, float]]:
    """Transitions a target-SoC policy can actually request."""
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    return [
        (start, target)
        for start in starts
        for target in levels
        if target > start + 1e-9
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--output", default="results/charging_curves/model_comparison.json"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    charging = config["charging"]
    battery_capacity = float(config["truck"]["battery_capacity"])
    efficiency = float(charging["dcfast"]["efficiency"])
    taper_start = float(charging["dcfast"]["taper_start_soc"])
    taper_min = float(charging["dcfast"]["taper_power_min"])
    model = ChargingCurveModel()

    report = {
        "battery_capacity_kwh": battery_capacity,
        "efficiency": efficiency,
        "taper_start_soc": taper_start,
        "taper_power_min_kw": taper_min,
        "power_classes": {},
    }

    for peak_power in [float(value) for value in charging["station_power_classes_kw"]]:
        charger_config = {
            "charge_rate": peak_power,
            "efficiency": efficiency,
            "use_realistic_curve": True,
            "taper_start_soc": taper_start,
            "taper_power_min": min(taper_min, peak_power),
        }
        # Three segments is the textbook Montoya form. The refined variant adds
        # one breakpoint inside the taper, which is where a single linear piece
        # cannot follow the curvature; reporting both answers "how many
        # segments does this curve actually need" rather than asserting it.
        # The curve has two sources of curvature: the ramp below 50% SoC and the
        # taper above the taper point. The literature's three-segment form
        # straddles both with one piece each, so the refinements subdivide them
        # in turn -- that is what makes the answer "how many segments" rather
        # than "three is or is not enough".
        segment_sets = {
            "montoya_3": (0.5, 0.8, 1.0),
            "montoya_4": (0.5, 0.8, 0.9, 1.0),
            "montoya_5": (0.1, 0.5, 0.8, 0.9, 1.0),
            "montoya_7": (0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 1.0),
        }
        breakpoint_sets = {
            name: model.montoya_breakpoints(
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency,
                taper_start_soc=taper_start,
                taper_power_min=min(taper_min, peak_power),
                boundaries=boundaries,
            )
            for name, boundaries in segment_sets.items()
        }
        breakpoints = breakpoint_sets["montoya_3"]

        rows = []
        for initial_soc, target_soc in _pairs():
            _, details = model.calculate_charge_to_target(
                initial_soc=initial_soc,
                target_soc=target_soc,
                battery_capacity=battery_capacity,
                charger_config=charger_config,
                charger_type="DCFast",
            )
            cccv_hours = float(details["actual_charge_hours"])
            energy = (target_soc - initial_soc) * battery_capacity
            linear_hours = energy / (peak_power * efficiency)
            row = {
                "initial_soc": initial_soc,
                "target_soc": target_soc,
                "cccv_hours": cccv_hours,
                "linear_hours": linear_hours,
                "linear_error": linear_hours - cccv_hours,
                "linear_relative_error": (linear_hours - cccv_hours) / cccv_hours,
            }
            for name, points in breakpoint_sets.items():
                hours = model.montoya_time_to_soc(points, initial_soc, target_soc)
                row[f"{name}_hours"] = hours
                row[f"{name}_relative_error"] = (hours - cccv_hours) / cccv_hours
            row["montoya_hours"] = row["montoya_3_hours"]
            row["montoya_relative_error"] = row["montoya_3_relative_error"]
            rows.append(row)

        def summarize(key: str) -> dict:
            values = [abs(row[key]) for row in rows]
            signed = [row[key] for row in rows]
            return {
                "mean_absolute": sum(values) / len(values),
                "max_absolute": max(values),
                "mean_signed": sum(signed) / len(signed),
            }

        entry = {
            "peak_power_kw": peak_power,
            "montoya_breakpoints": breakpoints,
            "montoya_breakpoint_sets": breakpoint_sets,
            "transitions": rows,
            "linear_relative_error": summarize("linear_relative_error"),
        }
        for name in breakpoint_sets:
            entry[f"{name}_relative_error"] = summarize(f"{name}_relative_error")
        entry["montoya_relative_error"] = entry["montoya_3_relative_error"]
        report["power_classes"][str(int(peak_power))] = entry

        parts = [
            f"linear {entry['linear_relative_error']['mean_absolute'] * 100:5.1f}%"
            f"/{entry['linear_relative_error']['max_absolute'] * 100:5.1f}%"
        ]
        parts.extend(
            f"{name.split('_')[1]}-seg "
            f"{entry[f'{name}_relative_error']['mean_absolute'] * 100:5.1f}%"
            f"/{entry[f'{name}_relative_error']['max_absolute'] * 100:5.1f}%"
            for name in breakpoint_sets
        )
        print(f"{peak_power:6.0f} kW  mean/max |err|  " + "  ".join(parts), flush=True)

    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
