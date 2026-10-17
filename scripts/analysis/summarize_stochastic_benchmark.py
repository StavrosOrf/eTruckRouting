"""Aggregate recorded simulator benchmarks and render a standalone scaling figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(paths):
    summaries = []
    for path in paths:
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[(row["trucks"], row["customers"], row["horizon_hours"])].append(row)
        suite = path.stem.removeprefix("stochastic_scale_").removesuffix("_2026-10-17")
        for (trucks, customers, horizon), rows in groups.items():
            ended = [r for r in rows if r["status"] == "completed"]
            progress = [r for r in rows if r["status"] == "progress"]
            timings = ended or progress[-1:]
            result = {
                "suite": suite,
                "trucks": trucks,
                "customers": customers,
                "horizon_hours": horizon,
                "battery_kwh": metadata["arguments"].get("battery_kwh", 400.0),
                "step_mode": metadata["arguments"]["step_mode"],
                "requested_repeats": metadata["arguments"]["repeats"],
                "completed_repeats": len(ended),
                "successful_routes": sum(r["success"] for r in ended),
                "cutoff": next(
                    (
                        r["status"]
                        for r in reversed(rows)
                        if r["status"].endswith("_budget")
                    ),
                    "",
                ),
                "case_budget_seconds": metadata["arguments"]["case_seconds"],
                "errors": [r["error"] for r in rows if r["status"] == "error"],
                "termination_counts": dict(
                    Counter(r["termination_reason"] for r in ended)
                ),
                "partial_steps": progress[-1]["steps"] if progress else None,
                "partial_simulated_hours": progress[-1]["simulated_hours"]
                if progress
                else None,
                "step_statistics_from": "completed episodes"
                if ended
                else "last partial episode",
                "peak_rss_mib": max(
                    (r.get("peak_rss_mib", r.get("rss_mib", 0)) for r in rows),
                    default=0,
                ),
                "observation_bytes": max(
                    (r.get("observation_bytes", 0) for r in rows), default=0
                ),
            }
            for key in (
                "episode_seconds",
                "steps",
                "simulated_hours",
                "served",
                "reset_seconds",
                "simulator_seconds",
                "policy_seconds",
            ):
                values = [r[key] for r in ended]
                result[f"{key}_median"] = median(values) if values else None
                if key == "episode_seconds":
                    result[f"{key}_min"] = min(values) if values else None
                    result[f"{key}_max"] = max(values) if values else None
            for key in ("step_median_ms", "step_p95_ms"):
                result[key] = median(r[key] for r in timings) if timings else None
            summaries.append(result)
    return summaries


def curve(ax, rows, label, color):
    measured = sorted(
        (r for r in rows if r["completed_repeats"]), key=lambda r: r["trucks"]
    )
    if measured:
        x = [r["trucks"] for r in measured]
        y = [r["episode_seconds_median"] for r in measured]
        ax.plot(x, y, "o-", label=label, color=color)
        ax.fill_between(
            x,
            [r["episode_seconds_min"] for r in measured],
            [r["episode_seconds_max"] for r in measured],
            alpha=0.15,
            color=color,
        )
    cut = [
        r for r in rows if r["cutoff"] == "wall_budget" and not r["completed_repeats"]
    ]
    if cut:
        ax.scatter(
            [r["trucks"] for r in cut],
            [r["case_budget_seconds"] for r in cut],
            marker="^",
            facecolors="none",
            edgecolors=color,
            label="Case budget reached; episode unfinished",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Configured trucks",
        ylabel="Wall seconds per episode",
    )
    ax.grid(alpha=0.25, which="both")


def figure(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    default = [r for r in rows if r["suite"] == "trucks"]
    control = [r for r in rows if r["suite"] == "energy_loose"]
    curve(axes[0, 0], default, "400 kWh, 40 customers", "tab:blue")
    axes[0, 0].set_title("Default battery: ended runs reach 200 h; routes fail")
    axes[0, 0].legend(fontsize=8)
    for customers, color in [(8, "tab:green"), (32, "tab:orange")]:
        data = sorted(
            (
                r
                for r in rows
                if r["suite"] == "horizons" and r["customers"] == customers
            ),
            key=lambda r: r["horizon_hours"],
        )
        axes[0, 1].plot(
            [r["horizon_hours"] for r in data],
            [r["episode_seconds_median"] for r in data],
            "o-",
            label=f"{customers // 4} trucks / {customers} customers",
            color=color,
        )
    axes[0, 1].set(
        title="Horizon sweep: medians of 3 episodes",
        xlabel="Simulated horizon (hours)",
        ylabel="Wall seconds per episode",
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25)
    curve(axes[1, 0], control, "10,000 kWh control, 40 customers", "tab:purple")
    axes[1, 0].set_title("Large-battery control: routes succeed where measured")
    axes[1, 0].legend(fontsize=8)
    for series, label, color in [
        (default, "400 kWh", "tab:blue"),
        (control, "10,000 kWh control", "tab:purple"),
    ]:
        axes[1, 1].plot(
            [r["trucks"] for r in series],
            [r["peak_rss_mib"] for r in series],
            "o-",
            label=label,
            color=color,
        )
    axes[1, 1].axhline(
        2048, color="tab:red", linestyle="--", label="Experiment RSS budget"
    )
    axes[1, 1].set(
        xscale="log",
        xlabel="Configured trucks (40 customers)",
        ylabel="Peak observed worker RSS (MiB)",
        title="Includes imports and shared pages; sampling can miss peaks",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(
        "Stochastic simulator scaling • 17 October 2026\nTravel, energy and service noise enabled; shared CPU host, one math thread",
        fontsize=13,
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=Path("notes"))
    args = parser.parse_args()
    output = args.notes / "stochastic_scale_summary_2026-10-17"
    rows = summarize(sorted(args.notes.glob("stochastic_scale_*_2026-10-17.jsonl")))
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )
    figure(rows, args.notes / "stochastic_scale_2026-10-17")
    print(f"Summarized {len(rows)} configurations to {output}")


if __name__ == "__main__":
    main()
