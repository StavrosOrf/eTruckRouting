"""Draw the passing policy against the best-known solution on the map.

Run from the repository root, after ``policy_vs_bks_traces.py``:
    PYTHONPATH=. python scripts/analysis/policy_vs_bks_maps.py TRACES_DIR OUT_DIR

One figure per episode: on the map, the legs both solutions drive in a neutral
tone and only the differences in colour; below it, what each truck did over
time. An overview puts the map panels of all episodes on one page, and
``--animate NAME`` writes a side-by-side animation of one episode.

The best-known solution is the best of a portfolio of planners on that seed. It
is not a proven optimum. Lines join consecutive stops; trucks follow the roads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "results/generalist_graphppo/suite_v4/configs"
NODE_MAPS = ROOT / "results/generalist_graphppo/agent/visuals/node_maps.json"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
ROAD = "#dcdbd5"
SHARED = "#3b3a37"
POLICY = "#2a78d6"
BEST = "#eb6834"
MISSED = "#c0272d"
ACTIVITY = {"routing": "#8a8984", "charging": "#1baf7a", "unloading": "#6250d6", "waiting": "#eda100"}
ACTIVITY_LABEL = {"routing": "Driving", "charging": "Charging", "unloading": "Service", "waiting": "Waiting, queueing or rest"}
PROBLEM = {
    "tx_longhaul": "Texas long-haul",
    "tx_longhaul_congested": "Texas congested",
    "pnw_longhaul_hos": "Northwest, driver rest rules",
    "ca_sameday": "California same-day",
    "ca_drayage_combined": "California drayage",
}
CASE = {
    "median": "median seed",
    "p90": "90th-percentile seed",
    "p10": "10th-percentile seed",
    "unsuccessful": "unsuccessful seed",
}
MOVES = ("CUSTOMER", "CHARGER", "DEPOT")


class Region:
    """A case study's coordinates and road backdrop."""

    def __init__(self, problem: str):
        config = yaml.safe_load((CONFIGS / f"{problem}.yaml").read_text())
        folder = ROOT / config["network"]["data_path"] / "vis_data"
        table = json.loads(NODE_MAPS.read_text())[config["network"]["data_path"]]
        self.nodes = {int(node): tuple(place) for node, place in table.items()}
        roads = np.load(folder / "roads.npz")
        major = roads["cls"] <= np.quantile(roads["cls"], 0.25)
        corners = [roads[key].astype(float)[major] / 1e5 for key in ("lon0", "lat0", "lon1", "lat1")]
        self.roads = np.stack([np.stack(corners[:2], 1), np.stack(corners[2:], 1)], 1)
        self.aspect = 1.0 / math.cos(math.radians(np.mean([lat for _, lat in self.nodes.values()])))
        self.horizon = float(config["environment"]["max_time"])

    def at(self, node: int) -> tuple[float, float] | None:
        return self.nodes.get(int(node))


def legs(result: dict) -> set[tuple[int, int]]:
    """The directed moves between different stops, whichever truck drove them."""
    moves = set()
    for row in result["decisions"]:
        if row["kind"] in MOVES and row["target"] is not None and row["target"] != row["node"]:
            moves.add((int(row["node"]), int(row["target"])))
    return moves


def stops(result: dict, kinds: tuple[str, ...]) -> set[int]:
    return {int(row["target"]) for row in result["decisions"] if row["kind"] in kinds and row["target"] is not None}


def charged_at(result: dict) -> set[int]:
    return {int(row["node"]) for row in result["decisions"] if row["kind"] == "CHARGE"}


def _segments(region: Region, pairs) -> list:
    return [[region.at(a), region.at(b)] for a, b in pairs if region.at(a) and region.at(b)]


def _frame(region: Region, ax, points: list) -> None:
    """Fit the axes to the episode with a small margin, keeping true proportions."""
    lons, lats = zip(*points)
    pad = max(max(lons) - min(lons), max(lats) - min(lats)) * 0.07 + 0.05
    ax.set_xlim(min(lons) - pad, max(lons) + pad)
    ax.set_ylim(min(lats) - pad, max(lats) + pad)
    ax.set_aspect(region.aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def overlay(ax, region: Region, trace: dict) -> dict:
    """Draw both solutions on one map; returns the leg counts."""
    policy, best = trace["policy"], trace["bks"]
    mine, theirs = legs(policy), legs(best)
    together, only_mine, only_theirs = mine & theirs, mine - theirs, theirs - mine
    ax.set_facecolor(SURFACE)
    ax.add_collection(LineCollection(region.roads, colors=ROAD, linewidths=0.35, zorder=1))
    ax.add_collection(LineCollection(_segments(region, together), colors=SHARED, linewidths=1.3, zorder=3))
    ax.add_collection(LineCollection(_segments(region, only_theirs), colors=BEST, linewidths=1.5, linestyles=(0, (4, 2)), zorder=4))
    ax.add_collection(LineCollection(_segments(region, only_mine), colors=POLICY, linewidths=1.5, zorder=5))
    customers = stops(policy, ("CUSTOMER",)) | stops(best, ("CUSTOMER",))
    missed = stops(best, ("CUSTOMER",)) - stops(policy, ("CUSTOMER",))
    chargers = stops(policy, ("CHARGER",)) | stops(best, ("CHARGER",)) | charged_at(policy) | charged_at(best)
    depots = {int(row["node"]) for row in best["decisions"][:1]}
    def scatter(nodes, **style):
        places = [region.at(node) for node in nodes if region.at(node)]
        if places:
            ax.scatter(*zip(*places), **style)
    scatter(chargers, s=20, marker="s", facecolor=ACTIVITY["charging"], edgecolor=SURFACE, linewidth=0.6, zorder=6)
    scatter(customers - missed, s=18, marker="o", facecolor=INK, edgecolor=SURFACE, linewidth=0.6, zorder=7)
    scatter(missed, s=70, marker="X", facecolor=MISSED, edgecolor=SURFACE, linewidth=0.8, zorder=8)
    scatter(depots, s=150, marker="*", facecolor=INK, edgecolor=SURFACE, linewidth=0.8, zorder=9)
    points = [region.at(node) for node in customers | chargers | depots if region.at(node)]
    _frame(region, ax, points)
    return {"shared": len(together), "policy": len(only_mine), "best": len(only_theirs), "missed": len(missed), "chargers": len(chargers)}


def timeline(ax, region: Region, result: dict, name: str, end: float) -> None:
    """One row per working truck, what it did and when."""
    used = [truck for truck in result["trucks"] if any((row["duration"] or 0) > 0 for row in truck["timeline"])]
    for row_index, truck in enumerate(used):
        for row in truck["timeline"]:
            start, length = float(row["start_time"]), float(row["duration"] or 0.0)
            if length <= 0.0:
                continue
            ax.barh(row_index, length, left=start, height=0.72, linewidth=0,
                    color=ACTIVITY.get(row["activity"], ACTIVITY["waiting"]))
    ax.axvline(region.horizon, color=SECONDARY, linewidth=1.0, linestyle=(0, (2, 2)))
    ax.set_ylim(-0.6, max(len(used), 1) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, end)
    ax.set_yticks(range(len(used)), [f"T{truck['id']}" for truck in used], fontsize=7, color=SECONDARY)
    ax.tick_params(axis="both", length=0, labelsize=8, labelcolor=SECONDARY)
    ax.set_title(name, loc="left", fontsize=9.5, color=INK, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ROAD)


def _outcome(summary: dict) -> str:
    text = f"{summary['served']}/{summary['tasks']} served, {summary['hours']:.1f} fleet hours"
    if summary["failed"]:
        return text + ", failed"
    if not summary["success"] and not summary["dynamic"]:
        return text + ", not finished in time"
    return text


def _short(summary: dict) -> str:
    ending = "fleet stopped" if summary["failed"] else "out of time"
    return f"{summary['served']}/{summary['tasks']} served, {ending}"


def headline(trace: dict) -> str:
    policy, best = trace["policy"]["summary"], trace["bks"]["summary"]
    case = trace["case"]
    title = f"{PROBLEM[case['problem']]}, {CASE[case['label']]}"
    if policy["dynamic"]:
        return f"{title}: policy {policy['served']} on time, best-known {best['served']}"
    if not policy["success"]:
        return f"{title}: policy {_short(policy)}"
    return f"{title}: policy {policy['hours'] / best['hours'] - 1:+.1%} hours vs best-known"


def longest_wait(result: dict) -> tuple[int, float, float, int | None] | None:
    """The longest wait of the episode as ``(truck, start, hours, node)``, if it is long enough to matter."""
    longest = None
    for truck in result["trucks"]:
        for row in truck["timeline"]:
            if row["activity"] == "waiting" and (longest is None or float(row["duration"] or 0.0) > longest[2]):
                longest = (int(truck["id"]), float(row["start_time"]), float(row["duration"] or 0.0))
    if longest is None or longest[2] < 8.0:
        return None
    before = [row for row in result["decisions"] if row["truck"] == longest[0] and row["clock"] <= longest[1] + 1e-6]
    return (*longest, int(before[-1]["node"]) if before else None)


def episode_figure(trace: dict, region: Region, target: Path) -> None:
    figure = plt.figure(figsize=(13.4, 7.6), dpi=160, facecolor=SURFACE)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.08], left=0.015, right=0.985, top=0.87, bottom=0.135, hspace=0.27, wspace=0.07)
    ax = figure.add_subplot(grid[:, 0])
    counts = overlay(ax, region, trace)
    policy, best = trace["policy"], trace["bks"]
    end = max(region.horizon, policy["clock"], best["clock"]) * 1.02
    top, bottom = figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1])
    timeline(top, region, best, f"Best-known ({trace['bks_method']}): {_outcome(best['summary'])}", end)
    timeline(bottom, region, policy, f"Policy: {_outcome(policy['summary'])}", end)
    bottom.set_xlabel("Hours into the episode (dotted line: horizon)", fontsize=8.5, color=SECONDARY)
    note = f"{counts['shared']} legs shared, {counts['policy']} only in the policy, {counts['best']} only in the best-known solution."
    wait = longest_wait(policy)
    if wait and not policy["summary"]["dynamic"]:
        note += f" Longest policy wait: truck T{wait[0]}, {wait[2]:.0f} h from hour {wait[1]:.0f}."
        place = region.at(wait[3]) if wait[3] is not None else None
        if place:
            ax.annotate(f"T{wait[0]} waits {wait[2]:.0f} h", place, xytext=(12, 12), textcoords="offset points", fontsize=9, color=INK,
                        arrowprops={"arrowstyle": "-", "color": SECONDARY, "linewidth": 0.8}, zorder=10,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9})
            ax.scatter(*place, s=90, facecolor="none", edgecolor="#eda100", linewidth=2.0, zorder=9)
    if not trace["policy_row_match"]:
        note += " This CPU replay differs from the recorded test row."
    figure.text(0.015, 0.945, headline(trace), fontsize=14, color=INK, weight="bold")
    figure.text(0.015, 0.908, note, fontsize=10, color=SECONDARY)
    figure.legend(handles=[
        Line2D([], [], color=SHARED, linewidth=1.5, label="Leg in both"),
        Line2D([], [], color=POLICY, linewidth=1.5, label="Policy only"),
        Line2D([], [], color=BEST, linewidth=1.5, linestyle=(0, (4, 2)), label="Best-known only"),
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="Customer or request stop"),
        *([Line2D([], [], marker="s", linestyle="", color=ACTIVITY["charging"], markersize=4.5, label="Charging stop")] if counts["chargers"] else []),
        Line2D([], [], marker="*", linestyle="", color=INK, markersize=8, label="Depot"),
        *([Line2D([], [], marker="X", linestyle="", color=MISSED, markersize=7, label="Missed by the policy")] if counts["missed"] else []),
    ], loc="lower left", bbox_to_anchor=(0.01, 0.052), ncol=4, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.2)
    figure.legend(handles=[Patch(color=ACTIVITY[key], label=ACTIVITY_LABEL[key]) for key in ("routing", "charging", "unloading", "waiting")],
                  loc="lower right", bbox_to_anchor=(0.985, 0.052), ncol=4, frameon=False, fontsize=8.5, handlelength=1.1, columnspacing=1.2)
    figure.text(0.015, 0.02, "Best-known is the best of a planner portfolio on this seed, not a proven optimum. Lines join stops; trucks follow the roads. Gaps in a bar: the truck is parked.",
                fontsize=8.5, color=SECONDARY)
    figure.savefig(target, facecolor=SURFACE)
    plt.close(figure)


def overview(traces: list[dict], regions: dict, target: Path) -> None:
    columns = ["median", "p90", "unsuccessful"]
    rows = ["tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos"]
    figure, axes = plt.subplots(len(rows), len(columns), figsize=(13, 12.6), dpi=170, facecolor=SURFACE)
    found = {(trace["case"]["problem"], trace["case"]["label"]): trace for trace in traces}
    for row, problem in enumerate(rows):
        for column, label in enumerate(columns):
            ax = axes[row][column]
            trace = found.get((problem, label))
            if trace is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "No unsuccessful seed\nin 500", ha="center", va="center", fontsize=10, color=SECONDARY, transform=ax.transAxes)
                continue
            counts = overlay(ax, regions[problem], trace)
            policy, best = trace["policy"]["summary"], trace["bks"]["summary"]
            excess = f"{policy['hours'] / best['hours'] - 1:+.1%} hours" if policy["success"] else _short(policy)
            name = "Unsuccessful" if label == "unsuccessful" else CASE[label].capitalize()
            ax.set_title(f"{name}: {excess}\n{counts['shared']} shared, {counts['policy']} policy only, {counts['best']} best-known only",
                         loc="left", fontsize=9, color=INK)
        axes[row][0].text(-0.04, 0.5, PROBLEM[problem], rotation=90, va="center", ha="right", fontsize=10.5, color=INK, transform=axes[row][0].transAxes)
    figure.legend(handles=[
        Line2D([], [], color=SHARED, linewidth=1.5, label="Leg in both"),
        Line2D([], [], color=POLICY, linewidth=1.5, label="Policy only"),
        Line2D([], [], color=BEST, linewidth=1.5, linestyle=(0, (4, 2)), label="Best-known only"),
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="Customer"),
        Line2D([], [], marker="s", linestyle="", color=ACTIVITY["charging"], markersize=4.5, label="Charging stop"),
        Line2D([], [], marker="*", linestyle="", color=INK, markersize=8, label="Depot"),
        Line2D([], [], marker="X", linestyle="", color=MISSED, markersize=7, label="Missed by the policy"),
    ], loc="lower center", ncol=7, frameon=False, fontsize=9.5)
    figure.suptitle("The learned policy against the best-known solution, on seeds it never saw",
                    x=0.05, ha="left", fontsize=15, color=INK, weight="bold", y=0.985)
    figure.text(0.05, 0.955, "Passing run s2. Seeds chosen by rank of hours excess among 500 test seeds. Best-known is the best of a planner portfolio, not a proven optimum.",
                fontsize=10, color=SECONDARY)
    figure.subplots_adjust(left=0.05, right=0.985, top=0.905, bottom=0.04, wspace=0.05, hspace=0.17)
    figure.savefig(target, facecolor=SURFACE)
    plt.close(figure)


def drives(result: dict, region: Region) -> dict[int, list[tuple]]:
    """Each truck's drives ``(start, end, from, to)``: its moves, in order, on its routing intervals."""
    moves: dict[int, list] = {}
    for row in result["decisions"]:
        if row["kind"] in MOVES and row["target"] is not None:
            moves.setdefault(int(row["truck"]), []).append((int(row["node"]), int(row["target"])))
    paths = {}
    for truck in result["trucks"]:
        intervals = [row for row in truck["timeline"] if row["activity"] == "routing"]
        paths[int(truck["id"])] = [
            (float(row["start_time"]), float(row["end_time"]), region.at(a), region.at(b))
            for row, (a, b) in zip(intervals, moves.get(int(truck["id"]), []), strict=False)
            if region.at(a) and region.at(b)
        ]
    return paths


def _positions(paths: dict, clock: float) -> list[tuple[float, float]]:
    """Each working truck's position at ``clock``."""
    places = []
    for path in paths.values():
        moving = [row for row in path if row[2] != row[3]]
        if not moving:
            continue
        here = moving[0][2]
        for start, end, a, b in moving:
            if clock < start:
                break
            share = min(1.0, (clock - start) / max(end - start, 1e-9))
            here = (a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share)
        places.append(here)
    return places


def animate(trace: dict, region: Region, target: Path, frames: int = 120) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.8), dpi=80, facecolor=SURFACE)
    policy, best = trace["policy"], trace["bks"]
    end = max(policy["clock"], best["clock"])
    everything = legs(policy) | legs(best)
    points = [region.at(node) for pair in everything for node in pair if region.at(node)]
    artists = []
    for ax, result, name, colour in ((axes[0], best, "Best-known", BEST), (axes[1], policy, "Policy", POLICY)):
        ax.set_facecolor(SURFACE)
        ax.add_collection(LineCollection(region.roads, colors=ROAD, linewidths=0.3, zorder=1))
        ax.add_collection(LineCollection(_segments(region, legs(result)), colors=colour, linewidths=0.9, alpha=0.35, zorder=2))
        _frame(region, ax, points)
        dots = ax.scatter([], [], s=46, facecolor=colour, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        ax.set_title(name, loc="left", fontsize=12, color=INK)
        artists.append((dots, drives(result, region)))
    clock_text = figure.text(0.5, 0.03, "", ha="center", fontsize=12, color=INK)
    figure.suptitle(headline(trace), x=0.03, ha="left", fontsize=12, color=INK, weight="bold")
    figure.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.08, wspace=0.04)
    writer = PillowWriter(fps=12)
    with writer.saving(figure, str(target), dpi=80):
        for frame in range(frames):
            clock = end * frame / (frames - 1)
            for dots, paths in artists:
                places = _positions(paths, clock)
                dots.set_offsets(np.array(places) if places else np.empty((0, 2)))
            clock_text.set_text(f"Hour {clock:.0f}")
            writer.grab_frame(facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--animate", nargs="*", default=[], help="Trace names to animate.")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    traces = [json.loads(path.read_text()) | {"name": path.stem} for path in sorted(arguments.traces.glob("*.json"))]
    regions = {problem: Region(problem) for problem in {trace["case"]["problem"] for trace in traces}}
    for trace in traces:
        episode_figure(trace, regions[trace["case"]["problem"]], arguments.output / f"{trace['name']}.png")
        print(trace["name"], flush=True)
    if {trace["case"]["problem"] for trace in traces} >= {"tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos"}:
        overview(traces, regions, arguments.output / "overview_static.png")
    for name in arguments.animate:
        trace = next(trace for trace in traces if trace["name"] == name)
        animate(trace, regions[trace["case"]["problem"]], arguments.output / f"{name}.gif")


if __name__ == "__main__":
    main()
