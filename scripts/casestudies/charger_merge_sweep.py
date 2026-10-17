"""How many charger nodes remain when DC-fast stations are merged within 0.1 ... 10 km.

    python scripts/casestudies/charger_merge_sweep.py            # uses each case's metadata + AFDC cache

Reads the raw AFDC station lists (``--raw-dir``) for each generated case study and merges the public open
DC-fast stations inside the case's study area with single linkage (chains of nearby stations join) and
with complete linkage (every pair inside a node is within the radius). Ports are always preserved; the
table shows what is lost in position: ``displacement`` is the distance from a station to its node centre.
Writes ``EVRoutingEnv/data/case_studies/charger_merge_sweep.{json,png}`` and prints a markdown table.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargers as chg  # noqa: E402
import fetch_data  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "EVRoutingEnv" / "data" / "case_studies"
RADII_M = [0, 100, 500, 1000, 2000, 5000, 10000]


def stats(clusters, n_raw):
    ports = np.array([c["ports"] for c in clusters])
    disp = np.array([c.get("displacement_m", 0.0) for c in clusters])
    diam = np.array([c["diameter_m"] for c in clusters])
    return {
        "nodes": len(clusters), "reduction_pct": round(100 * (1 - len(clusters) / n_raw), 1),
        "median_ports": float(np.median(ports)), "max_ports": int(ports.max()),
        "max_diameter_km": round(float(diam.max()) / 1000, 1), "max_displacement_km": round(float(disp.max()) / 1000, 1),
        "merged_nodes": int(sum(c.get("n_stations", 1) > 1 for c in clusters)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=fetch_data.default_raw_dir())
    a = ap.parse_args()
    result = {}
    for meta_path in sorted(DATA.glob("*/metadata.json")):
        meta = json.load(open(meta_path))
        if not meta.get("generated", True):
            continue
        preset = meta["preset"]
        bbox = meta["study_bbox_min_lat_lon_max_lat_lon"]
        states = preset["state"] if isinstance(preset["state"], list) else [preset["state"]]
        sites = []
        for st in states:
            sites += chg.load_dcfast(fetch_data.ensure_afdc(a.raw_dir, st), bbox)
        n_raw = len(sites)
        rows = {"stations": n_raw, "ports": sum(s["ports"] for s in sites), "single": {}, "complete": {}}
        for method in ("single", "complete"):
            for r in RADII_M:
                clusters = chg.merge_sites(sites, r, method) if r else [
                    {"ports": s["ports"], "diameter_m": 0.0, "displacement_m": 0.0, "n_stations": 1} for s in sites]
                rows[method][r] = stats(clusters, n_raw)
        result[meta["name"]] = rows
        print(f"\n{meta['title']}: {n_raw} stations, {rows['ports']} ports")
        print("| radius | single: nodes (-%) | max ports | max diameter | complete: nodes (-%) | max ports | max displacement |")
        print("|---|---|---|---|---|---|---|")
        for r in RADII_M:
            s_, c_ = rows["single"][r], rows["complete"][r]
            print(f"| {'none' if r == 0 else (str(r) + ' m' if r < 1000 else str(r // 1000) + ' km')} | {s_['nodes']} (-{s_['reduction_pct']}%) | "
                  f"{s_['max_ports']} | {s_['max_diameter_km']} km | {c_['nodes']} (-{c_['reduction_pct']}%) | {c_['max_ports']} | {c_['max_displacement_km']} km |")
    json.dump(result, open(DATA / "charger_merge_sweep.json", "w"), indent=1)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    xs = [r / 1000 for r in RADII_M[1:]]
    for (name, rows), col in zip(result.items(), ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]):
        for method, ls in (("single", "-"), ("complete", "--")):
            axs[0].plot(xs, [rows[method][r]["nodes"] / rows["stations"] * 100 for r in RADII_M[1:]], ls, c=col, marker="o",
                        label=f"{name} ({method})" if True else None)
        axs[1].plot(xs, [rows["single"][r]["max_ports"] for r in RADII_M[1:]], "-", c=col, marker="o", label=name)
        axs[2].plot(xs, [rows["complete"][r]["max_displacement_km"] for r in RADII_M[1:]], "-", c=col, marker="o", label=name)
        axs[2].plot(xs, [rows["single"][r]["max_diameter_km"] for r in RADII_M[1:]], ":", c=col, marker="x")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("merge radius (km)"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("charger nodes (% of stations)"); axs[0].set_title("Nodes left after merging"); axs[0].legend(fontsize=7)
    axs[1].set_ylabel("ports in the biggest node"); axs[1].set_title("Port concentration (single linkage)"); axs[1].legend(fontsize=8)
    axs[2].set_yscale("log"); axs[2].set_ylabel("km"); axs[2].set_title("Position error: complete-linkage displacement (solid),\nsingle-linkage chain diameter (dotted)", fontsize=10)
    fig.savefig(DATA / "charger_merge_sweep.png", dpi=140)
    print("\nwrote", DATA / "charger_merge_sweep.png")


if __name__ == "__main__":
    main()
