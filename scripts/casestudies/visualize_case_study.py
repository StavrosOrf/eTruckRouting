"""Figures for a case-study folder (EVRoutingEnv/data/case_studies/<name>/).

    python scripts/casestudies/visualize_case_study.py EVRoutingEnv/data/case_studies/seattle

Writes <case>/figures/{overview,terrain,asymmetry}.png using only files inside the folder.

Matrix case studies (europe_10k, od_energy_kwh.npy) have 10^8 node pairs: the asymmetry figure then uses a random
sample of `--sample-nodes` nodes (default 2500) and, when given `--dist-km <od_dist_km.npy>` (written next to the OD
shards in the raw cache), their true route lengths instead of straight-line distances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LightSource, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROLE_STYLE = {
    "customer": dict(c="#2c7fb8", s=9, marker="o", label="Delivery stop", zorder=5, lw=0),
    "depot": dict(c="#d95f02", s=170, marker="*", label="Depot", zorder=7, edgecolors="black", lw=0.8),
}


def sample_matrix_od(case: Path, meta: dict, n_sample: int, dist_km: Path | None, seed: int = 0) -> dict:
    """energy/time (and route length) between a random subset of nodes of a matrix case study."""
    ids = np.load(case / "od_node_ids.npy")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ids), min(n_sample, len(ids)), replace=False))
    sub = lambda f: np.asarray(np.load(f, mmap_mode="r")[pick][:, pick], dtype=np.float32)
    E, T = sub(case / meta["files"]["energy"]), sub(case / meta["files"]["time"])
    K = sub(dist_km) if dist_km is not None and Path(dist_km).exists() else np.full(E.shape, np.nan, np.float32)
    return {"node_ids": ids[pick], "energy_net_kwh": E, "time_h": T, "dist_km": K}


def load_case(case: Path, n_sample: int = 2500, dist_km: Path | None = None):
    d = {"meta": json.load(open(case / "metadata.json")),
         "nodes": json.load(open(case / "vis_data" / "nodes.json")),
         "stations": json.load(open(case / "vis_data" / "stations.json")),
         "all_sites": json.load(open(case / "vis_data" / "all_dcfast_sites.json"))}
    r = np.load(case / "vis_data" / "roads.npz")
    d["roads"] = {k: r[k] for k in r.files}
    t = np.load(case / "vis_data" / "terrain.npz")
    d["terrain"] = {k: t[k] for k in t.files}
    if d["meta"].get("format") == "npy":
        d["od"] = sample_matrix_od(case, d["meta"], n_sample, dist_km)
    else:
        o = np.load(case / "vis_data" / "od_matrices.npz")
        d["od"] = {k: o[k] for k in o.files}
    return d


def _aspect(ax, lat):
    ax.set_aspect(1.0 / np.cos(np.radians(lat)))


def _segments(roads, sel=None):
    s = lambda k: roads[k].astype(float) / 1e5
    lat0, lon0, lat1, lon1 = s("lat0"), s("lon0"), s("lat1"), s("lon1")
    seg = np.stack([np.stack([lon0, lat0], 1), np.stack([lon1, lat1], 1)], 1)
    return seg if sel is None else seg[sel]


def _hillshade(ax, terr, alpha=1.0):
    z = terr["elev"].astype(float)
    ny, nx = z.shape
    lat = terr["lat0"] + np.arange(ny) * terr["dlat"]
    lon = terr["lon0"] + np.arange(nx) * terr["dlon"]
    dx = terr["dlon"] * 111320 * np.cos(np.radians(lat.mean()))
    dy = terr["dlat"] * 110540
    hs = LightSource(azdeg=315, altdeg=35).hillshade(np.where(z < 1, 0.0, z), vert_exag=6, dx=dx, dy=dy)
    g = 0.58 + 0.42 * hs
    rgb = np.dstack([g * 0.97, g * 0.98, g])
    rgb[z < 1] = (0.78, 0.87, 0.95)  # open water
    ax.imshow(rgb, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], alpha=alpha, zorder=0, interpolation="bilinear")
    ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1])


def _map_height(terr, width, max_h=12.0):
    """Figure height that keeps a lat/lon map at true aspect for a given axes width."""
    ny, nx = terr["elev"].shape
    h = width * (ny * terr["dlat"]) / (nx * terr["dlon"] * np.cos(np.radians(terr["lat0"] + ny * terr["dlat"] / 2)))
    return min(h, max_h)


def fig_overview(case: Path, d: dict):
    nodes, st, meta = d["nodes"], d["stations"], d["meta"]
    w = 8.0
    fig, ax = plt.subplots(figsize=(w + 1.6, _map_height(d["terrain"], w) + 1.2), layout="constrained")
    _hillshade(ax, d["terrain"])
    roads = d["roads"]
    major = roads["cls"] <= 3
    ax.add_collection(LineCollection(_segments(roads, ~major), colors="#9a9a9a", linewidths=0.2 if meta.get("scale_km", 100) > 300 else 0.25, alpha=0.7, zorder=1))
    ax.add_collection(LineCollection(_segments(roads, major), colors="#555555", linewidths=0.7, alpha=0.9, zorder=2))
    # unselected DC-fast sites for context
    other = [s for s in d["all_sites"] if not s["selected"]]
    if other:
        ax.scatter([s["lon"] for s in other], [s["lat"] for s in other], s=10, marker="^", c="none",
                   edgecolors="#c28e8e", lw=0.5, zorder=3)
    big = meta.get("scale_km", 100) > 300          # state-scale maps need smaller markers
    dense = len(nodes) > 3000                      # continent-scale cases: thousands of nodes per map
    k = 0.12 if dense else 0.35 if big else 1.0
    for role in ("customer", "depot"):
        pts = [n for n in nodes if n["role"] == role]
        style = dict(ROLE_STYLE[role]); style["s"] = style["s"] * (0.45 if dense and role == "customer" else 2.2 if big and role == "customer" else 0.8 if big else 1)
        ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts], **style)
    ports = np.array([s["ports"] for s in st], float)
    kinds = np.array([s.get("station_type", "DCFast") for s in st])
    sc = None
    for kind, marker in (("DCFast", "^"), ("Level2", "s")):
        m = kinds == kind
        if m.any():
            sc = ax.scatter(np.array([s["lon"] for s in st])[m], np.array([s["lat"] for s in st])[m],
                            s=(30 + 7 * np.minimum(ports[m], 40)) * k, c=ports[m], cmap="YlOrRd", vmin=0,
                            vmax=max(ports.max(), 1), marker=marker, edgecolors="black", lw=0.2 if dense else 0.5 if big else 0.8, zorder=6)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("ports at charger node" if (kinds != "DCFast").any() else "DC-fast ports at charger node")
    c = meta["counts"]
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", c="#2c7fb8", ms=5, label=f"Delivery stops ({c['customers']})"),
        Line2D([], [], marker="*", ls="", c="#d95f02", mec="black", ms=14, label=f"Depots ({c['depots']})"),
        Line2D([], [], marker="^", ls="", c="#fdae61", mec="black", ms=9, label=f"Charger nodes ({c['chargers']}, {c.get('charger_ports_total', c['dcfast_ports_in_nodes'])} ports"
                                                                                   + (", squares = Level 2" if (kinds != "DCFast").any() else "") + ")"),
    ] + ([Line2D([], [], marker="^", ls="", mfc="none", mec="#c28e8e", ms=6, label="DC-fast sites not used")] if other else []),
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, framealpha=0.9)
    ax.set_title(f"{meta['title']}\ndelivery stops, depots, DC-fast charger nodes", fontsize=11)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    _aspect(ax, np.mean(ax.get_ylim()))
    fig.savefig(case / "figures" / "overview.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_terrain(case: Path, d: dict):
    meta, roads = d["meta"], d["roads"]
    grade = roads["grade"].astype(float) * 100
    mh = _map_height(d["terrain"], 4.6, max_h=9.5)
    fig = plt.figure(figsize=(12.5, max(mh + 0.8, 6.0)), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[:, 0])
    _hillshade(ax, d["terrain"], alpha=0.5)
    lc = LineCollection(_segments(roads), array=np.abs(grade), cmap="magma_r", linewidths=0.5, zorder=2)
    lc.set_clim(0, 8)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, fraction=0.05, pad=0.01).set_label("|road grade| per segment (%)")
    ax.set_title("Road grade on the main network"); _aspect(ax, np.mean(ax.get_ylim()))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(grade[np.abs(grade) <= 15], bins=120, color="#4575b4")
    ax.set_yscale("log"); ax.set_xlabel("segment grade (%)"); ax.set_ylabel("segments (log)")
    ax.set_title(f"Grade distribution (median |g| = {np.median(np.abs(grade)):.2f} %, "
                 f"99th pct = {np.percentile(np.abs(grade), 99):.1f} %)", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    el = np.array([n["elev_m"] for n in d["nodes"]])
    role = np.array([n["role"] for n in d["nodes"]])
    bins = np.linspace(min(0, el.min()) if el.min() < 20 else el.min() - 5, el.max() + 1, 40)
    for r, col in (("customer", "#2c7fb8"), ("charger", "#e31a1c")):
        ax.hist(el[role == r], bins=bins, color=col, alpha=0.7, label=f"{r} nodes ({(role == r).sum()})")
    for e in el[role == "depot"]:
        ax.axvline(e, color="#d95f02", lw=2, label="depot" if e == el[role == "depot"][0] else None)
    ax.set_xlabel("node elevation (m)"); ax.set_ylabel("nodes"); ax.legend(fontsize=8)
    ax.set_title("Elevation of case-study nodes")
    fig.suptitle(f"{meta['title']}: terrain", fontsize=12)
    fig.savefig(case / "figures" / "terrain.png", dpi=150); plt.close(fig)


def fig_asymmetry(case: Path, d: dict):
    od, meta = d["od"], d["meta"]
    E, T, K = od["energy_net_kwh"].astype(float), od["time_h"].astype(float), od["dist_km"].astype(float)
    n = len(E)
    dist_label = "route length (km)"
    if np.isfinite(K[~np.eye(n, dtype=bool)]).mean() < 0.5:  # road length unknown for most pairs (California)
        pos = {nd["node"]: nd for nd in d["nodes"]}
        la = np.radians([pos[int(i)]["lat"] for i in od["node_ids"]]); lo = np.radians([pos[int(i)]["lon"] for i in od["node_ids"]])
        a = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None] * np.sin((lo[None] - lo[:, None]) / 2) ** 2
        K = 12742.0 * np.arcsin(np.sqrt(a)); dist_label = "straight-line distance (km)"
    E = np.where(np.isfinite(E), E, 0.0); T = np.where(np.isfinite(T), T, 0.0)
    iu = np.triu_indices(n, 1)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    ok = (E[iu] > 0.5) & (E.T[iu] > 0.5)
    hb = ax.hexbin(E[iu][ok], E.T[iu][ok], gridsize=60, bins="log", cmap="viridis", mincnt=1)
    m = max(E[iu][ok].max(), E.T[iu][ok].max())
    ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set_xlabel("energy i→j (kWh)"); ax.set_ylabel("energy j→i (kWh)"); ax.set_title("Energy: forward vs reverse")
    ax.set_aspect("equal")
    ax = axs[0, 1]
    ok = (T[iu] > 0.02) & (T.T[iu] > 0.02)
    ax.hexbin(T[iu][ok] * 60, T.T[iu][ok] * 60, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    m = max(T[iu][ok].max(), T.T[iu][ok].max()) * 60
    ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set_xlabel("time i→j (min)"); ax.set_ylabel("time j→i (min)"); ax.set_title("Travel time: forward vs reverse")
    ax.set_aspect("equal")
    ax = axs[0, 2]
    for name, M, floor, col in (("energy", E, 1.0, "#d95f02"), ("time", T, 0.05, "#1b9e77")):
        f, r = M[iu], M.T[iu]
        ok = (f > floor) & (r > floor)
        rel = np.abs(f[ok] - r[ok]) / np.maximum(f[ok], r[ok]) * 100
        ax.hist(rel, bins=np.linspace(0, 100, 51), alpha=0.65, color=col,
                label=f"{name}: median {np.median(rel):.1f} %, >5 %: {np.mean(rel > 5) * 100:.0f} %")
    ax.set_xlabel("|fwd − rev| / max (%)"); ax.set_ylabel("node pairs"); ax.legend(fontsize=9)
    ax.set_title("Asymmetry of node pairs")
    ax = axs[1, 0]
    el = {n_["node"]: n_["elev_m"] for n_ in d["nodes"]}
    h = np.array([el[int(i)] for i in od["node_ids"]])
    climb = (h[None, :] - h[:, None])[iu]
    diff = (E - E.T)[iu]
    ok = (K[iu] > 1) & (E[iu] > 0) & (E.T[iu] > 0)
    ax.scatter(climb[ok], diff[ok], s=2, alpha=0.15, c="#7570b3", rasterized=True)
    xs = np.linspace(climb.min(), climb.max(), 10)
    ax.plot(xs, np.polyval(np.polyfit(climb[ok], diff[ok], 1), xs), "k-", lw=1)
    ax.set_xlabel("elevation of j minus i (m)"); ax.set_ylabel("E(i→j) − E(j→i)  (kWh)")
    ax.set_title("Energy asymmetry follows net climb")
    ax = axs[1, 1]
    ok = (K > 1) & (E > 0)
    whkm = (E / np.maximum(K, 0.1))[ok] * 1000
    ax.hexbin(K[ok], whkm, gridsize=60, bins="log", cmap="plasma", mincnt=1,
              extent=(0, np.percentile(K[ok], 99.5), 0, max(900.0, 1.15 * np.percentile(whkm, 99.5))))
    ax.set_xlabel(dist_label); ax.set_ylabel("Wh per km"); ax.set_title("Consumption per directed pair")
    ax = axs[1, 2]
    ax.hist(E[~np.eye(n, dtype=bool)], bins=60, color="#7570b3")
    ax.set_xlabel("directed leg energy (kWh)"); ax.set_ylabel("directed pairs")
    r = meta["ranges"]["energy_kwh_p5_p50_p95_max"]
    ax.set_title(f"Leg energy: p5 {r[0]:.0f}, median {r[1]:.0f}, p95 {r[2]:.0f} kWh", fontsize=11)
    scope = f"a random sample of {n} of the {meta['counts']['nodes']}" if n < meta["counts"]["nodes"] else f"all {n}"
    fig.suptitle(f"{meta['title']}: direction-dependent energy and time between {scope} nodes", fontsize=12)
    fig.tight_layout(); fig.savefig(case / "figures" / "asymmetry.png", dpi=140); plt.close(fig)


def fig_scenarios(case: Path, d: dict):
    """Coverage and outcomes of the scenarios drawn by check_scenarios.py."""
    chk = json.load(open(case / "scenario_check.json"))
    sc = [x for x in chk["scenarios"] if "result" in x]
    if not sc:
        return
    meta, nodes = d["meta"], {n["node"]: n for n in d["nodes"]}
    use = {}
    for x in sc:
        for c in x["customers"]:
            use[c] = use.get(c, 0) + 1
    dep = {}
    for x in sc:
        dep[x["depot"]] = dep.get(x["depot"], 0) + 1
    w = 8.0
    fig = plt.figure(figsize=(15.5, max(_map_height(d["terrain"], w, 8.5) + 0.8, 6.0)), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[:, 0])
    _hillshade(ax, d["terrain"], alpha=0.8)
    major = d["roads"]["cls"] <= 3
    ax.add_collection(LineCollection(_segments(d["roads"], major), colors="#777777", linewidths=0.4, alpha=0.7, zorder=1))
    free = [n for n in d["nodes"] if n["role"] != "charger" and n["node"] not in use and n["node"] not in dep]
    ax.scatter([n["lon"] for n in free], [n["lat"] for n in free], s=6, c="#bbbbbb", zorder=2, label="never drawn")
    ids = list(use)
    sca = ax.scatter([nodes[i]["lon"] for i in ids], [nodes[i]["lat"] for i in ids], s=14, c=[use[i] for i in ids],
                     cmap="viridis", zorder=3, label="customers (colour = times drawn)")
    fig.colorbar(sca, ax=ax, fraction=0.04, pad=0.01).set_label(f"times drawn in {len(sc)} scenarios")
    ids = list(dep)
    ax.scatter([nodes[i]["lon"] for i in ids], [nodes[i]["lat"] for i in ids], s=[60 + 25 * dep[i] for i in ids], marker="*",
               c="#d95f02", edgecolors="black", lw=0.6, zorder=4, label="depots drawn")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{len(sc)} scenarios: {chk['distinct_depots']} distinct depots, {chk['distinct_customers_used']} distinct customers")
    _aspect(ax, np.mean(ax.get_ylim())); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    r = [x["result"] for x in sc]
    ok = np.array([bool(q["success"]) for q in r])
    dist = np.array([q["total_distance"] for q in r]); ch = np.array([q["charging_sessions"] for q in r])
    en = np.array([q["total_energy_consumed"] for q in r]); trucks = chk["trucks"]; batt = chk["battery_kwh"]
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(dist[ok], ch[ok], c="#1b9e77", label=f"success ({ok.sum()})", s=40)
    ax.scatter(dist[~ok], ch[~ok], c="#d95f02", marker="x", label=f"failed ({(~ok).sum()})", s=50)
    ax.set_xlabel("total distance driven (km)"); ax.set_ylabel("charging sessions"); ax.legend(fontsize=8)
    ax.set_title("Outcome of a heuristic policy per scenario")
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(en / (trucks * batt), bins=15, color="#7570b3")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"energy consumed per truck / battery ({batt:g} kWh)"); ax.set_ylabel("scenarios")
    ax.set_title(f"{chk['episodes_needing_charge']}/{len(sc)} scenarios needed charging (dashed: one full battery)")
    fig.suptitle(f"{meta['title']}: scenario generation check", fontsize=12)
    fig.savefig(case / "figures" / "scenarios.png", dpi=140); plt.close(fig)


def main(case: Path, n_sample: int = 2500, dist_km: Path | None = None):
    (case / "figures").mkdir(exist_ok=True)
    d = load_case(case, n_sample, dist_km)
    fig_overview(case, d); fig_terrain(case, d); fig_asymmetry(case, d)
    if (case / "scenario_check.json").exists():
        fig_scenarios(case, d)
    print("figures ->", case / "figures")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--sample-nodes", type=int, default=2500, help="matrix case studies: nodes sampled for the asymmetry figure")
    ap.add_argument("--dist-km", type=Path, help="matrix case studies: od_dist_km.npy of the build (route lengths)")
    a = ap.parse_args()
    main(a.case, a.sample_nodes, a.dist_km)
