"""Compare the California data with the generated case studies (asymmetry, leg sizes, scale).

    python scripts/casestudies/compare_case_studies.py   # writes data/case_studies/comparison.png
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "EVRoutingEnv" / "data"


def matrix_from_dicts(energy_path: Path, time_path: Path):
    def load(p):
        d = {}
        for k, v in json.load(open(p)).items():
            u, w = map(int, re.findall(r"-?\d+", k))
            d[(u, w)] = v
        return d

    e, t = load(energy_path), load(time_path)
    ids = sorted({u for u, _ in e} | {v for _, v in e})
    ix = {n: i for i, n in enumerate(ids)}
    E = np.full((len(ids), len(ids)), np.nan); T = E.copy()
    for (u, v), x in e.items():
        E[ix[u], ix[v]] = x
    for (u, v), x in t.items():
        T[ix[u], ix[v]] = x
    return E, T


def rel_asym(M, floor):
    iu = np.triu_indices(len(M), 1)
    f, r = M[iu], M.T[iu]
    ok = np.isfinite(f) & np.isfinite(r) & (f > floor) & (r > floor)
    return np.abs(f[ok] - r[ok]) / np.maximum(f[ok], r[ok]) * 100


def main():
    sets = {}
    meta = {}
    folders = sorted((DATA / "case_studies").glob("*/metadata.json"), key=lambda q: (q.parent.name != "california", q.parent.name))
    for p in folders:  # California first
        m = json.load(open(p))
        meta[m["name"]] = m
        who = "original data" if not m.get("generated", True) else m["vehicle_profile"].replace("_", " ")
        sets[f"{m['title']} ({who}, {m['battery_kwh']:.0f} kWh)"] = matrix_from_dicts(
            p.parent / "shortest_path_energy_dict.json", p.parent / "shortest_path_time_dict.json")
    cols = ["#444444", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    for (name, (E, T)), c in zip(sets.items(), cols):
        a = rel_asym(E, 1.0)
        axs[0].hist(a, bins=np.linspace(0, 60, 61), histtype="step", lw=2, color=c, density=True,
                    label=f"{name}: median {np.median(a):.1f} %, >5 %: {np.mean(a > 5) * 100:.0f} %")
        a = rel_asym(T, 0.05)
        axs[1].hist(a, bins=np.linspace(0, 40, 61), histtype="step", lw=2, color=c, density=True,
                    label=f"median {np.median(a):.1f} %, >5 %: {np.mean(a > 5) * 100:.0f} %")
        off = E[np.isfinite(E) & (E > 0)]
        axs[2].hist(off / np.median(off), bins=np.linspace(0, 4, 81), histtype="step", lw=2, color=c, density=True)
    axs[0].set_title("Energy asymmetry |E(i→j) − E(j→i)| / max"); axs[0].set_xlabel("%"); axs[0].legend(fontsize=7.5)
    axs[1].set_title("Travel-time asymmetry"); axs[1].set_xlabel("%"); axs[1].legend(fontsize=7.5)
    axs[2].set_title("Leg energy relative to the case's median leg"); axs[2].set_xlabel("E / median(E)")
    for a in axs:
        a.set_ylabel("density")
    fig.suptitle("Case studies vs California: asymmetry and leg-size distributions", fontsize=12)
    out = DATA / "case_studies" / "comparison.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
