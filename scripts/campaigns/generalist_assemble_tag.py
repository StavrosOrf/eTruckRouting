"""Assemble one evaluation tag from per-problem source tags (oob_local).

A deployed generalist policy may fix a different inference configuration per
problem (decode temperature, planner repair, none); each configuration's rows
are evaluated into its own tag by ``generalist_inference_eval.py``. This tool
assembles the declared per-problem choice into one tag the near-optimal gate
reads as a single run, and records the provenance and the per-problem wrapper
parameters and latencies in the sidecar. Rows are copied verbatim: no seed is
re-picked, dropped, or mixed within a problem.

    PYTHONPATH=. python scripts/campaigns/generalist_assemble_tag.py \
        --campaign results/generalist_graphppo/oob_v1 --split validation \
        --out-tag 8M_assembled --run oob_privcritic_s0 \
        --use tx_longhaul=8M_repair --use tx_longhaul_congested=8M_repair \
        --use pnw_longhaul_hos=8M_repair --use ca_sameday=8M_deca \
        --use ca_drayage_combined=8M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBLEMS = ("ca_sameday", "ca_drayage_combined", "tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--use", action="append", required=True, metavar="PROBLEM=TAG")
    arguments = parser.parse_args()

    sources = dict(item.split("=", 1) for item in arguments.use)
    missing = sorted(set(PROBLEMS) - set(sources))
    if missing:
        raise SystemExit(f"every problem needs a source tag; missing {missing}")
    base = arguments.campaign / "evaluation" / arguments.split
    rows, provenance = [], {}
    for problem, tag in sources.items():
        source = base / tag / f"{arguments.run}.json"
        if not source.is_file():
            raise SystemExit(f"{source} does not exist")
        chosen = [row for row in json.loads(source.read_text()) if row["problem"] == problem]
        if not chosen:
            raise SystemExit(f"{source} holds no rows for {problem}")
        seeds = [row["seed"] for row in chosen]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"{source} repeats seeds for {problem}")
        side = base / tag / "wrappers" / f"{arguments.run}.json"
        sidecar = json.loads(side.read_text()) if side.is_file() else None
        wrapper = (sidecar or {}).get("problems", {}).get(problem, {"wrapper": "base"})
        provenance[problem] = {"tag": tag, "seeds": len(seeds), **wrapper}
        rows.extend(chosen)

    out = base / arguments.out_tag
    (out / "wrappers").mkdir(parents=True, exist_ok=True)
    (out / f"{arguments.run}.json").write_text(json.dumps(rows))
    (out / "wrappers" / f"{arguments.run}.json").write_text(json.dumps(
        {"assembled_from": provenance, "run": arguments.run, "split": arguments.split}, indent=1,
    ))
    counts = {problem: provenance[problem]["seeds"] for problem in PROBLEMS}
    print(f"{out} assembled: {counts}")


if __name__ == "__main__":
    main()
