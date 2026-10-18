"""Run the Julia branch-and-price solver on a freshly reset environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.export import write_instance

RUNNER = Path(__file__).resolve().parent / "julia" / "run_bp.sh"


@dataclass(frozen=True)
class BPParameters:
    """Solver settings; ``extra`` passes any other ``BPParameters`` field of the Julia side."""

    time_limit_seconds: float = 3600.0
    threads: int = 1
    ng_size: int = 8
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    def cli(self) -> list[str]:
        values = {
            "time_limit": float(self.time_limit_seconds),
            "threads": int(self.threads),
            "ng_size": int(self.ng_size),
            "verbose": bool(self.verbose),
            **self.extra,
        }
        args = []
        for key, value in values.items():
            args += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
        return args

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BPSolution:
    """Certificate of one solve; ``routes`` are environment witness steps per used truck."""

    status: str
    objective_hours: float | None
    best_bound_hours: float | None
    gap: float | None
    wall_seconds: float
    routes: list[tuple[tuple[int, float, bool], ...]]
    certificate: dict

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"


def run_solver(instance_path: Path, result_path: Path, parameters: BPParameters,
               log_path: Path | None = None, timeout: float | None = None) -> dict:
    command = [str(RUNNER), str(instance_path), str(result_path), *parameters.cli()]
    env = dict(os.environ)
    env.setdefault("EVRP_BP_THREADS", str(parameters.threads))
    with open(log_path or os.devnull, "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                       timeout=timeout)
    return json.loads(Path(result_path).read_text())


def solution_from_certificate(certificate: dict, wall_seconds: float) -> BPSolution:
    routes = [
        tuple((int(node), float(value), bool(charge)) for node, value, charge in route["witness"])
        for route in certificate["routes"]
    ]
    return BPSolution(
        status=certificate["status"],
        objective_hours=certificate["objective_hours"],
        best_bound_hours=certificate["best_bound_hours"],
        gap=certificate["gap"],
        wall_seconds=wall_seconds,
        routes=routes,
        certificate=certificate,
    )


def solve_bp_plan(env, parameters: BPParameters | None = None, *, workdir: str | Path | None = None,
                  warm_start=None) -> BPSolution:
    """Export ``env`` (just after ``reset``), solve, and return the certificate."""
    parameters = parameters or BPParameters()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=workdir) as scratch:
        scratch = Path(scratch)
        write_instance(env, scratch / "instance.json", warm_start=warm_start)
        certificate = run_solver(
            scratch / "instance.json", scratch / "result.json", parameters,
            log_path=scratch / "solver.log" if workdir is None else Path(workdir) / "solver.log",
            timeout=parameters.time_limit_seconds * 2 + 600,
        )
    return solution_from_certificate(certificate, time.perf_counter() - started)
