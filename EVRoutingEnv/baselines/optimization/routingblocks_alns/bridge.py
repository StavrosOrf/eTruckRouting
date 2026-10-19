"""Run the RoutingBlocks baseline on a freshly reset environment.

The project interpreter (Python 3.13) exports the nominal instance and launches
``run_solver`` in the solver interpreter (Python 3.11 with ``routingblocks``,
see ``scripts/setup_rb_env.sh``) as a subprocess, exactly as the Julia
branch-and-price baseline does. The certificate it returns has the same route
records as ``evrp-bp-result/v1``, so ``BPSolution`` and the replay validators
are shared.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.export import write_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import BPSolution, solution_from_certificate

MODULE = "EVRoutingEnv.baselines.optimization.routingblocks_alns.run_solver"
REPO_ROOT = Path(__file__).resolve().parents[4]


def solver_python() -> Path:
    """The interpreter that has routingblocks: ``$EVRP_RB_PYTHON``, else the nearest ``.venv-rb``."""
    override = os.environ.get("EVRP_RB_PYTHON")
    if override:
        return Path(override)
    for root in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = root / ".venv-rb" / "bin" / "python"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no solver interpreter: run scripts/setup_rb_env.sh or set EVRP_RB_PYTHON"
    )


@dataclass(frozen=True)
class RBParameters:
    """Solver settings; ``extra`` passes any other ``SearchParameters`` field."""

    time_limit_seconds: float = 60.0
    workers: int = 1
    seed: int = 0
    objective: str = "travel"
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    def cli(self) -> list[str]:
        args = ["--time-limit", str(float(self.time_limit_seconds)), "--workers", str(int(self.workers)),
                "--seed", str(int(self.seed)), "--objective", self.objective]
        if self.verbose:
            args.append("--verbose")
        for key, value in self.extra.items():
            args += ["--param", str(key), str(value)]
        return args

    def as_dict(self) -> dict:
        return asdict(self)


def run_solver(instance_path: Path, result_path: Path, parameters: RBParameters,
               log_path: Path | None = None, timeout: float | None = None) -> dict:
    command = [str(solver_python()), "-m", MODULE, "--instance", str(instance_path), "--out", str(result_path),
               *parameters.cli()]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("OMP_NUM_THREADS", "1")
    with open(log_path or os.devnull, "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return json.loads(Path(result_path).read_text())


def solve_rb_plan(env, parameters: RBParameters | None = None, *, workdir: str | Path | None = None,
                  warm_start=None, sequences=None) -> BPSolution:
    """Export ``env`` (just after ``reset``), solve, and return the certificate as a ``BPSolution``."""
    parameters = parameters or RBParameters()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=workdir) as scratch:
        scratch = Path(scratch)
        write_instance(env, scratch / "instance.json", objective=parameters.objective,
                       warm_start=warm_start, sequences=sequences)
        certificate = run_solver(
            scratch / "instance.json", scratch / "result.json", parameters,
            log_path=scratch / "solver.log" if workdir is None else Path(workdir) / "solver.log",
            timeout=parameters.time_limit_seconds * 3 + 900,
        )
    return solution_from_certificate(certificate, time.perf_counter() - started)
