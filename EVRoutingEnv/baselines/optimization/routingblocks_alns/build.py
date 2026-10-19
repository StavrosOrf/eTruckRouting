"""Compile the native route-pricing core for the interpreter running this file.

The solver runs in its own Python 3.11 environment (``scripts/setup_rb_env.sh``)
because the RoutingBlocks wheel has no 3.13 build. This module compiles
``native/evrp_core.cpp`` with ``g++`` and pybind11 into ``native/``, once per
interpreter, and rebuilds when the source is newer than the library.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native" / "evrp_core.cpp"


def library_path() -> Path:
    return HERE / "native" / f"evrp_core{sysconfig.get_config_var('EXT_SUFFIX')}"


def build(force: bool = False, openmp: bool = True) -> Path:
    target = library_path()
    if target.exists() and not force and target.stat().st_mtime >= SOURCE.stat().st_mtime:
        return target
    import pybind11

    command = [
        "g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-march=native",
        *(["-fopenmp"] if openmp else []),
        f"-I{pybind11.get_include()}", f"-I{sysconfig.get_paths()['include']}",
        str(SOURCE), "-o", str(target),
    ]
    subprocess.run(command, check=True)
    return target


if __name__ == "__main__":
    print(build(force="--force" in sys.argv))
