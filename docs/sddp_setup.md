# Running the Julia SDDP baseline

This repo does not bundle Julia. Install it locally and add the required packages before running `EVRoutingEnv/baselines/sddp_vrp.jl`.

## 1) Install Julia and set `PATH`
1. Download a recent Julia (1.9+ recommended) from julialang.org for your OS.
2. Unpack/install and add `julia` to your `PATH`, e.g.:
   - Linux/macOS: `export PATH="$HOME/julia-1.x.y/bin:$PATH"`
   - Windows (PowerShell): `[Environment]::SetEnvironmentVariable("PATH", "$($env:PATH);C:\path\to\Julia-1.x.y\bin", "User")`
3. Verify: `julia --version`

## 2) Ensure Gurobi is installed and licensed
- Install Gurobi (matching your platform) and set `GUROBI_HOME`, `PATH`, and `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` as required by Gurobi.
- Verify from a shell: `grbgetkey` works and license is valid.

## 3) Install Julia packages
From the repo root:
```bash
julia --project -e 'using Pkg; Pkg.add(["JuMP","Gurobi","SDDP","JSON","DataFrames","Statistics","Random","Plots","CSV"])'
```
If you keep Julia packages in a shared depot, you can omit `--project`; otherwise this pins the environment to the repo.

## 4) Set Gurobi environment for Julia
In Julia REPL (once):
```julia
julia --project
using Gurobi
ENV["GUROBI_HOME"] = "/path/to/gurobi"   # adjust
ENV["GRB_LICENSE_FILE"] = "/path/to/gurobi.lic"  # adjust if needed
Gurobi.Env()  # should create an environment without errors
```

## 5) Run the baseline
From the repo root:
```bash
julia --project EVRoutingEnv/baselines/sddp_vrp.jl
```
The script reads data from `EVRoutingEnv/data/` JSON files; ensure they are present. If you want to test the simpler single-truck template, run:
```bash
julia --project EVRoutingEnv/baselines/sddp_ev_single_truck.jl
```

## Troubleshooting
- `julia: command not found`: ensure Julia bin is on `PATH`.
- Gurobi license errors: verify `GUROBI_HOME`, `GRB_LICENSE_FILE`, and that your license is valid.
- Package not found: rerun the `Pkg.add` command above with `--project`.
