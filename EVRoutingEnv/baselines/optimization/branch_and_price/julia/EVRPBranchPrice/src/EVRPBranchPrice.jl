"""
Exact branch-and-price for the nominal joint electric-truck routing problem of
`EventDrivenTruckEnv`, adapted from ElectricRouting.jl (Jacquillat & Lo,
arXiv:2407.02640): path-based set-partitioning master, label-setting pricing
over the env's discrete target-SoC charging, ng-route relaxation, and a
branch-and-bound tree on route counts and customer successions so that the
reported value is a proven optimum (or a certified bound at the time limit).

Input is the `evrp-bp-instance/v1` JSON written by
`EVRoutingEnv/baselines/optimization/branch_and_price/export.py`; output is the
`evrp-bp-result/v1` certificate.
"""
module EVRPBranchPrice

using DataStructures
using JSON3
using JuMP
using Gurobi
using HiGHS
using Printf

include("nodeset.jl")
include("instance.jl")
include("routes.jl")
include("cuts.jl")
include("pricing.jl")
include("bidirectional.jl")
include("backend.jl")
include("master.jl")
include("bnp.jl")
include("output.jl")

export read_instance, BPParameters, solve, write_result, Backend

end
