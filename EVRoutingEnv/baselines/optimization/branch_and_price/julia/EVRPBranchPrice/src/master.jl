# Set-partitioning restricted master problem.
#
#   min  sum_p travel_p z_p + M * artificials
#   s.t. sum_p a_ip z_p == 1          for every customer i   (dual pi_i)
#        lo <= sum_p z_p <= hi        route count            (duals kappa)
#        z >= 0
#
# Partitioning rather than covering is required here: the tables violate the
# triangle inequality and the environment refuses a second visit, so a
# covering solution could not be repaired by skipping a customer.
# Artificial columns keep every branch node's LP feasible; a node whose
# converged LP still uses them is infeasible.

mutable struct Master
    model::Model
    cover::Vector{ConstraintRef}
    fleet_max::ConstraintRef
    fleet_min::ConstraintRef
    artificial::Vector{VariableRef}
    z::Vector{VariableRef}
    routes::Vector{Route}
    succ::Vector{Vector{Tuple{Int,Int}}}
    elementary::BitVector # visits every customer at most once
    feasible::BitVector   # elementary and feasible with exact charging times
    ng_ok::BitVector      # false: not an ng-route under the grown neighbourhoods
    arc_ok::BitVector     # false: uses an arc removed by reduced-cost fixing
    phase_one::Bool
    blocked::BitVector
    known::Set{UInt64}
    cuts::Vector{CapacityCut}
    cut_rows::Vector{ConstraintRef}
    known_cuts::Set{BitVector}
    sr_cuts::Vector{SubsetRowCut}
    sr_rows::Vector{ConstraintRef}
    known_sr::Set{NTuple{3,Int}}
    big_m::Float64
end

function Master(inst::Instance, backend::Backend; threads::Int, big_m::Float64, method::Int)
    model = new_model(backend; threads = threads, method = method)
    n = inst.n
    art = @variable(model, [1:n+2], lower_bound = 0.0)
    @objective(model, Min, big_m * sum(art))
    cover = [@constraint(model, art[i] == 1.0) for i in 1:n]
    fleet_max = @constraint(model, -art[n+1] <= inst.fleet)
    fleet_min = @constraint(model, art[n+2] >= 0.0)
    Master(model, cover, fleet_max, fleet_min, art, VariableRef[], Route[],
           Vector{Tuple{Int,Int}}[], falses(0), falses(0), falses(0), falses(0), false, falses(0), Set{UInt64}(),
           CapacityCut[], ConstraintRef[], Set{BitVector}(),
           SubsetRowCut[], ConstraintRef[], Set{NTuple{3,Int}}(), big_m)
end

"""
Add `r` unless its node sequence is known. Columns sharing a node sequence
have the same cost and coefficients, so one per sequence suffices; it is
stored with an exactly feasible charging plan when one exists.
"""
function add_column!(mp::Master, inst::Instance, r::Route)
    key = hash(r.nodes)
    key in mp.known && return false
    push!(mp.known, key)
    scheduled = schedule_route(inst, r)
    feasible = scheduled !== nothing
    feasible && (r = scheduled)
    z = @variable(mp.model, lower_bound = 0.0)
    set_objective_coefficient(mp.model, z, mp.phase_one ? 0.0 : r.travel)
    counts = Dict{Int,Int}()
    for c in r.customers
        counts[c] = get(counts, c, 0) + 1
    end
    for (c, k) in counts
        set_normalized_coefficient(mp.cover[c], z, Float64(k))
    end
    set_normalized_coefficient(mp.fleet_max, z, 1.0)
    set_normalized_coefficient(mp.fleet_min, z, 1.0)
    succ = successions(inst, r)
    for (cut, row) in zip(mp.cuts, mp.cut_rows)
        e = entries(cut, succ, inst.n)
        e > 0 && set_normalized_coefficient(row, z, Float64(e))
    end
    for (cut, row) in zip(mp.sr_cuts, mp.sr_rows)
        e = sr_coefficient(cut, r.customers)
        e > 0 && set_normalized_coefficient(row, z, Float64(e))
    end
    push!(mp.z, z)
    push!(mp.routes, r)
    push!(mp.succ, succ)
    push!(mp.elementary, allunique(r.customers))
    push!(mp.feasible, feasible)
    push!(mp.ng_ok, true)
    push!(mp.arc_ok, all(inst.usable[r.nodes[t], r.nodes[t+1]] for t in 1:length(r.nodes)-1))
    push!(mp.blocked, false)
    true
end

"""
Block every column using a forbidden succession or no longer an ng-route;
with `exact_time`, also every elementary column that only the charging-time
relaxation admits. Non-elementary ng columns stay: they belong to every
phase's relaxation and can never enter an integer solution. Sets the
route-count range.
"""
function restrict!(mp::Master, forbidden::BitMatrix, lo::Int, hi::Int, exact_time::Bool)
    for p in eachindex(mp.z)
        bad = !mp.ng_ok[p] || !mp.arc_ok[p] || (exact_time && mp.elementary[p] && !mp.feasible[p]) ||
              any(forbidden[a, b] for (a, b) in mp.succ[p])
        if bad != mp.blocked[p]
            bad ? set_upper_bound(mp.z[p], 0.0) : delete_upper_bound(mp.z[p])
            mp.blocked[p] = bad
        end
    end
    set_normalized_rhs(mp.fleet_max, Float64(hi))
    set_normalized_rhs(mp.fleet_min, Float64(lo))
end

function solve_lp!(mp::Master)
    optimize!(mp.model)
    # Numerical trouble (HiGHS reports OTHER_ERROR now and then): solve once
    # more from scratch, then with the other algorithm.
    for attempt in 1:2
        termination_status(mp.model) == MOI.OPTIMAL && break
        retry_lp!(mp.model, attempt)
    end
    status = termination_status(mp.model)
    status == MOI.OPTIMAL || error("restricted master LP ended with status $(status)")
    objective_value(mp.model)
end

function retry_lp!(model::Model, attempt::Int)
    inner = backend(model)
    if inner isa HiGHS.Optimizer
        HiGHS.Highs_clearSolver(inner.inner)
        attempt == 2 && set_attribute(model, "solver", "ipm")
        optimize!(model)
        attempt == 2 && set_attribute(model, "solver", "choose")
    else
        method = get_attribute(model, "Method")
        set_attribute(model, "Method", attempt == 1 ? 0 : 2)    # primal simplex, then barrier
        optimize!(model)
        set_attribute(model, "Method", method)
    end
end

"""Cover duals, the combined route-count dual, capacity-cut and subset-row duals."""
function master_duals(mp::Master)
    π = Float64[dual(c) for c in mp.cover]
    κ = dual(mp.fleet_max) + dual(mp.fleet_min)
    σ = Float64[dual(c) for c in mp.cut_rows]
    τ = Float64[dual(c) for c in mp.sr_rows]
    π, κ, σ, τ
end

"""Add a globally valid subset-row cut (z = 0 satisfies it, so no artificial)."""
function add_sr_cut!(mp::Master, cut::SubsetRowCut)
    (cut.members in mp.known_sr || length(mp.sr_cuts) >= MAX_SUBSET_ROW) && return false
    push!(mp.known_sr, cut.members)
    # the whole row at once: coefficient by coefficient is slow in HiGHS
    terms = [(k, e) for (k, e) in ((k, sr_coefficient(cut, mp.routes[k].customers)) for k in eachindex(mp.z)) if e > 0]
    row = @constraint(mp.model, 0.0 * mp.artificial[1] + sum(Float64(e) * mp.z[k] for (k, e) in terms; init = 0.0) <= 1.0)
    push!(mp.sr_cuts, cut)
    push!(mp.sr_rows, row)
    true
end

"""Add a globally valid capacity cut (with its own artificial for infeasible nodes)."""
function add_cut!(mp::Master, inst::Instance, cut::CapacityCut)
    cut.members in mp.known_cuts && return false
    push!(mp.known_cuts, cut.members)
    art = @variable(mp.model, lower_bound = 0.0)
    set_objective_coefficient(mp.model, art, mp.phase_one ? 1.0 : mp.big_m)
    push!(mp.artificial, art)
    terms = [(k, e) for (k, e) in ((k, entries(cut, mp.succ[k], inst.n)) for k in eachindex(mp.z)) if e > 0]
    row = @constraint(mp.model, art + sum(Float64(e) * mp.z[k] for (k, e) in terms; init = 0.0) >= cut.rhs)
    push!(mp.cuts, cut)
    push!(mp.cut_rows, row)
    true
end

column_values(mp::Master) = isempty(mp.z) ? Float64[] : value.(mp.z)
artificial_mass(mp::Master) = sum(value.(mp.artificial))

"""
Switch between the real objective and phase one (minimise artificial use),
whose duals price Farkas columns: an infeasible node is proven so when no
column with negative phase-one reduced cost exists.
"""
function set_phase_one!(mp::Master, on::Bool, big_m::Float64)
    mp.phase_one == on && return
    for (k, z) in enumerate(mp.z)
        set_objective_coefficient(mp.model, z, on ? 0.0 : mp.routes[k].travel)
    end
    for a in mp.artificial
        set_objective_coefficient(mp.model, a, on ? 1.0 : big_m)
    end
    mp.phase_one = on
end
