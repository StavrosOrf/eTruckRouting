# Branch-and-price.
#
# Bounds. After an exact pricing round at a node with route-count bound `hi`,
#   LP + hi * min(0, min reduced cost)
# is a valid Lagrangian lower bound for that node; once pricing finds no
# negative column the node LP value itself is. Children inherit the parent's
# bound, and the global bound is the least bound over open nodes.
#
# Branching. (1) a fractional number of routes; (2) the most fractional
# customer succession y_ij (j directly after i among customers, the depot as
# both ends). Forbidding a succession only removes arcs of the pricing search,
# so no new dual is introduced. With every y_ij integral each customer has one
# successor, so every column in the support follows a fixed customer
# sequence; columns sharing a sequence differ only in their charging stops,
# and taking the cheapest of each is an integer solution no costlier than the
# LP. Hence the node is solved and the rule is complete.
#
# Two phases. Phase 1 searches the charging-time relaxation (see pricing.jl),
# whose optimum z_R is a valid lower bound. Integer solutions found there are
# re-timed exactly; if the relaxed optimum is truly feasible it is optimal,
# which is the usual outcome because charging rarely decides feasibility under
# a travel-time objective. Otherwise phase 2 repeats the search with
# time-aware pricing over truly feasible columns only, starting from z_R.

Base.@kwdef struct BPParameters
    time_limit::Float64 = 3600.0
    threads::Int = 1
    lp_method::Int = 1          # dual simplex: re-solves after branching and new columns warm-start
    ng_size::Int = 8
    ng_adapt_rounds::Int = 50
    cut_rounds::Int = 50
    cuts_per_round::Int = 50
    cut_violation::Float64 = 0.05
    cut_tailoff::Float64 = 1e-4
    subset_row_cuts::Int = 20      # per round, after capacity cuts are exhausted (0 = off)
    bucket_cap::Int = 64
    max_columns::Int = 300
    heuristic_pricing::Bool = true
    heuristic_neighbors::Int = 25
    relax_charging_time::Bool = true
    relax_horizon::Bool = true   # phase 1 without windows: drop the horizon too
    bidirectional::Bool = true   # bidirectional pricing in that (untimed) mode
    exact_fallback::Bool = true
    rc_tol::Float64 = 1e-6
    gap_tol::Float64 = 1e-6
    big_m::Float64 = 1e5
    capacity_unit::Float64 = 0.05
    mip_every::Int = 25
    strong_branching::Int = 8    # candidates scored by child LPs over current columns (0 = off)
    dive_time::Float64 = 300.0   # diving heuristic budget at the root (0 = off)
    dive_every::Int = 200        # repeat the dive every this many nodes (0 = root only)
    mip_time::Float64 = 60.0
    max_nodes::Int = typemax(Int)
    label_limit::Int = typemax(Int)
    verbose::Bool = true
end

struct BBNode
    id::Int
    depth::Int
    bound::Float64
    forbidden::Vector{Tuple{Int,Int}}
    lo::Int
    hi::Int
    label::String
end

mutable struct BPState{W}
    inst::Instance
    params::BPParameters
    env::Gurobi.Env
    master::Master
    pricing::PricingWorkspace{W}
    backward::BackwardSpace{W}
    start::Float64
    timefree::Bool
    untimed::Bool
    incumbent::Vector{Route}
    upper::Float64
    relaxed_upper::Float64
    relaxed_routes::Vector{Route}
    rounds::Int
    exact_rounds::Int
    pricing_seconds::Float64
    master_seconds::Float64
    log::Vector{Dict{String,Any}}
end

elapsed(s::BPState) = time() - s.start
timed_out(s::BPState) = elapsed(s) >= s.params.time_limit
tolerance(s::BPState, value::Float64) = s.params.gap_tol * max(1.0, abs(value))
# Value nodes are pruned against: integer solutions of the problem being searched.
prune_value(s::BPState) = s.timefree ? min(s.upper, s.relaxed_upper) : s.upper

function note!(s::BPState, msg::AbstractString)
    s.params.verbose && (println(@sprintf("[%8.1fs] ", elapsed(s)), msg); flush(stdout))
end

function covers_exactly(inst::Instance, routes::Vector{Route})
    count = zeros(Int, inst.n)
    for r in routes, c in r.customers
        count[c] += 1
    end
    all(==(1), count) && length(routes) <= inst.fleet
end

function improve!(s::BPState, routes::Vector{Route}, source::String)
    covers_exactly(s.inst, routes) || return false
    cost = sum(r.travel for r in routes; init = 0.0)
    cost < s.upper - 1e-9 || return false
    for r in routes
        ok, why, _ = simulate(s.inst, r)
        ok || error("internal: $(source) produced an infeasible route ($(why))")
    end
    s.upper = cost
    s.incumbent = routes
    note!(s, @sprintf("incumbent %.6f h from %s (%d routes)", cost, source, length(routes)))
    push!(s.log, Dict("t" => elapsed(s), "event" => "incumbent", "value" => cost, "source" => source))
    true
end

function improve_relaxed!(s::BPState, routes::Vector{Route}, source::String)
    covers_exactly(s.inst, routes) || return false
    cost = sum(r.travel for r in routes; init = 0.0)
    cost < s.relaxed_upper - 1e-9 || return false
    s.relaxed_upper = cost
    s.relaxed_routes = routes
    note!(s, @sprintf("relaxation incumbent %.6f h from %s", cost, source))
    true
end

"""Record an integer solution over master columns `cols` (true and relaxed incumbents)."""
function integer_solution!(s::BPState, cols::Vector{Int}, source::String)
    mp = s.master
    routes = [mp.routes[k] for k in cols]
    improve_relaxed!(s, routes, source)
    if all(mp.feasible[k] for k in cols)
        improve!(s, routes, source)
        return
    end
    # Same customer sequences with an exactly feasible charging plan, if generated.
    repaired = Route[]
    for k in cols
        if mp.feasible[k]
            push!(repaired, mp.routes[k])
            continue
        end
        options = [q for q in eachindex(mp.routes)
                   if mp.feasible[q] && mp.routes[q].customers == mp.routes[k].customers]
        isempty(options) && return
        push!(repaired, mp.routes[argmin(q -> mp.routes[q].travel, options)])
    end
    improve!(s, repaired, source * " (re-timed)")
end

"""One pricing round in the mode of the current phase."""
function price_round(s::BPState{W}, restr::Restrictions{W}, π, κ, σ, τ; exact::Bool, farkas::Bool,
                     time_limit::Float64) where {W}
    p, mp = s.params, s.master
    if s.timefree && s.untimed && p.bidirectional
        price_bidirectional!(s.pricing, s.backward, s.inst, π, κ, restr; exact = exact,
                             max_columns = p.max_columns, rc_tol = p.rc_tol, time_limit = time_limit,
                             label_limit = p.label_limit, bucket_cap = p.bucket_cap,
                             cost_scale = farkas ? 0.0 : 1.0, cuts = mp.cuts, σ = σ,
                             sr_cuts = mp.sr_cuts, τ = τ)
    else
        price!(s.pricing, s.inst, π, κ, restr; exact = exact, timefree = s.timefree,
               untimed = s.timefree && s.untimed,
               max_columns = p.max_columns, rc_tol = p.rc_tol, time_limit = time_limit,
               label_limit = p.label_limit, bucket_cap = p.bucket_cap,
               cost_scale = farkas ? 0.0 : 1.0, cuts = mp.cuts, σ = σ, sr_cuts = mp.sr_cuts, τ = τ)
    end
end

"""Successions forcing `i` to be directly followed by `j` among customers."""
function force_succession!(pairs::Vector{Tuple{Int,Int}}, inst::Instance, i::Int, j::Int)
    d = inst.depot
    if i != d
        for k in 1:inst.n+1
            (k != j && k != i) && push!(pairs, (i, k))
        end
    end
    if j != d
        for k in 1:inst.n+1
            (k != i && k != j) && push!(pairs, (k, j))
        end
    end
    pairs
end

"""
Diving heuristic: fix the largest fractional column (all its successions),
re-run heuristic column generation, repeat until the LP is integral or the
restriction becomes infeasible. Never certifies anything; only feeds the
incumbent.
"""
function dive!(s::BPState{W}, node::BBNode, budget::Float64) where {W}
    inst, p, mp = s.inst, s.params, s.master
    deadline = min(s.start + p.time_limit, time() + budget)
    forbidden = copy(node.forbidden)
    fixed = 0
    while time() < deadline
        restr = Restrictions{W}(inst, forbidden)
        restrict!(mp, restr.forbidden, node.lo, node.hi, !s.timefree)
        while time() < deadline
            solve_lp!(mp)
            artificial_mass(mp) > 1e-7 && (note!(s, "dive: infeasible after $(fixed) fixings"); return false)
            π, κ, σ, τ = master_duals(mp)
            res = price_round(s, restr, π, κ, σ, τ; exact = false, farkas = false,
                              time_limit = deadline - time())
            added = count(r -> add_column!(mp, inst, r), res.routes)
            added == 0 && break
            s.timefree || restrict!(mp, restr.forbidden, node.lo, node.hi, true)
        end
        time() < deadline || break
        solve_lp!(mp)
        z = column_values(mp)
        if all(v -> v < 1e-6 || v > 1 - 1e-6, z)
            before = s.upper
            integer_solution!(s, [k for k in eachindex(z) if z[k] > 0.5], "dive")
            note!(s, @sprintf("dive: integral after %d fixings (%.6f)", fixed, s.upper))
            return s.upper < before
        end
        pick, best = 0, 0.0
        for k in eachindex(z)
            if mp.elementary[k] && 1e-6 < z[k] < 1 - 1e-6 && z[k] > best
                pick, best = k, z[k]
            end
        end
        pick == 0 && return false
        for (a, b) in mp.succ[pick]
            force_succession!(forbidden, inst, a, b)
        end
        fixed += 1
    end
    false
end

"""
Column generation at one node. Returns (status, bound, values, lp). With
`certify = false` the node LP is only converged under heuristic pricing and
the returned bound is the node's inherited one.

While the restricted master needs artificial columns it is solved in phase
one and priced with Farkas duals (zero travel cost); a node whose exact
Farkas pricing finds nothing is infeasible. Pricing against big-M duals
instead would defeat every completion bound.
"""
function solve_node!(s::BPState{W}, node::BBNode; certify::Bool = true) where {W}
    inst, p, mp = s.inst, s.params, s.master
    bound = node.bound
    node.lo > node.hi && return (:infeasible, bound, Float64[], NaN)
    restr = Restrictions{W}(inst, node.forbidden)
    restrict!(mp, restr.forbidden, node.lo, node.hi, !s.timefree)
    heuristic = p.heuristic_pricing
    while true
        timed_out(s) && (set_phase_one!(mp, false, p.big_m); return (:timeout, bound, Float64[], NaN))
        t = time()
        lp = solve_lp!(mp)
        s.master_seconds += time() - t
        if !mp.phase_one && artificial_mass(mp) > 1e-7
            set_phase_one!(mp, true, p.big_m)
            continue
        elseif mp.phase_one && artificial_mass(mp) <= 1e-7
            set_phase_one!(mp, false, p.big_m)
            continue
        end
        farkas = mp.phase_one
        π, κ, σ, τ = master_duals(mp)
        exact = !heuristic
        res = price_round(s, restr, π, κ, σ, τ; exact = exact, farkas = farkas,
                          time_limit = p.time_limit - elapsed(s))
        s.pricing_seconds += res.seconds
        s.rounds += 1
        if res.exact && !farkas
            s.exact_rounds += 1
            bound = max(bound, lp + node.hi * min(0.0, res.min_rc))
        end
        if p.verbose && (s.rounds % 10 == 0 || isempty(res.routes))
            note!(s, @sprintf("node %d round %d %s %.6f bound %.6f rc %.4g cols %d (+%d) labels %d %s %.2fs",
                node.id, s.rounds, farkas ? "infeasibility" : "lp", lp, bound, res.min_rc, length(mp.z),
                length(res.routes), res.labels, exact ? "exact" : "heur", res.seconds))
        end
        cutoff = prune_value(s)
        if !farkas && bound >= cutoff - tolerance(s, cutoff)
            return (:pruned, bound, Float64[], NaN)
        end
        added = count(r -> add_column!(mp, inst, r), res.routes)
        if added > 0 && !s.timefree
            # Phase 2: block new elementary columns the exact schedule rejects.
            restrict!(mp, restr.forbidden, node.lo, node.hi, true)
        end
        if added == 0
            # Nothing new (an empty round, or columns already present whose
            # reduced cost is negative only within the LP tolerance).
            if !exact
                # Uncertified solves (root strengthening) stop at heuristic
                # convergence: good enough to separate cuts, not a bound.
                !certify && !farkas && return (:solved, bound, column_values(mp), lp)
                heuristic = false
                continue
            end
            if !res.exact
                set_phase_one!(mp, false, p.big_m)
                return (:timeout, bound, Float64[], NaN)
            end
            if farkas
                set_phase_one!(mp, false, p.big_m)
                return (:infeasible, bound, Float64[], NaN)
            end
            bound = max(bound, lp)
            return (:solved, bound, column_values(mp), lp)
        end
        heuristic = p.heuristic_pricing
    end
end

"""Re-mark columns that are no longer ng-routes after the neighbourhoods grew."""
function refresh_ng!(s::BPState)
    mp = s.master
    for k in eachindex(mp.routes)
        mp.ng_ok[k] = ng_feasible(s.pricing, s.inst, mp.routes[k])
    end
end

function succession_flows(mp::Master, z::Vector{Float64})
    y = Dict{Tuple{Int,Int},Float64}()
    for (k, v) in enumerate(z)
        v > 1e-9 || continue
        for pair in mp.succ[k]
            y[pair] = get(y, pair, 0.0) + v
        end
    end
    y
end

function cheapest_per_sequence(mp::Master, z::Vector{Float64})
    best = Dict{Vector{Int},Int}()
    for (k, v) in enumerate(z)
        v > 1e-9 || continue
        key = mp.routes[k].customers
        if !haskey(best, key) || mp.routes[k].travel < mp.routes[best[key]].travel
            best[key] = k
        end
    end
    collect(values(best))
end

"""Restricted-master MIP over the truly feasible elementary columns."""
function mip_heuristic!(s::BPState)
    inst, mp = s.inst, s.master
    cols = [k for k in eachindex(mp.routes) if mp.feasible[k]]
    isempty(cols) && return false
    budget = min(s.params.mip_time, s.params.time_limit - elapsed(s))
    budget > 1.0 || return false
    model = direct_model(Gurobi.Optimizer(s.env))
    set_silent(model)
    set_attribute(model, "Threads", s.params.threads)
    set_attribute(model, "TimeLimit", budget)
    x = @variable(model, [1:length(cols)], Bin)
    @objective(model, Min, sum(mp.routes[k].travel * x[i] for (i, k) in enumerate(cols)))
    members = [Int[] for _ in 1:inst.n]
    for (i, k) in enumerate(cols), c in mp.routes[k].customers
        push!(members[c], i)
    end
    any(isempty, members) && return false
    @constraint(model, [c in 1:inst.n], sum(x[i] for i in members[c]) == 1)
    @constraint(model, sum(x) <= inst.fleet)
    isfinite(s.upper) && set_attribute(model, "Cutoff", s.upper - 1e-7)
    optimize!(model)
    has_values(model) || return false
    chosen = [mp.routes[cols[i]] for i in eachindex(cols) if value(x[i]) > 0.5]
    improve!(s, chosen, "restricted-master MIP")
end

function branch_children(s::BPState{W}, node::BBNode, z::Vector{Float64}, next_id::Ref{Int}) where {W}
    inst, mp = s.inst, s.master
    routes = sum(z)
    child(bound, forbidden, lo, hi, lab) = BBNode((next_id[] += 1), node.depth + 1, bound, forbidden, lo, hi, lab)
    if abs(routes - round(routes)) > 1e-6
        f, c = floor(Int, routes), ceil(Int, routes)
        return [child(node.bound, node.forbidden, node.lo, min(node.hi, f), "routes<=$(f)"),
                child(node.bound, node.forbidden, max(node.lo, c), node.hi, "routes>=$(c)")]
    end
    y = succession_flows(mp, z)
    fractional = [(min(v - floor(v), ceil(v) - v), pair) for (pair, v) in y]
    filter!(f -> f[1] > 1e-6, fractional)
    isempty(fractional) && return BBNode[]
    sort!(fractional; by = f -> (-f[1], f[2]))
    branches(pair) = (force_succession!(copy(node.forbidden), inst, pair...), vcat(node.forbidden, [pair]))
    pick = fractional[1][2]
    k = min(s.params.strong_branching, length(fractional))
    if k > 1 && !timed_out(s)
        # Child LPs over the current columns: not bounds (pricing could still
        # lower them), only a ranking of how much each candidate moves the LP.
        parent = sum(mp.routes[q].travel * z[q] for q in eachindex(z))
        best = -Inf
        for (_, pair) in fractional[1:k]
            gains = map(branches(pair)) do forbidden
                restrict!(mp, Restrictions{W}(inst, forbidden).forbidden, node.lo, node.hi, !s.timefree)
                max(solve_lp!(mp) - parent, 1e-6)
            end
            score = gains[1] * gains[2]
            score > best && ((best, pick) = (score, pair))
        end
    end
    i, j = pick
    one_branch, zero_branch = branches(pick)
    d = inst.depot
    name(a) = a == d ? "D" : string(a)
    [child(node.bound, one_branch, node.lo, node.hi, "$(name(i))->$(name(j))=1"),
     child(node.bound, zero_branch, node.lo, node.hi, "$(name(i))->$(name(j))=0")]
end

"""
Root tightening: adaptive ng-routes, then rounded capacity cuts, then
subset-row cuts, each round on an LP converged only under heuristic pricing
(cuts and cycles found there are valid all the same). The bound is certified
by exact pricing once the cuts tail off; if that moves the LP materially the
rounds resume.
"""
function strengthen_root!(s::BPState, node::BBNode, result, nbound, z, lpvalue)
    inst, p, mp = s.inst, s.params, s.master
    resolve(certify) = solve_node!(s, BBNode(1, 0, nbound, node.forbidden, node.lo, node.hi, "root"); certify = certify)
    adapted = 0
    # :fast rounds use LPs converged under heuristic pricing and capacity cuts
    # only; :exact rounds certify every LP and add subset-row cuts once no
    # capacity cut is violated. Subset-row duals make heuristic pricing
    # unreliable, so they are only ever separated from certified LPs.
    phase = :fast
    for round in 1:p.cut_rounds
        if result != :solved
            note!(s, "cut loop: node $(result) at round $(round)")
            return (result, nbound, z, lpvalue)
        end
        certified = phase == :exact
        while adapted < p.ng_adapt_rounds && grow_neighbourhoods!(s.pricing, mp.routes, z)
            refresh_ng!(s)
            adapted += 1
            result, nbound, z, lpvalue = resolve(certified)
            result == :solved || return (result, nbound, z, lpvalue)
        end
        cuts = separate_capacity_cuts(inst, succession_flows(mp, z), mp.known_cuts;
                                      max_cuts = p.cuts_per_round, min_violation = p.cut_violation)
        rows = SubsetRowCut[]
        if isempty(cuts) && phase == :fast
            phase = :exact
            result, nbound, z, lpvalue = resolve(true)
            note!(s, @sprintf("root certified: lp %.6f, %d ng rounds", lpvalue, adapted))
            continue
        end
        if isempty(cuts) && p.subset_row_cuts > 0
            rows = separate_subset_row_cuts(inst, mp.routes, z, mp.known_sr;
                max_cuts = min(p.subset_row_cuts, MAX_SUBSET_ROW - length(mp.sr_cuts)),
                min_violation = p.cut_violation)
        end
        if isempty(cuts) && isempty(rows)
            note!(s, "cut loop: no violated cuts at round $(round)")
            break
        end
        foreach(c -> add_cut!(mp, inst, c), cuts)
        foreach(c -> add_sr_cut!(mp, c), rows)
        before = lpvalue
        result, nbound, z, lpvalue = resolve(certified)
        note!(s, @sprintf("cut round %d (%s): +%d capacity / +%d subset-row cuts (%d / %d total), lp %.6f",
            round, phase, length(cuts), length(rows), length(mp.cuts), length(mp.sr_cuts), lpvalue))
        if result == :solved && !(lpvalue - before > p.cut_tailoff * max(1.0, abs(before)))
            if phase == :fast
                phase = :exact
                result, nbound, z, lpvalue = resolve(true)
                note!(s, @sprintf("root certified: lp %.6f, %d ng rounds", lpvalue, adapted))
            else
                note!(s, @sprintf("cut loop: tail-off at round %d (%.3g)", round, lpvalue - before))
                break
            end
        end
    end
    if result == :solved && phase == :fast
        result, nbound, z, lpvalue = resolve(true)
    end
    (result, nbound, z, lpvalue)
end

"""Best-bound tree search. Returns (closed, lower bound over unexplored nodes, explored, open)."""
function search!(s::BPState, floor_bound::Float64)
    inst, p, mp = s.inst, s.params, s.master
    open_nodes = BinaryMinHeap{Tuple{Float64,Int,Int}}()
    store = Dict{Int,BBNode}()
    next_id = Ref(1)
    # Every route carries at most `payload`, so at least this many are needed.
    least = ceil(Int, sum(inst.demand) / (inst.payload + EPS) - 1e-12)
    root = BBNode(1, 0, floor_bound, Tuple{Int,Int}[], least, inst.fleet, "root")
    store[1] = root
    push!(open_nodes, (root.bound, 0, 1))
    explored = 0
    pending = Inf
    closed = true
    root_lp, root_bound = NaN, -Inf
    while !isempty(open_nodes)
        bound, _, id = pop!(open_nodes)
        node = pop!(store, id)
        cutoff = prune_value(s)
        bound >= cutoff - tolerance(s, cutoff) && continue
        if timed_out(s) || explored >= p.max_nodes
            pending = min(pending, bound)
            closed = false
            break
        end
        explored += 1
        result, nbound, z, lpvalue = solve_node!(s, node; certify = id != 1)
        if id == 1
            result, nbound, z, lpvalue = strengthen_root!(s, node, result, nbound, z, lpvalue)
            root_bound = nbound
            if p.dive_time > 0 && result == :solved
                cutoff = prune_value(s)
                nbound < cutoff - tolerance(s, cutoff) && dive!(s, node, p.dive_time)
                # the dive left the master restricted; re-solve the root as it was
                result, nbound, z, lpvalue = solve_node!(s, BBNode(1, 0, nbound, node.forbidden, node.lo, node.hi, "root"))
            end
            root_lp = lpvalue
            note!(s, @sprintf("%s root: lp %.6f bound %.6f, %d columns",
                s.timefree ? "relaxed" : "exact", root_lp, root_bound, length(mp.z)))
            mip_heuristic!(s)
        end
        if result == :timeout
            pending = min(pending, max(node.bound, nbound))
            closed = false
            break
        end
        result in (:pruned, :infeasible) && continue
        if all(v -> v < 1e-6 || v > 1 - 1e-6, z)
            integer_solution!(s, [k for k in eachindex(z) if z[k] > 0.5], "integral LP at node $(id)")
            continue
        end
        here = BBNode(node.id, node.depth, nbound, node.forbidden, node.lo, node.hi, node.label)
        children = branch_children(s, here, z, next_id)
        if isempty(children)
            integer_solution!(s, cheapest_per_sequence(mp, z), "integral successions at node $(id)")
            continue
        end
        for c in children
            store[c.id] = c
            push!(open_nodes, (c.bound, -c.depth, c.id))
        end
        explored % p.mip_every == 0 && mip_heuristic!(s)
        if p.dive_every > 0 && p.dive_time > 0 && explored % p.dive_every == 0
            dive!(s, here, p.dive_time / 2)
        end
        if p.verbose && explored % 10 == 0
            lb = min(pending, first(open_nodes)[1])
            note!(s, @sprintf("tree: %d explored, %d open, bound %.6f, incumbent %.6f (relaxed %.6f)",
                explored, length(open_nodes), lb, s.upper, s.relaxed_upper))
        end
    end
    remaining = [b for (b, _, _) in open_nodes.valtree]
    lower = min(pending, isempty(remaining) ? Inf : minimum(remaining))
    (closed, lower, explored, length(remaining), root_lp, root_bound)
end

struct BPResult
    status::String
    upper::Float64
    lower::Float64
    routes::Vector{Route}
    relaxed_optimum::Float64
    phases::Int
    root_lp::Float64
    root_bound::Float64
    nodes::Int
    open::Int
    columns::Int
    rounds::Int
    exact_rounds::Int
    seconds::Float64
    pricing_seconds::Float64
    master_seconds::Float64
    log::Vector{Dict{String,Any}}
end

function solve(inst::Instance; params::BPParameters = BPParameters(), env::Gurobi.Env = Gurobi.Env())
    W = words_for(inst.n + 1)
    _solve(Val(W), inst, params, env)
end

function _solve(::Val{W}, inst::Instance, params::BPParameters, env::Gurobi.Env) where {W}
    start = time()
    mp = Master(inst, env; threads = params.threads, big_m = params.big_m, method = params.lp_method)
    ws = PricingWorkspace{W}(inst; ng_size = params.ng_size, unit = params.capacity_unit,
                             near = params.heuristic_neighbors)
    windows = any(isfinite, inst.tw_close[1:inst.n]) || any(>(0), inst.tw_open[1:inst.n])
    s = BPState{W}(inst, params, env, mp, ws, BackwardSpace{W}(inst, ws.R), start, params.relax_charging_time,
                   params.relax_horizon && !windows, Route[], Inf, Inf,
                   Route[], 0, 0, 0.0, 0.0, Dict{String,Any}[])

    seeded = Route[]
    for steps in inst.initial_routes
        r = route_from_witness(inst, steps)
        r === nothing && continue
        ok, why, _ = simulate(inst, r)
        if ok
            add_column!(mp, inst, r)
            push!(seeded, r)
        else
            note!(s, "dropping an initial route the nominal rules reject: $(why)")
        end
    end
    improve!(s, seeded, "warm-start witness")
    singles = singleton_routes(inst)
    foreach(r -> add_column!(mp, inst, r), singles)
    note!(s, @sprintf("%d customers, %d chargers, %d trucks, %d seed columns",
        inst.n, inst.m, inst.fleet, length(mp.z)))

    closed, lower, explored, open_count, root_lp, root_bound = search!(s, -Inf)
    relaxed_optimum = NaN
    phases = 1
    if s.timefree
        if closed
            relaxed_optimum = min(s.upper, s.relaxed_upper)
            lower = relaxed_optimum
            note!(s, @sprintf("relaxation solved: %.6f (true incumbent %.6f)", relaxed_optimum, s.upper))
        else
            lower = min(lower, s.upper, s.relaxed_upper)
        end
        if closed && s.upper > lower + tolerance(s, lower) && params.exact_fallback && !timed_out(s)
            note!(s, "relaxed optimum needs charging time: exact time-aware search from bound $(lower)")
            s.timefree = false
            phases = 2
            closed, lower2, explored2, open_count, _, _ = search!(s, lower)
            explored += explored2
            lower = max(lower, min(lower2, s.upper))
        end
    else
        lower = min(lower, s.upper)
    end

    # A closed search without any integer solution proves infeasibility; in
    # phase 1 that is infeasibility of the relaxation, hence of the problem.
    status = if isempty(s.incumbent)
        closed && (!s.timefree || !isfinite(s.relaxed_upper)) ? "INFEASIBLE" : "NO_INCUMBENT"
    elseif s.upper <= lower + tolerance(s, lower) || (closed && !s.timefree)
        "OPTIMAL"
    else
        timed_out(s) ? "TIME_LIMIT" : "GAP"
    end
    status == "OPTIMAL" && (lower = s.upper)
    note!(s, @sprintf("%s: incumbent %.6f, bound %.6f, %d nodes, %d open", status, s.upper, lower,
        explored, open_count))
    BPResult(status, s.upper, lower, s.incumbent, relaxed_optimum, phases, root_lp, root_bound,
             explored, open_count, length(mp.z), s.rounds, s.exact_rounds, elapsed(s),
             s.pricing_seconds, s.master_seconds, s.log)
end
