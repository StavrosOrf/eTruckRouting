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
    lp_solver::String = "gurobi" # or "highs" (no licence; slower master, same certificates)
    ng_size::Int = 8
    ng_adapt_rounds::Int = 50
    cut_rounds::Int = 50
    cuts_per_round::Int = 50
    cut_violation::Float64 = 0.05
    cut_tailoff::Float64 = 1e-4
    subset_row_cuts::Int = 20      # per round, after capacity cuts are exhausted (0 = off)
    subset_row_limit::Int = 128    # in total (at most MAX_SUBSET_ROW); each cut slows exact pricing
    bucket_cap::Int = 64
    max_columns::Int = 300
    heuristic_pricing::Bool = true
    wide_bucket_cap::Int = 512   # second heuristic level: full graph, this many labels per node
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
    arc_fixing::Bool = true          # reduced-cost arc fixing with the root duals
    enum_gap::Float64 = 0.006        # try route enumeration at nodes within this relative gap (0 = off)
    enum_route_cap::Int = 1_000_000  # give up enumerating beyond this many routes
    enum_label_limit::Int = 30_000_000
    enum_time::Float64 = 900.0       # seconds per enumeration attempt
    enum_mip_time::Float64 = 900.0   # seconds for the MIP over the enumerated routes
    lazy_certify::Bool = true        # certify tree nodes only when it can fathom them (see search!)
    lazy_depth::Int = 3              # ...but always certify nodes this shallow, so the global bound moves
    certify_cut_rounds::Bool = false # certify every root subset-row round (else only at the end)
    lns_time::Float64 = 600.0        # route-subset re-optimisation budget after the root (0 = off)
    primal_only::Bool = false        # heuristic root, then MIP + re-optimisation until the time limit; no bound
    lns_routes::Int = 3              # incumbent routes re-optimised together
    lns_sub_time::Float64 = 60.0     # time limit of each sub-problem solve
    root_mip_time::Float64 = 300.0   # the root column pool is the richest; give its MIP longer
    max_nodes::Int = typemax(Int)
    label_limit::Int = typemax(Int)
    verbose::Bool = true
end

BPParameters(base::BPParameters; kwargs...) =
    BPParameters(; (k => get(kwargs, k, getfield(base, k)) for k in fieldnames(BPParameters))...)

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
    env::Backend
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
    # duals and value of the exactly converged root LP, for reduced-cost arc fixing
    root_duals::Union{Nothing,Tuple{Vector{Float64},Float64,Vector{Float64},Vector{Float64}}}
    root_z::Float64
    fixed_at::Float64       # incumbent value at the last fixing pass
    arcs_fixed::Int
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

"""
Cheapest feasible charging plan for a fixed customer order (e.g. an ALNS
route), or `nothing`: time-free pricing restricted to exactly the sequence's
successions, with a large dual on its customers so that only routes serving
all of them are negative, then exact re-timing.
"""
function stitch_sequence(s::BPState{W}, seq::Vector{Int}) where {W}
    inst = s.inst
    (isempty(seq) || !allunique(seq)) && return nothing
    chain = vcat(inst.depot, seq, inst.depot)
    allowed = Set(zip(chain[1:end-1], chain[2:end]))
    pairs = [(a, b) for a in 1:inst.n+1, b in 1:inst.n+1 if a != b && !((a, b) in allowed)]
    restr = Restrictions{W}(inst, pairs)
    π = zeros(inst.n)
    π[seq] .= 1e3
    res = price!(s.pricing, inst, π, 0.0, restr; exact = true, timefree = true, untimed = s.untimed,
                 max_columns = 1, time_limit = 60.0)
    isempty(res.routes) && return nothing
    r = first(res.routes)
    r.customers == seq || return nothing
    schedule_route(inst, r)
end

"""
Large-neighbourhood search with the solver itself: pick a few neighbouring
incumbent routes, re-solve their customers as a sub-instance (other routes
fixed), keep any improvement. Primal only; it certifies nothing.
"""
function lns!(s::BPState, budget::Float64)
    inst, p, mp = s.inst, s.params, s.master
    (isempty(s.incumbent) || budget <= 0) && return false
    deadline = min(s.start + p.time_limit, time() + budget)
    improved = false
    tried = Set{Vector{Int}}()
    sub_params = BPParameters(; time_limit = p.lns_sub_time, threads = p.threads, verbose = false,
                              lns_time = 0.0, dive_time = min(p.dive_time, 30.0), root_mip_time = 20.0,
                              mip_time = 10.0, ng_size = p.ng_size, subset_row_cuts = p.subset_row_cuts)
    # route "distance": cheapest travel between any two of their customers
    near(a, b) = minimum(inst.hours[i, j] + inst.hours[j, i] for i in a.customers, j in b.customers)
    width = p.lns_routes
    while time() < deadline
        routes = s.incumbent
        k = min(width, length(routes))
        k < 2 && break
        order = sortperm([r.travel for r in routes]; rev = true)
        seed = nothing
        for first in order
            group = sort(sortperm([q == first ? -Inf : near(routes[first], routes[q]) for q in eachindex(routes)])[1:k])
            group in tried && continue
            seed = group
            break
        end
        if seed === nothing
            # every neighbourhood of this width failed: widen it, up to six routes
            (width >= min(6, length(routes))) && break
            width += 1
            note!(s, "route re-optimisation: widening to $(width) routes")
            continue
        end
        push!(tried, seed)
        chosen = routes[seed]
        customers = sort!(vcat([r.customers for r in chosen]...))
        current = sum(r.travel for r in chosen)
        idle = inst.fleet - length(routes)
        sub, back = subinstance(inst, customers, k + min(idle, 1))
        local_index = Dict(c => i for (i, c) in enumerate(customers))
        append!(sub.initial_sequences, [[local_index[c] for c in r.customers] for r in chosen])
        res = solve(sub; params = BPParameters(sub_params; time_limit = min(p.lns_sub_time, deadline - time())),
                    env = s.env)
        isfinite(res.upper) && res.upper < current - 1e-6 || continue
        replacement = [Route(inst, back[r.nodes], r.levels) for r in res.routes]
        all(r -> simulate(inst, r)[1], replacement) || continue
        foreach(r -> add_column!(mp, inst, r), replacement)
        others = routes[setdiff(eachindex(routes), seed)]
        if improve!(s, vcat(others, replacement), "route re-optimisation")
            improved = true
            empty!(tried)
        end
    end
    improved
end

"""One pricing round in the mode of the current phase."""
function price_round(s::BPState{W}, restr::Restrictions{W}, π, κ, σ, τ; exact::Bool, farkas::Bool,
                     time_limit::Float64, wide::Bool = false) where {W}
    p, mp = s.params, s.master
    cap = wide ? p.wide_bucket_cap : p.bucket_cap
    if s.timefree && s.untimed && p.bidirectional
        price_bidirectional!(s.pricing, s.backward, s.inst, π, κ, restr; exact = exact,
                             max_columns = p.max_columns, rc_tol = p.rc_tol, time_limit = time_limit,
                             label_limit = p.label_limit, bucket_cap = cap,
                             cost_scale = farkas ? 0.0 : 1.0, cuts = mp.cuts, σ = σ,
                             sr_cuts = mp.sr_cuts, τ = τ, full_graph = wide)
    else
        price!(s.pricing, s.inst, π, κ, restr; exact = exact, timefree = s.timefree,
               untimed = s.timefree && s.untimed,
               max_columns = p.max_columns, rc_tol = p.rc_tol, time_limit = time_limit,
               label_limit = p.label_limit, bucket_cap = cap,
               cost_scale = farkas ? 0.0 : 1.0, cuts = mp.cuts, σ = σ, sr_cuts = mp.sr_cuts, τ = τ,
               full_graph = wide)
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
            # the wide level: narrow pricing misses too much at 100+ customers
            res = price_round(s, restr, π, κ, σ, τ; exact = false, farkas = false,
                              time_limit = deadline - time(), wide = true)
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
    # pricing level: 0 narrow heuristic, 1 wide heuristic, 2 exact
    first_level = p.heuristic_pricing ? 0 : 2
    level = first_level
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
        exact = level == 2
        res = price_round(s, restr, π, κ, σ, τ; exact = exact, farkas = farkas,
                          time_limit = p.time_limit - elapsed(s), wide = level == 1)
        s.pricing_seconds += res.seconds
        s.rounds += 1
        if res.exact && !farkas
            s.exact_rounds += 1
            bound = max(bound, lp + node.hi * min(0.0, res.min_rc))
        end
        if p.verbose && (s.rounds % 10 == 0 || isempty(res.routes))
            note!(s, @sprintf("node %d round %d %s %.6f bound %.6f rc %.4g cols %d (+%d) labels %d %s %.2fs",
                node.id, s.rounds, farkas ? "infeasibility" : "lp", lp, bound, res.min_rc, length(mp.z),
                length(res.routes), res.labels, ("narrow", "wide", "exact")[level+1], res.seconds))
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
                !certify && !farkas && level == 1 && return (:solved, bound, column_values(mp), lp)
                level += 1
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
        level = first_level
    end
end

"""Re-mark columns that are no longer ng-routes after the neighbourhoods grew."""
function refresh_ng!(s::BPState)
    mp = s.master
    for k in eachindex(mp.routes)
        mp.ng_ok[k] = ng_feasible(s.pricing, s.inst, mp.routes[k])
    end
end

"""
Reduced-cost arc fixing. With duals of an exactly converged root LP of value
z (every column then has reduced cost >= 0), any integer solution x costs at
least z + rc(p) for each route p in x. A route through arc (i, j) has reduced
cost at least prefix(i) + rc(i, j) + completion(j) - kappa, from the q-route
bounds (capacity split over every load at i; cut earnings at their maximum,
subset-row penalties dropped). If that exceeds incumbent - z, no improving
solution uses the arc, so it leaves the pricing graph and its columns leave
the master. Returns the number of arcs removed.
"""
function fix_arcs!(s::BPState)
    (s.params.arc_fixing && s.root_duals !== nothing) || return 0
    upper = prune_value(s)
    isfinite(upper) && upper < s.fixed_at - 1e-9 || return 0
    s.fixed_at = upper
    inst, ws, bs, mp = s.inst, s.pricing, s.backward, s.master
    π, κ, σ, _ = s.root_duals
    n, N, depot, R = inst.n, inst.N, inst.depot, ws.R
    gap = upper - s.root_z + tolerance(s, upper)
    earn = copy(π)
    if any(>(1e-12), σ)
        succession_duals!(ws.rho, mp.cuts, σ, n)
        for w in 1:n
            earn[w] = π[w] + maximum(view(ws.rho, :, w))
        end
    end
    completion_bound!(ws, inst, earn, 1.0)
    prefix_bound!(bs, ws, inst, earn, 1.0)
    g, h, du = ws.bound, bs.prefix, ws.dunits
    removed = Tuple{Int,Int}[]
    @inbounds for i in 1:N, j in inst.out[i]
        best = Inf
        if j == depot
            i == depot && continue
            best = h[i, R+1] + inst.hours[i, j]
        else
            dj = j <= n ? du[j] : 0
            arc = inst.hours[i, j] - (j <= n ? earn[j] : 0.0)
            if i == depot
                best = arc + g[j, R-dj+1]     # nothing served before leaving the depot
            else
                for a in 0:(R - dj)
                    pre = h[i, a+1]
                    pre < Inf || continue
                    val = pre + arc + g[j, R-a-dj+1]
                    val < best && (best = val)
                end
            end
        end
        best - κ > gap && push!(removed, (i, j))
    end
    isempty(removed) && return 0
    for (i, j) in removed
        inst.usable[i, j] = false
    end
    for i in 1:N
        filter!(j -> inst.usable[i, j], inst.out[i])
        filter!(j -> inst.usable[i, j], ws.near[i])
    end
    for k in eachindex(mp.routes)
        nodes = mp.routes[k].nodes
        mp.arc_ok[k] && any(!inst.usable[nodes[t], nodes[t+1]] for t in 1:length(nodes)-1) && (mp.arc_ok[k] = false)
    end
    s.arcs_fixed += length(removed)
    note!(s, @sprintf("arc fixing at gap %.4f: removed %d arcs (%d total, %d left)", upper - s.root_z,
        length(removed), s.arcs_fixed, sum(inst.usable)))
    length(removed)
end

"""
Close a node by route enumeration. With duals of the node's exactly converged
LP (value `lp`), every solution in the node's subtree that beats the cutoff
uses only routes of reduced cost below cutoff - lp. When that threshold is
small, enumerate all such elementary routes (respecting the node's branching
restrictions) and solve the set-partitioning MIP over them: its optimum, or
its infeasibility, settles the whole subtree. Returns :closed, or :open when
the gap is too wide or a cap or time limit stopped the attempt.
"""
function close_by_enumeration!(s::BPState{W}, node::BBNode, lp::Float64, duals) where {W}
    p, inst, mp = s.params, s.inst, s.master
    cutoff = prune_value(s)
    p.enum_gap > 0 && isfinite(cutoff) || return :open
    threshold = cutoff - lp
    threshold <= p.enum_gap * max(1.0, abs(cutoff)) || return :open
    π, κ, σ, τ = duals
    restr = Restrictions{W}(inst, node.forbidden)
    t = time()
    res = enumerate_routes!(s.pricing, inst, π, κ, restr; threshold = threshold + tolerance(s, cutoff),
                            timefree = s.timefree, untimed = s.timefree && s.untimed, cuts = mp.cuts, σ = σ,
                            sr_cuts = mp.sr_cuts, τ = τ, time_limit = min(p.enum_time, p.time_limit - elapsed(s)),
                            label_limit = p.enum_label_limit, route_cap = p.enum_route_cap)
    if !res.exact
        note!(s, @sprintf("enumeration at node %d (threshold %.4f) gave up after %.0fs, %d labels",
            node.id, threshold, time() - t, res.labels))
        return :open
    end
    routes = res.routes
    # in phase 2 only truly feasible routes belong to the problem
    if !s.timefree
        routes = [r2 for r2 in (schedule_route(inst, r) for r in routes) if r2 !== nothing]
    end
    note!(s, @sprintf("enumeration at node %d: %d routes within %.4f h in %.0fs", node.id, length(routes),
        threshold, time() - t))
    isempty(routes) && return :closed
    budget = min(p.enum_mip_time, p.time_limit - elapsed(s))
    budget > 1.0 || return :open
    model = new_model(s.env; threads = p.threads)
    set_time_limit_sec(model, budget)
    x = @variable(model, [1:length(routes)], Bin)
    @objective(model, Min, sum(r.travel * x[k] for (k, r) in enumerate(routes)))
    members = [Int[] for _ in 1:inst.n]
    for (k, r) in enumerate(routes), c in r.customers
        push!(members[c], k)
    end
    any(isempty, members) && return :closed      # some customer cannot be served below the cutoff
    @constraint(model, [c in 1:inst.n], sum(x[k] for k in members[c]) == 1)
    @constraint(model, node.lo <= sum(x) <= node.hi)
    set_cutoff!(model, s.env, cutoff - tolerance(s, cutoff))
    optimize!(model)
    status = termination_status(model)
    if has_values(model) && objective_value(model) < cutoff - tolerance(s, cutoff)
        status in (MOI.OPTIMAL, MOI.OBJECTIVE_LIMIT) || return :open    # e.g. a time limit
        chosen = [routes[k] for k in eachindex(routes) if value(x[k]) > 0.5]
        foreach(r -> add_column!(mp, inst, r), chosen)
        index = Dict(hash(mp.routes[q].nodes) => q for q in eachindex(mp.routes))
        integer_solution!(s, [index[hash(r.nodes)] for r in chosen], "enumeration at node $(node.id)")
        return :closed
    end
    # No solution below the cutoff: proven by optimality, infeasibility or the cutoff itself.
    status in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.OBJECTIVE_LIMIT) ? :closed : :open
end

"""Whether the LP solution `z` would fathom its node (integral columns or successions)."""
function fathomable(mp::Master, z::Vector{Float64})
    all(v -> v < 1e-6 || v > 1 - 1e-6, z) && return true
    routes = sum(z)
    abs(routes - round(routes)) > 1e-6 && return false
    all(v -> abs(v - round(v)) <= 1e-6, values(succession_flows(mp, z)))
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
function mip_heuristic!(s::BPState; seconds::Float64 = s.params.mip_time)
    inst, mp = s.inst, s.master
    cols = [k for k in eachindex(mp.routes) if mp.feasible[k]]
    isempty(cols) && return false
    budget = min(seconds, s.params.time_limit - elapsed(s))
    budget > 1.0 || return false
    model = new_model(s.env; threads = s.params.threads)
    set_time_limit_sec(model, budget)
    x = @variable(model, [1:length(cols)], Bin)
    @objective(model, Min, sum(mp.routes[k].travel * x[i] for (i, k) in enumerate(cols)))
    members = [Int[] for _ in 1:inst.n]
    for (i, k) in enumerate(cols), c in mp.routes[k].customers
        push!(members[c], i)
    end
    any(isempty, members) && return false
    @constraint(model, [c in 1:inst.n], sum(x[i] for i in members[c]) == 1)
    @constraint(model, sum(x) <= inst.fleet)
    isfinite(s.upper) && set_cutoff!(model, s.env, s.upper - 1e-7)
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
    # Stages: :fast rounds use LPs converged under heuristic pricing and add
    # capacity cuts only; :capacity rounds certify every LP; :subset_row rounds
    # add subset-row cuts as well (their duals make heuristic pricing
    # unreliable, so they are only separated from certified LPs). A stage ends
    # when it finds nothing or its last round barely moved the LP.
    stage = :fast
    sr_certified = p.certify_cut_rounds
    last_certified = false
    function advance!()
        if stage == :fast
            stage = :capacity
            result, nbound, z, lpvalue = resolve(true)
            last_certified = true
            note!(s, @sprintf("root certified: lp %.6f, %d ng rounds", lpvalue, adapted))
        elseif stage == :capacity && p.subset_row_cuts > 0
            stage = :subset_row
        else
            stage = :done
        end
    end
    round = 0
    while round < p.cut_rounds
        round += 1
        if result != :solved
            note!(s, "cut loop: node $(result) at round $(round)")
            return (result, nbound, z, lpvalue)
        end
        if stage == :done
            last_certified && break
            # Subset-row rounds ran on heuristic LPs: certify, and if that
            # moves the LP materially, repeat them on certified LPs.
            before = lpvalue
            result, nbound, z, lpvalue = resolve(true)
            last_certified = true
            note!(s, @sprintf("root certified: lp %.6f (heuristic %.6f)", lpvalue, before))
            if result == :solved && !sr_certified && p.subset_row_cuts > 0 &&
               before - lpvalue > 10 * p.cut_tailoff * max(1.0, abs(lpvalue))
                sr_certified = true
                stage = :subset_row
                continue
            end
            break
        end
        certified = stage == :capacity || (stage == :subset_row && sr_certified)
        while adapted < p.ng_adapt_rounds && grow_neighbourhoods!(s.pricing, mp.routes, z)
            refresh_ng!(s)
            adapted += 1
            result, nbound, z, lpvalue = resolve(certified)
            last_certified = certified
            result == :solved || break
        end
        result == :solved || continue
        cuts = separate_capacity_cuts(inst, succession_flows(mp, z), mp.known_cuts;
                                      max_cuts = p.cuts_per_round, min_violation = p.cut_violation)
        rows = stage == :subset_row ?
            separate_subset_row_cuts(inst, mp.routes, z, mp.known_sr;
                max_cuts = min(p.subset_row_cuts, min(p.subset_row_limit, MAX_SUBSET_ROW) - length(mp.sr_cuts)),
                min_violation = p.cut_violation) : SubsetRowCut[]
        if isempty(cuts) && isempty(rows)
            note!(s, "cut loop: nothing violated in stage $(stage) at round $(round)")
            advance!()
            continue
        end
        foreach(c -> add_cut!(mp, inst, c), cuts)
        foreach(c -> add_sr_cut!(mp, c), rows)
        before = lpvalue
        result, nbound, z, lpvalue = resolve(certified)
        last_certified = certified
        note!(s, @sprintf("cut round %d (%s): +%d capacity / +%d subset-row cuts (%d / %d total), lp %.6f",
            round, stage, length(cuts), length(rows), length(mp.cuts), length(mp.sr_cuts), lpvalue))
        if result == :solved && !(lpvalue - before > p.cut_tailoff * max(1.0, abs(before)))
            note!(s, @sprintf("cut loop: stage %s tailed off (%.3g)", stage, lpvalue - before))
            advance!()
        end
    end
    if result == :solved && !last_certified
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
        if p.primal_only
            # Upper bounds only: a heuristically converged root feeds the
            # restricted-master MIP and route re-optimisation for the whole
            # budget. Bounds certified elsewhere stay valid against its result.
            result, nbound, z, lpvalue = solve_node!(s, node; certify = false)
            mip_heuristic!(s; seconds = p.root_mip_time)
            lns!(s, p.time_limit - elapsed(s))
            note!(s, @sprintf("primal-only search finished at %.6f", s.upper))
            pending, closed = -Inf, false
            break
        end
        # Lazy certification: a node converged under heuristic pricing may be
        # branched on (its children keep the inherited, still valid bound);
        # exact pricing runs only where a certified LP decides something,
        # namely when it could prune the node or the LP is integral.
        lazy = id != 1 && p.lazy_certify && node.depth > p.lazy_depth
        result, nbound, z, lpvalue = solve_node!(s, node; certify = !lazy && id != 1)
        certified = id != 1 && !lazy
        if lazy && result == :solved
            cutoff = prune_value(s)
            near = p.enum_gap > 0 && cutoff - lpvalue <= p.enum_gap * max(1.0, abs(cutoff))
            if lpvalue >= cutoff - tolerance(s, cutoff) || fathomable(mp, z) || near
                result, nbound, z, lpvalue = solve_node!(s, node; certify = true)
                certified = true
            end
        end
        if certified && result == :solved
            here0 = BBNode(node.id, node.depth, nbound, node.forbidden, node.lo, node.hi, node.label)
            close_by_enumeration!(s, here0, lpvalue, master_duals(mp)) == :closed && continue
        end
        if id == 1
            result, nbound, z, lpvalue = strengthen_root!(s, node, result, nbound, z, lpvalue)
            root_bound = nbound
            if result == :solved
                s.root_duals = master_duals(mp)
                s.root_z = lpvalue
            end
            if p.dive_time > 0 && result == :solved
                cutoff = prune_value(s)
                nbound < cutoff - tolerance(s, cutoff) && dive!(s, node, p.dive_time)
                # The dive left the master restricted; re-solve the root as it was.
                # Its bound is already certified and the dive only added
                # columns, so heuristic convergence suffices to branch on.
                result, nbound, z, lpvalue = solve_node!(s, BBNode(1, 0, nbound, node.forbidden, node.lo, node.hi, "root");
                                                         certify = false)
            end
            root_lp = lpvalue
            note!(s, @sprintf("%s root: lp %.6f bound %.6f, %d columns",
                s.timefree ? "relaxed" : "exact", root_lp, root_bound, length(mp.z)))
            mip_heuristic!(s; seconds = p.root_mip_time)
            if p.lns_time > 0 && s.upper > nbound + tolerance(s, nbound)
                before = s.upper
                lns!(s, p.lns_time)
                note!(s, @sprintf("route re-optimisation: %.6f -> %.6f", before, s.upper))
            end
            fix_arcs!(s)
            if result == :solved && s.root_duals !== nothing &&
               close_by_enumeration!(s, node, s.root_z, s.root_duals) == :closed
                continue
            end
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
        fix_arcs!(s)
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

function solve(inst::Instance; params::BPParameters = BPParameters(),
               env::Backend = Backend(Symbol(params.lp_solver)))
    W = words_for(inst.n + 1)
    _solve(Val(W), inst, params, env)
end

function _solve(::Val{W}, inst::Instance, params::BPParameters, env::Backend) where {W}
    start = time()
    mp = Master(inst, env; threads = params.threads, big_m = params.big_m, method = params.lp_method)
    ws = PricingWorkspace{W}(inst; ng_size = params.ng_size, unit = params.capacity_unit,
                             near = params.heuristic_neighbors)
    windows = any(isfinite, inst.tw_close[1:inst.n]) || any(>(0), inst.tw_open[1:inst.n])
    s = BPState{W}(inst, params, env, mp, ws, BackwardSpace{W}(inst, ws.R), start, params.relax_charging_time,
                   params.relax_horizon && !windows, Route[], Inf, Inf,
                   Route[], 0, 0, 0.0, 0.0, Dict{String,Any}[], nothing, -Inf, Inf, 0)

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
    if !isempty(inst.initial_sequences)
        stitched = Route[]
        for seq in inst.initial_sequences
            r = stitch_sequence(s, seq)
            r === nothing && continue
            add_column!(mp, inst, r)
            push!(stitched, r)
        end
        note!(s, "stitched $(length(stitched)) of $(length(inst.initial_sequences)) warm-start sequences")
        improve!(s, stitched, "stitched warm-start sequences")
    end
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
