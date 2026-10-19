# Pricing: elementary-relaxed (ng-route) shortest path with resource
# constraints, solved by forward label setting in nondecreasing time.
#
# This is the path-label pricing of ElectricRouting.jl's benchmark mode
# (after Desaulniers et al. 2016), adapted to the environment:
# * charging is a choice among the discrete target-SoC action levels at a
#   charger, timed by that station's own curve, so no continuous "charge
#   later" relaxation is involved (with mixed station speeds that relaxation
#   would not be a valid bound);
# * every leg needs `margin * energy` on board while consuming `energy`;
# * customer successions can be forbidden by the branching rule.
#
# Labels are processed in nondecreasing `time`: every extension adds a
# strictly positive travel time and every dominator of a label is no later,
# so a label is final when popped.
#
# The search is exact when run on the full graph with all levels and no label
# or time cap; only then may its minimum reduced cost enter a Lagrangian bound.
#
# Time-free mode prices a relaxation: charging takes no time, so `time` is a
# lower bound (travel, service and window waiting) that still enforces windows
# and the horizon, and every charger stop charges to the top level, which is
# never worse when time is free. Every truly feasible route stays feasible, so
# bounds remain valid; priced routes are re-timed exactly before they may
# enter an incumbent (see `schedule_route`).
#
# `untimed` (instances without time windows) also drops the horizon, so time
# leaves dominance altogether; only the processing order still uses it.
# Charger detours stay finite: at a charger every label leaves full, so a
# costlier later arrival with the same load and memory is dominated.

mutable struct Label{W}
    node::Int
    parent::Int
    dep_level::Float64   # level charged at the parent's node before leaving it
    cost::Float64        # reduced cost so far
    time::Float64        # customers: after service; chargers: on arrival
    battery::Float64
    load::Float64
    ng::NodeSet{W}
    last::Int            # last customer served, depot index before the first
    sr::SRBits           # parity of visits to each subset-row triple
    alive::Bool
end

struct Restrictions{W}
    forbidden::BitMatrix
    successors::Vector{NodeSet{W}}
    predecessors::Vector{NodeSet{W}}
    active::Bool
end

function Restrictions{W}(inst::Instance, pairs) where {W}
    k = inst.n + 1
    forbidden = falses(k, k)
    for (a, b) in pairs
        forbidden[a, b] = true
    end
    successors = [nodeset(Val(W), findall(forbidden[a, :])) for a in 1:k]
    predecessors = [nodeset(Val(W), findall(forbidden[:, b])) for b in 1:k]
    Restrictions{W}(forbidden, successors, predecessors, any(forbidden))
end

mutable struct PricingWorkspace{W}
    pool::Vector{Label{W}}
    buckets::Vector{Vector{Int}}
    bcost::Vector{Vector{Float64}}
    heap::BinaryMinHeap{Tuple{Float64,Int}}
    options::Vector{NTuple{3,Float64}}
    cmod::Matrix{Float64}
    bound::Matrix{Float64}
    bound2::Matrix{Float64}
    first::Matrix{Int}
    cpath::Matrix{Float64}
    dunits::Vector{Int}
    unit::Float64
    R::Int
    ng::Vector{NodeSet{W}}
    near::Vector{Vector{Int}}
    rho::Matrix{Float64}      # capacity-cut earnings rho[last, w]
    slack::Matrix{Float64}    # dominance correction between last customers
    earn::Vector{Float64}     # pi_w + max_l rho[l, w], for the completion bound
    has_cuts::Bool
    sr_mask::Vector{SRBits}   # subset-row triples containing each customer
    sr_keep::Vector{SRBits}   # subset-row cuts whose memory holds each customer
    sr_pen::Vector{Float64}   # |tau| per subset-row cut
    has_sr::Bool
    enumerating::Bool         # route enumeration: dominance only between equal visited sets
end

function PricingWorkspace{W}(inst::Instance; ng_size::Int, unit::Float64, near::Int) where {W}
    N, n = inst.N, inst.n
    cpath = [inst.usable[i, j] ? inst.hours[i, j] : Inf for i in 1:N, j in 1:N]
    for k in inst.depot+1:N, i in 1:N, j in 1:N
        cpath[i, j] = min(cpath[i, j], cpath[i, k] + cpath[k, j])
    end
    for i in 1:N
        cpath[i, i] = Inf
    end
    # Floor demands and ceil capacity on the grid so the bound stays a relaxation.
    dunits = [max(1, floor(Int, inst.demand[j] / unit - 1e-7)) for j in 1:n]
    R = floor(Int, inst.payload / unit + 1e-6)
    lists = map(1:N) do v
        customers = sort([w for w in inst.out[v] if is_customer(inst, w)]; by = w -> inst.hours[v, w])
        others = [w for w in inst.out[v] if !is_customer(inst, w)]
        vcat(customers[1:min(near, length(customers))], others)
    end
    PricingWorkspace{W}(
        Label{W}[], [Int[] for _ in 1:N], [Float64[] for _ in 1:N], BinaryMinHeap{Tuple{Float64,Int}}(),
        NTuple{3,Float64}[], zeros(N, N), zeros(N, R + 1), zeros(N, R + 1), zeros(Int, N, R + 1),
        cpath, dunits, unit, R,
        ng_neighbourhoods(inst, Val(W), ng_size), lists,
        zeros(n + 1, n), zeros(n + 1, n + 1), zeros(n), false,
        zeros(SRBits, n), zeros(SRBits, n), zeros(MAX_SUBSET_ROW), false, false,
    )
end

struct PricingResult
    routes::Vector{Route}
    rcs::Vector{Float64}
    min_rc::Float64
    exact::Bool
    labels::Int
    seconds::Float64
end

@inline capacity_units(ws::PricingWorkspace, inst::Instance, load::Float64) =
    clamp(floor(Int, (inst.payload - load) / ws.unit + 1e-6), 0, ws.R)

"""Sum of subset-row penalties over the cuts in `bits`."""
@inline function srpen_word(pen::Vector{Float64}, bits::UInt128, offset::Int)
    total = 0.0
    while bits != zero(UInt128)
        total += @inbounds pen[offset+trailing_zeros(bits)+1]
        bits &= bits - one(UInt128)
    end
    total
end

@inline srpen(pen::Vector{Float64}, bits::SRBits) = srpen_word(pen, bits.lo, 0) + srpen_word(pen, bits.hi, 128)

"""Install subset-row duals: a label pays |tau| whenever its parity bit for a cut flips back to even."""
function set_subset_rows!(ws::PricingWorkspace, sr_cuts::Vector{SubsetRowCut}, τ::Vector{Float64}, n::Int)
    fill!(ws.sr_mask, zero(SRBits))
    fill!(ws.sr_keep, zero(SRBits))
    fill!(ws.sr_pen, 0.0)
    ws.has_sr = false
    for (c, (cut, t)) in enumerate(zip(sr_cuts, τ))
        t < -1e-12 || continue
        ws.has_sr = true
        ws.sr_pen[c] = -t
        bit = sr_bit(c)
        for w in cut.members
            ws.sr_mask[w] |= bit
        end
        for w in 1:n
            cut.memory[w] && (ws.sr_keep[w] |= bit)
        end
    end
end

# q-route bound with 2-cycle elimination: least reduced cost of any
# depot-bound completion from `v` serving customers whose floored demand fits
# the remaining capacity, with energy, time and elementarity relaxed except
# that a completion never steps straight back to the customer it came from,
# and legs priced at the cheapest charger-only detour. For every (v, r) the
# best value and the best value with a different first customer are kept.
function completion_bound!(ws::PricingWorkspace, inst::Instance, π::Vector{Float64}, scale::Float64)
    g, g2, f, cp, du = ws.bound, ws.bound2, ws.first, ws.cpath, ws.dunits
    n, N, depot = inst.n, inst.N, inst.depot
    @inbounds for r in 0:ws.R
        for v in 1:N
            best = v == depot || cp[v, depot] == Inf ? Inf : scale * cp[v, depot]
            first = 0
            second = Inf
            for j in 1:n
                d = du[j]
                d <= r || continue
                c = cp[v, j]
                c < Inf || continue
                tail = f[j, r-d+1] == v ? g2[j, r-d+1] : g[j, r-d+1]
                val = scale * c - π[j] + tail
                if val < best
                    second = best
                    best, first = val, j
                elseif val < second
                    second = val
                end
            end
            g[v, r+1] = best
            g2[v, r+1] = second
            f[v, r+1] = first
        end
    end
end

# Departure options at `l.node`, sorted by time (and battery): stay as is, or
# charge to any level strictly above the current SoC.
function fill_options!(ws::PricingWorkspace, inst::Instance, l::Label, levels::Vector{Float64}, timefree::Bool)
    opts = ws.options
    empty!(opts)
    if timefree
        top = last(levels)
        if is_charger(inst, l.node) && top * inst.battery > l.battery + EPS
            push!(opts, (top * inst.battery, l.time, top))
        else
            push!(opts, (l.battery, l.time, 0.0))
        end
        return opts
    end
    push!(opts, (l.battery, l.time, 0.0))
    if is_charger(inst, l.node)
        for L in levels
            target = L * inst.battery
            target > l.battery + EPS || continue
            push!(opts, (target, l.time + charge_hours(inst, l.node, l.battery, L), L))
        end
    end
    opts
end

# Earliest departure of a label at charger `w` holding at least `need` kWh.
@inline function earliest_with(inst::Instance, w::Int, t::Float64, b::Float64, need::Float64, levels)
    b >= need - 1e-12 && return t
    for L in levels
        target = L * inst.battery
        (target >= need - 1e-12 && target > b + EPS) || continue
        return t + charge_hours(inst, w, b, L)
    end
    Inf
end

# At a charger `a` dominates `b` if for every departure option of `b` it has
# one with at least as much energy that leaves no later.
function charger_dominates(inst::Instance, w::Int, a::Label, b::Label, levels)
    earliest_with(inst, w, a.time, a.battery, b.battery, levels) <= b.time + 1e-12 || return false
    for L in levels
        target = L * inst.battery
        target > b.battery + EPS || continue
        tb = b.time + charge_hours(inst, w, b.battery, L)
        earliest_with(inst, w, a.time, a.battery, target, levels) <= tb + 1e-12 || return false
    end
    true
end

@inline function dominates(ws::PricingWorkspace, inst::Instance, restr::Restrictions, a::Label, b::Label, levels,
                           timefree::Bool, untimed::Bool)
    # Enumeration needs every route below the threshold, not just the cheapest,
    # so only labels over the same customers may dominate each other.
    ws.enumerating && a.ng != b.ng && return false
    a.load <= b.load + 1e-12 || return false
    # a may still pay a subset-row penalty b would not: on the cuts where only a is odd
    acost = ws.has_sr ? a.cost + srpen(ws.sr_pen, a.sr & ~b.sr) : a.cost
    if is_charger(inst, a.node) && a.last != b.last
        # Different last customers earn different cut duals from here on.
        acost <= b.cost + (ws.has_cuts ? ws.slack[a.last, b.last] : 0.0) + 1e-12 || return false
        issubset(a.ng, b.ng) || return false
        if restr.active
            issubset(restr.successors[a.last], restr.successors[b.last]) || return false
        end
    else
        acost <= b.cost + 1e-12 || return false
        issubset(a.ng, b.ng) || return false
    end
    if is_charger(inst, a.node)
        # Time-free: both leave fully charged, so arrival energy is irrelevant.
        timefree && return untimed || a.time <= b.time + 1e-12
        return charger_dominates(inst, a.node, a, b, levels)
    end
    (untimed || a.time <= b.time + 1e-12) && a.battery >= b.battery - 1e-12
end

"""
Insert `lab` into a bucket kept sorted by cost (with its costs mirrored in
`costs`), applying dominance in both directions. Every dominance rule here
needs the dominating label to be no costlier, so only cheaper labels are
tried as dominators of `lab`, and only costlier ones as its victims. With a
finite `cap` (heuristic rounds) only the `cap` cheapest labels are kept.
Returns the new pool index, or 0 when `lab` is discarded.
"""
function sorted_insert!(dom::F, pool::Vector{L}, bucket::Vector{Int}, costs::Vector{Float64}, lab::L,
                        cap::Int) where {F,L}
    c = lab.cost
    hi = searchsortedlast(costs, c + 1e-12)
    @inbounds for k in 1:hi
        dom(pool[bucket[k]], lab) && return 0
    end
    lo = searchsortedfirst(costs, c - 1e-12)
    w = lo
    @inbounds for k in lo:length(bucket)
        old = pool[bucket[k]]
        if dom(lab, old)
            old.alive = false
        else
            bucket[w] = bucket[k]
            costs[w] = costs[k]
            w += 1
        end
    end
    resize!(bucket, w - 1)
    resize!(costs, w - 1)
    if length(bucket) >= cap
        c < costs[end] || return 0
        pool[bucket[end]].alive = false
        pop!(bucket)
        pop!(costs)
    end
    push!(pool, lab)
    idx = length(pool)
    pos = searchsortedfirst(costs, c)
    insert!(bucket, pos, idx)
    insert!(costs, pos, c)
    idx
end

function insert_label!(ws::PricingWorkspace{W}, inst::Instance, restr::Restrictions{W}, lab::Label{W}, levels,
                       timefree::Bool, untimed::Bool, cap::Int) where {W}
    dom(a, b) = dominates(ws, inst, restr, a, b, levels, timefree, untimed)
    idx = sorted_insert!(dom, ws.pool, ws.buckets[lab.node], ws.bcost[lab.node], lab, cap)
    idx == 0 && return false
    push!(ws.heap, (lab.time, idx))
    true
end

function rebuild_route(ws::PricingWorkspace, inst::Instance, idx::Int, last_level::Float64)
    chain = Int[]
    k = idx
    while k != 0
        push!(chain, k)
        k = ws.pool[k].parent
    end
    reverse!(chain)
    nodes = [ws.pool[k].node for k in chain]
    push!(nodes, inst.depot)
    levels = zeros(length(nodes))
    for p in 2:length(chain)
        levels[p-1] = ws.pool[chain[p]].dep_level
    end
    levels[length(chain)] = last_level
    Route(inst, nodes, levels)
end

"""
Price routes against cover duals `π` and route-count dual `κ`.

`exact = false` runs a restricted search (nearest customers, full charges
only, at most `bucket_cap` labels per node); its result never certifies
optimality. `cost_scale = 0` prices Farkas columns for an infeasible master. `timefree` prices the
charging-time relaxation described above (exactly, when `exact`). Returns up
to `max_columns` routes with reduced cost below `-rc_tol`, best first.
"""
function price!(
    ws::PricingWorkspace{W}, inst::Instance, π::Vector{Float64}, κ::Float64,
    restr::Restrictions{W};
    exact::Bool = true, timefree::Bool = false, untimed::Bool = false, max_columns::Int = 200, rc_tol::Float64 = 1e-6,
    time_limit::Float64 = Inf, label_limit::Int = typemax(Int), bucket_cap::Int = 64,
    cost_scale::Float64 = 1.0, cuts::Vector{CapacityCut} = CapacityCut[], σ::Vector{Float64} = Float64[],
    sr_cuts::Vector{SubsetRowCut} = SubsetRowCut[], τ::Vector{Float64} = Float64[], full_graph::Bool = false,
    threshold::Float64 = -Inf, route_cap::Int = typemax(Int),
) where {W}
    # threshold > -Inf: enumeration mode, collect every route with reduced cost <= threshold
    listing = threshold > -Inf
    t0 = time()
    n, N, depot = inst.n, inst.N, inst.depot
    empty!(ws.pool)
    foreach(empty!, ws.buckets)
    foreach(empty!, ws.bcost)
    ws.heap = BinaryMinHeap{Tuple{Float64,Int}}()
    @inbounds for i in 1:N, j in 1:N
        ws.cmod[i, j] = cost_scale * inst.hours[i, j] - (j <= n ? π[j] : 0.0)
    end
    ws.has_cuts = any(>(1e-12), σ)
    if ws.has_cuts
        succession_duals!(ws.rho, cuts, σ, n)
        cut_slack!(ws.slack, ws.rho, n)
        for w in 1:n
            ws.earn[w] = π[w] + maximum(view(ws.rho, :, w))
        end
    else
        ws.earn .= π
    end
    completion_bound!(ws, inst, ws.earn, cost_scale)
    set_subset_rows!(ws, sr_cuts, τ, n)
    cap = exact ? typemax(Int) : bucket_cap
    untimed && (timefree || error("untimed pricing requires the time-free relaxation"))
    horizon = untimed ? Inf : inst.horizon
    levels = (exact && !timefree) ? inst.levels : [maximum(inst.levels)]
    nbrs = (exact || full_graph) ? inst.out : ws.near

    root = Label{W}(depot, 0, 0.0, 0.0, 0.0, inst.start_battery, 0.0, NodeSet{W}(), depot, zero(SRBits), true)
    push!(ws.pool, root)
    push!(ws.heap, (0.0, 1))

    found = Tuple{Float64,Int,Float64}[]
    min_rc = Inf
    complete = true
    popped = 0
    while !isempty(ws.heap)
        popped += 1
        if popped % 64 == 0 && (time() - t0 > time_limit || length(ws.pool) > label_limit ||
                                length(found) > route_cap)
            complete = false
            break
        end
        _, li = pop!(ws.heap)
        l = ws.pool[li]
        l.alive || continue
        v = l.node
        opts = fill_options!(ws, inst, l, levels, timefree)
        @inbounds for w in nbrs[v]
            e = inst.energy[v, w]
            need = inst.margin * e
            if w == depot
                v == depot && continue
                restr.active && restr.forbidden[l.last, depot] && continue
                for (b, t, lev) in opts
                    b + EPS >= need || continue
                    t + inst.hours[v, w] <= horizon + EPS || break
                    rc = l.cost + ws.cmod[v, w] - κ
                    rc < min_rc && (min_rc = rc)
                    if listing
                        rc <= threshold && push!(found, (rc, li, lev))
                    else
                        rc < -rc_tol && push!(found, (rc, li, lev))
                    end
                    break
                end
                continue
            end
            customer = w <= n
            load = l.load
            if customer
                w in l.ng && continue
                restr.active && restr.forbidden[l.last, w] && continue
                load += inst.demand[w]
                load <= inst.payload + EPS || continue
            end
            cost = l.cost + ws.cmod[v, w] - (customer && ws.has_cuts ? ws.rho[l.last, w] : 0.0)
            sr = l.sr
            if customer && ws.has_sr
                m = ws.sr_mask[w]
                sr &= ws.sr_keep[w]          # forget cuts whose memory w is outside
                cost += srpen(ws.sr_pen, sr & m)
                sr = sr ⊻ m
            end
            completion = cost + ws.bound[w, capacity_units(ws, inst, load)+1] - κ
            (listing ? completion > threshold : completion >= 0.0) && continue
            ng = customer ? with(l.ng & ws.ng[w], w) : l.ng
            last = customer ? w : l.last
            for (b, t, lev) in opts
                b + EPS >= need || continue
                nb = b - e
                nt = t + inst.hours[v, w]
                if customer
                    nt <= inst.tw_close[w] + EPS || break
                    nt = max(nt, inst.tw_open[w]) + inst.service[w]
                    nb + EPS >= inst.min_battery_to_leave[w] || continue
                end
                nt + inst.min_hours_to_depot[w] <= horizon + EPS || break
                insert_label!(ws, inst, restr, Label{W}(w, li, lev, cost, nt, nb, load, ng, last, sr, true), levels, timefree, untimed, cap)
            end
        end
    end

    sort!(found; by = first)
    routes = Route[]
    rcs = Float64[]
    for (rc, li, lev) in found
        !listing && length(routes) >= max_columns && break
        push!(routes, rebuild_route(ws, inst, li, lev))
        push!(rcs, rc)
    end
    PricingResult(routes, rcs, min_rc, exact && complete, length(ws.pool), time() - t0)
end

"""Whether `r` is an ng-route under the current neighbourhoods."""
function ng_feasible(ws::PricingWorkspace{W}, inst::Instance, r::Route) where {W}
    memory = NodeSet{W}()
    for c in r.customers
        c in memory && return false
        memory = with(memory & ws.ng[c], c)
    end
    true
end

"""
Adaptive ng-routes (as in ElectricRouting.jl): for every customer repeated by
a column in the LP support, remember it at each customer of the cycle.
Returns whether any neighbourhood grew.
"""
function grow_neighbourhoods!(ws::PricingWorkspace{W}, routes, z::Vector{Float64}) where {W}
    grew = false
    for (k, r) in enumerate(routes)
        z[k] > 1e-6 || continue
        seq = r.customers
        for p in eachindex(seq), q in p+1:length(seq)
            seq[p] == seq[q] || continue
            c = seq[p]
            for m in p+1:q-1
                u = seq[m]
                if !(c in ws.ng[u])
                    ws.ng[u] = with(ws.ng[u], c)
                    grew = true
                end
            end
        end
    end
    grew
end

"""
Every elementary route with reduced cost at most `threshold` under the given
duals and restrictions (`exact` in the result: the enumeration completed
within the caps). Full customer memory makes routes elementary; dominance is
limited to labels over the same customers, and pruning uses the completion
bound against the threshold.
"""
function enumerate_routes!(ws::PricingWorkspace{W}, inst::Instance, π, κ, restr::Restrictions{W};
                           threshold::Float64, timefree::Bool, untimed::Bool, cuts, σ, sr_cuts, τ,
                           time_limit::Float64, label_limit::Int, route_cap::Int) where {W}
    saved = ws.ng
    everyone = nodeset(Val(W), 1:inst.n)
    ws.ng = fill(everyone, inst.n)
    ws.enumerating = true
    try
        price!(ws, inst, π, κ, restr; exact = true, timefree = timefree, untimed = untimed,
               cuts = cuts, σ = σ, sr_cuts = sr_cuts, τ = τ, threshold = threshold,
               route_cap = route_cap, time_limit = time_limit, label_limit = label_limit)
    finally
        ws.ng = saved
        ws.enumerating = false
    end
end
