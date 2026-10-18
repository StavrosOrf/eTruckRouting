# Bidirectional pricing for the time-free, untimed relaxation (instances
# without time windows; see pricing.jl for the relaxation itself).
#
# Forward labels (the `Label`s of pricing.jl) grow from the depot and stop once
# their load exceeds half the payload. Backward labels grow from the depot
# towards the start and stop likewise. A route is assembled at the first
# customer u where its prefix load exceeds C/2: the prefix up to u is a
# forward label (every earlier forward label had load <= C/2, so it was
# extended), and the suffix after u carries less than C/2, so every backward
# label along it was extended too. Routes that never exceed C/2 are completed
# by the forward search. Dominance in either direction keeps a label at least
# as good for every join, so the minimum reduced cost is exact.
#
# A backward label at node u describes the suffix u -> ... -> depot:
# * cost:  reduced cost of the suffix arcs (u's own dual sits on the arc into u,
#          which belongs to the prefix), cut earnings of its successions;
# * load:  demand of the suffix customers, u included;
# * need:  battery required when leaving u. At a charger the truck leaves at
#          the top level, so a suffix from a charger is feasible iff its need
#          is at most that level, and a predecessor only has to reach it with
#          the per-leg margin;
# * ng:    backward ng memory, (memory(next) & N(u)) | {u} at customers; a join
#          at u is accepted when the two memories share only u;
# * first: first customer of the suffix (u itself at a customer), which fixes
#          the succession and cut term of a later predecessor.

mutable struct BLabel{W}
    node::Int
    parent::Int          # label of the next node towards the depot (0 at the depot)
    cost::Float64
    load::Float64
    need::Float64
    ng::NodeSet{W}
    first::Int
    sr::UInt128
    alive::Bool
end

mutable struct BackwardSpace{W}
    pool::Vector{BLabel{W}}
    buckets::Vector{Vector{Int}}
    heap::BinaryMinHeap{Tuple{Float64,Int}}
    prefix::Matrix{Float64}   # least prefix cost depot -> v within r capacity units
    slack::Matrix{Float64}    # dominance correction between first customers
end

BackwardSpace{W}(inst::Instance, R::Int) where {W} = BackwardSpace{W}(
    BLabel{W}[], [Int[] for _ in 1:inst.N], BinaryMinHeap{Tuple{Float64,Int}}(),
    zeros(inst.N, R + 1), zeros(inst.n + 1, inst.n + 1),
)

# Mirror of `completion_bound!`: least reduced cost of a path from the depot
# that arrives at v (v's dual included) with floored prefix demand within r.
function prefix_bound!(bs::BackwardSpace, ws::PricingWorkspace, inst::Instance, earn::Vector{Float64}, scale::Float64)
    h, cp, du = bs.prefix, ws.cpath, ws.dunits
    n, N, depot = inst.n, inst.N, inst.depot
    @inbounds for r in 0:ws.R
        for v in 1:n
            d = du[v]
            if d > r
                h[v, r+1] = Inf
                continue
            end
            best = cp[depot, v] == Inf ? Inf : scale * cp[depot, v]
            for j in 1:n
                j == v && continue
                c = cp[j, v]
                c < Inf || continue
                val = h[j, r-d+1] + scale * c
                val < best && (best = val)
            end
            h[v, r+1] = best - earn[v]
        end
        for v in depot:N
            best = v == depot || cp[depot, v] == Inf ? Inf : scale * cp[depot, v]
            for j in 1:n
                c = cp[j, v]
                c < Inf || continue
                val = h[j, r+1] + scale * c
                val < best && (best = val)
            end
            h[v, r+1] = best
        end
    end
end

@inline function backward_dominates(inst::Instance, restr::Restrictions, bs::BackwardSpace, ws::PricingWorkspace,
                                    a::BLabel, b::BLabel)
    a.load <= b.load + 1e-12 || return false
    a.need <= b.need + 1e-12 || return false
    acost = ws.has_sr ? a.cost + srpen(ws.sr_pen, a.sr & ~b.sr) : a.cost
    if a.first != b.first
        acost <= b.cost + (ws.has_cuts ? bs.slack[a.first, b.first] : 0.0) + 1e-12 || return false
        # forbidden predecessors of a's first customer must be among b's
        restr.active && !issubset(restr.predecessors[a.first], restr.predecessors[b.first]) && return false
    else
        acost <= b.cost + 1e-12 || return false
    end
    issubset(a.ng, b.ng)
end

function insert_backward!(bs::BackwardSpace{W}, inst::Instance, restr::Restrictions{W}, ws::PricingWorkspace{W},
                          lab::BLabel{W}, cap::Int) where {W}
    bucket = bs.buckets[lab.node]
    if length(bucket) >= cap
        worst = argmax(k -> bs.pool[bucket[k]].cost, eachindex(bucket))
        lab.cost < bs.pool[bucket[worst]].cost || return false
        bs.pool[bucket[worst]].alive = false
        bucket[worst] = bucket[end]
        pop!(bucket)
    end
    i = 1
    @inbounds while i <= length(bucket)
        old = bs.pool[bucket[i]]
        backward_dominates(inst, restr, bs, ws, old, lab) && return false
        if backward_dominates(inst, restr, bs, ws, lab, old)
            old.alive = false
            bucket[i] = bucket[end]
            pop!(bucket)
            continue
        end
        i += 1
    end
    push!(bs.pool, lab)
    idx = length(bs.pool)
    push!(bucket, idx)
    push!(bs.heap, (lab.load, idx))
    true
end

@inline succession_earning(ws::PricingWorkspace, l::Int, w::Int, n::Int) =
    ws.has_cuts && w <= n ? ws.rho[l, w] : 0.0

"""Forward labels from the depot up to half the payload. Returns (complete, found, min rc)."""
function forward_pass!(ws::PricingWorkspace{W}, inst::Instance, restr::Restrictions{W}, nbrs, κ::Float64,
                       rc_tol::Float64, cap::Int, levels::Vector{Float64}, top::Float64, topb::Float64,
                       half::Float64, deadline::Float64, label_limit::Int) where {W}
    n, depot = inst.n, inst.depot
    found = Tuple{Float64,Int,Int,Float64}[]   # (rc, forward label, backward label or 0, level at last node)
    min_rc = Inf
    empty!(ws.pool)
    foreach(empty!, ws.buckets)
    ws.heap = BinaryMinHeap{Tuple{Float64,Int}}()
    push!(ws.pool, Label{W}(depot, 0, 0.0, 0.0, 0.0, inst.start_battery, 0.0, NodeSet{W}(), depot, zero(UInt128), true))
    push!(ws.heap, (0.0, 1))
    popped = 0
    while !isempty(ws.heap)
        popped += 1
        if popped % 64 == 0 && (time() > deadline || length(ws.pool) > label_limit)
            return (false, found, min_rc)
        end
        _, li = pop!(ws.heap)
        l = ws.pool[li]
        (l.alive && l.load <= half + 1e-12) || continue
        v = l.node
        charge = is_charger(inst, v) && topb > l.battery + EPS
        b = charge ? topb : l.battery
        lev = charge ? top : 0.0
        @inbounds for w in nbrs[v]
            e = inst.energy[v, w]
            inst.margin * e <= b + EPS || continue
            if w == depot
                v == depot && continue
                restr.active && restr.forbidden[l.last, depot] && continue
                rc = l.cost + ws.cmod[v, w] - κ
                rc < min_rc && (min_rc = rc)
                rc < -rc_tol && push!(found, (rc, li, 0, lev))
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
            cost = l.cost + ws.cmod[v, w] - succession_earning(ws, l.last, w, n)
            sr = l.sr
            if customer && ws.has_sr
                m = ws.sr_mask[w]
                cost += srpen(ws.sr_pen, sr & m)
                sr = sr ⊻ m
            end
            cost + ws.bound[w, capacity_units(ws, inst, load)+1] - κ >= 0.0 && continue
            nb = b - e
            customer && nb + EPS < inst.min_battery_to_leave[w] && continue
            ng = customer ? with(l.ng & ws.ng[w], w) : l.ng
            insert_label!(ws, inst, restr, Label{W}(w, li, lev, cost, l.time + inst.hours[v, w], nb, load, ng,
                          customer ? w : l.last, sr, true), levels, true, true, cap)
        end
    end
    (true, found, min_rc)
end

"""Backward labels from the depot up to half the payload. Returns whether it completed."""
function backward_pass!(bs::BackwardSpace{W}, ws::PricingWorkspace{W}, inst::Instance, restr::Restrictions{W},
                        κ::Float64, cap::Int, topb::Float64, half::Float64, deadline::Float64,
                        label_limit::Int) where {W}
    n, N, depot = inst.n, inst.N, inst.depot
    empty!(bs.pool)
    foreach(empty!, bs.buckets)
    bs.heap = BinaryMinHeap{Tuple{Float64,Int}}()
    push!(bs.pool, BLabel{W}(depot, 0, 0.0, 0.0, 0.0, NodeSet{W}(), depot, zero(UInt128), true))
    push!(bs.heap, (0.0, 1))
    popped = 0
    while !isempty(bs.heap)
        popped += 1
        if popped % 64 == 0 && (time() > deadline || length(bs.pool) > label_limit)
            return false
        end
        _, ui = pop!(bs.heap)
        u_lab = bs.pool[ui]
        (u_lab.alive && u_lab.load <= half + 1e-12) || continue
        u = u_lab.node
        ucharger = is_charger(inst, u)
        @inbounds for v in 1:N
            v == u && continue
            v == depot && continue          # backward labels never close at the start
            inst.usable[v, u] || continue
            e = inst.energy[v, u]
            need = max(inst.margin * e, ucharger ? 0.0 : e + u_lab.need)
            customer = v <= n
            if customer
                v in u_lab.ng && continue
                restr.active && restr.forbidden[v, u_lab.first] && continue
                need <= inst.battery + EPS || continue
                load = u_lab.load + inst.demand[v]
                load <= inst.payload + EPS || continue
            else
                need <= topb + EPS || continue    # a charger is left at the top level
                load = u_lab.load
            end
            cost = u_lab.cost + ws.cmod[v, u] - (customer ? succession_earning(ws, v, u_lab.first, n) : 0.0)
            sr = u_lab.sr
            if customer && ws.has_sr
                m = ws.sr_mask[v]
                cost += srpen(ws.sr_pen, sr & m)
                sr = sr ⊻ m
            end
            # the prefix arriving at v (v included) carries at most payload - (suffix after v)
            after = load - (customer ? inst.demand[v] : 0.0)
            cost + bs.prefix[v, capacity_units(ws, inst, after)+1] - κ >= 0.0 && continue
            ng = customer ? with(u_lab.ng & ws.ng[v], v) : u_lab.ng
            insert_backward!(bs, inst, restr, ws,
                             BLabel{W}(v, ui, cost, load, need, ng, customer ? v : u_lab.first, sr, true), cap)
        end
    end
    true
end

"""
Bidirectional exact (or capped heuristic) pricing of the time-free untimed
relaxation. Same contract as `price!`.
"""
function price_bidirectional!(
    ws::PricingWorkspace{W}, bs::BackwardSpace{W}, inst::Instance, π::Vector{Float64}, κ::Float64,
    restr::Restrictions{W}; exact::Bool, max_columns::Int, rc_tol::Float64, time_limit::Float64,
    label_limit::Int, bucket_cap::Int, cost_scale::Float64, cuts::Vector{CapacityCut}, σ::Vector{Float64},
    sr_cuts::Vector{SubsetRowCut} = SubsetRowCut[], τ::Vector{Float64} = Float64[],
) where {W}
    t0 = time()
    n, N, depot = inst.n, inst.N, inst.depot
    half = inst.payload / 2
    top = maximum(inst.levels)
    topb = top * inst.battery
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
        # backward correction: min over predecessors x of rho[x, a] - rho[x, b]
        @inbounds for a in 1:n+1, b in 1:n+1
            m = 0.0
            if a != b
                for x in 1:n
                    ra = a <= n ? ws.rho[x, a] : 0.0
                    rb = b <= n ? ws.rho[x, b] : 0.0
                    d = ra - rb
                    d < m && (m = d)
                end
            end
            bs.slack[a, b] = m
        end
    else
        ws.earn .= π
    end
    completion_bound!(ws, inst, ws.earn, cost_scale)
    prefix_bound!(bs, ws, inst, ws.earn, cost_scale)
    set_subset_rows!(ws, sr_cuts, τ, n)
    cap = exact ? typemax(Int) : bucket_cap
    nbrs = exact ? inst.out : ws.near
    deadline = t0 + time_limit
    # The two halves share only read-only pricing data, so with more than one
    # Julia thread the backward pass runs alongside the forward one.
    task = Threads.nthreads() > 1 ?
        Threads.@spawn(backward_pass!(bs, ws, inst, restr, κ, cap, topb, half, deadline, label_limit)) : nothing
    fcomplete, found, min_rc = forward_pass!(ws, inst, restr, nbrs, κ, rc_tol, cap, [top], top, topb, half,
                                             deadline, label_limit)
    bcomplete = task === nothing ?
        (fcomplete && backward_pass!(bs, ws, inst, restr, κ, cap, topb, half, deadline, label_limit)) : fetch(task)
    complete = fcomplete && bcomplete

    # ---- joins at customers where the forward load first exceeds C/2 ----
    if complete
        for u in 1:n
            fwd = [k for k in ws.buckets[u] if ws.pool[k].load > half + 1e-12]
            isempty(fwd) && continue
            bwd = sort!([k for k in bs.buckets[u] if bs.pool[k].alive]; by = k -> bs.pool[k].cost)
            isempty(bwd) && continue
            self = nodeset(Val(W), (u,))
            mu = ws.has_sr ? ws.sr_mask[u] : zero(UInt128)
            # u was counted on both sides: a triple holding u was charged once
            # too often when both sides are even; one not holding u is owed a
            # penalty when both are odd. The refund is at most `credit`.
            credit = ws.has_sr ? srpen(ws.sr_pen, mu) : 0.0
            for fk in fwd
                f = ws.pool[fk]
                for bk in bwd
                    g = bs.pool[bk]
                    f.cost + g.cost - κ - credit >= 0.0 && break
                    rc = f.cost + g.cost - κ
                    if ws.has_sr
                        rc += srpen(ws.sr_pen, f.sr & g.sr & ~mu) - srpen(ws.sr_pen, ~f.sr & ~g.sr & mu)
                    end
                    rc >= 0.0 && continue
                    f.load + g.load - inst.demand[u] <= inst.payload + EPS || continue
                    f.battery + EPS >= g.need || continue
                    (f.ng & g.ng) == self || continue
                    rc < min_rc && (min_rc = rc)
                    rc < -rc_tol && push!(found, (rc, fk, bk, 0.0))
                end
            end
        end
    end

    sort!(found; by = first)
    routes = Route[]
    rcs = Float64[]
    seen = Set{UInt64}()
    for (rc, fk, bk, lev) in found
        length(routes) >= max_columns && break
        r = bk == 0 ? rebuild_route(ws, inst, fk, lev) : join_route(ws, bs, inst, fk, bk, top)
        h = hash(r.nodes)
        h in seen && continue
        push!(seen, h)
        push!(routes, r)
        push!(rcs, rc)
    end
    PricingResult(routes, rcs, min_rc, exact && complete, length(ws.pool) + length(bs.pool), time() - t0)
end

function join_route(ws::PricingWorkspace, bs::BackwardSpace, inst::Instance, fk::Int, bk::Int, top::Float64)
    prefix = rebuild_route(ws, inst, fk, 0.0)   # depot ... u, depot (drop the closing depot)
    nodes = prefix.nodes[1:end-1]
    levels = prefix.levels[1:end-1]
    k = bs.pool[bk].parent
    while k != 0
        v = bs.pool[k].node
        push!(nodes, v)
        push!(levels, is_charger(inst, v) ? top : 0.0)
        k = bs.pool[k].parent
    end
    levels[end] = 0.0
    Route(inst, nodes, levels)
end
