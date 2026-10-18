# A route is the node sequence depot -> ... -> depot plus, for every position,
# the SoC level charged there before leaving (0.0 means no charge). It is
# simulated under the environment's nominal rules, which is both the final
# feasibility check and the source of the reported schedule.

struct Route
    nodes::Vector{Int}
    levels::Vector{Float64}
    travel::Float64
    customers::Vector{Int}
end

function Route(inst::Instance, nodes::Vector{Int}, levels::Vector{Float64})
    travel = sum(inst.hours[nodes[k], nodes[k+1]] for k in 1:length(nodes)-1; init = 0.0)
    Route(nodes, levels, travel, [v for v in nodes if is_customer(inst, v)])
end

route_key(r::Route) = hash((r.nodes, r.levels))

"""Customer successions of a route, depot as n+1 at both ends."""
function successions(inst::Instance, r::Route)
    seq = vcat(inst.depot, r.customers, inst.depot)
    [(seq[k], seq[k+1]) for k in 1:length(seq)-1]
end

struct Stop
    node::Int
    arrival::Float64
    arrival_battery::Float64
    level::Float64
    charge_hours::Float64
    departure::Float64
    departure_battery::Float64
    load_after::Float64
end

"""Simulate `r`; returns (feasible, reason, stops)."""
function simulate(inst::Instance, r::Route)
    nodes, levels = r.nodes, r.levels
    stops = Stop[]
    (first(nodes) == inst.depot && last(nodes) == inst.depot) || return (false, "must start and end at the depot", stops)
    t, b, load = 0.0, inst.start_battery, 0.0
    arrival, arrival_b = 0.0, b
    seen = Set{Int}()
    for k in eachindex(nodes)
        v = nodes[k]
        level = levels[k]
        charged = 0.0
        if level > 0
            is_charger(inst, v) || return (false, "charge at non-charger $(v)", stops)
            if level * inst.battery > b + EPS
                charged = charge_hours(inst, v, b, level)
                t += charged
                b = level * inst.battery
            else
                level = 0.0  # already at or above target: the replay skips it
            end
        end
        push!(stops, Stop(v, arrival, arrival_b, level, charged, t, b, load))
        k == length(nodes) && break
        w = nodes[k+1]
        inst.usable[v, w] || return (false, "unusable leg $(v)->$(w)", stops)
        inst.margin * inst.energy[v, w] <= b + EPS || return (false, "insufficient energy $(v)->$(w)", stops)
        b -= inst.energy[v, w]
        t += inst.hours[v, w]
        arrival, arrival_b = t, b
        if is_customer(inst, w)
            w in seen && return (false, "customer $(w) visited twice", stops)
            push!(seen, w)
            t <= inst.tw_close[w] + EPS || return (false, "window of $(w) missed", stops)
            load += inst.demand[w]
            load <= inst.payload + EPS || return (false, "payload exceeded", stops)
            t = max(t, inst.tw_open[w]) + inst.service[w]
        elseif w == inst.depot
            t <= inst.horizon + EPS || return (false, "horizon exceeded", stops)
        end
    end
    (true, "", stops)
end

"""
Feasible charging plan for the node sequence of `r`, or `nothing`.

Charging to the top level at every charger is tried first (it leaves the most
energy margin for stochastic execution); otherwise a Pareto search over
(time, battery) per stop finds the earliest-finishing feasible plan. Travel
cost depends only on the node sequence, so either result keeps the cost.
"""
function schedule_route(inst::Instance, r::Route)
    top = maximum(inst.levels)
    full = Route(r.nodes, [is_charger(inst, v) ? top : 0.0 for v in r.nodes], r.travel, r.customers)
    simulate(inst, full)[1] && return full
    simulate(inst, r)[1] && return r
    # states: (time, battery, levels so far)
    states = [(0.0, inst.start_battery, Float64[])]
    nodes = r.nodes
    for k in 1:length(nodes)-1
        v, w = nodes[k], nodes[k+1]
        expanded = Tuple{Float64,Float64,Vector{Float64}}[]
        for (t, b, lv) in states
            push!(expanded, (t, b, vcat(lv, 0.0)))
            if is_charger(inst, v)
                for L in inst.levels
                    L * inst.battery > b + EPS || continue
                    push!(expanded, (t + charge_hours(inst, v, b, L), L * inst.battery, vcat(lv, L)))
                end
            end
        end
        moved = Tuple{Float64,Float64,Vector{Float64}}[]
        for (t, b, lv) in expanded
            inst.margin * inst.energy[v, w] <= b + EPS || continue
            nt, nb = t + inst.hours[v, w], b - inst.energy[v, w]
            if is_customer(inst, w)
                nt <= inst.tw_close[w] + EPS || continue
                nt = max(nt, inst.tw_open[w]) + inst.service[w]
            end
            nt + inst.min_hours_to_depot[w] <= inst.horizon + EPS || continue
            push!(moved, (nt, nb, lv))
        end
        sort!(moved; by = s -> (s[1], -s[2]))
        states = Tuple{Float64,Float64,Vector{Float64}}[]
        best_b = -Inf
        for s in moved
            if s[2] > best_b + 1e-12
                push!(states, s)
                best_b = s[2]
            end
        end
        isempty(states) && return nothing
    end
    t, _, lv = first(states)
    candidate = Route(nodes, vcat(lv, 0.0), r.travel, r.customers)
    simulate(inst, candidate)[1] ? candidate : nothing
end

"""
Cheapest single-customer route for every customer (depot -> chargers -> i ->
chargers -> depot, charging fully at each charger), when one exists. Seeding
the master with them bounds every cover dual by a real route cost from the
first iteration; without them the duals of a sparse master are bounded only
by the artificial cost and no completion bound can prune.
"""
function singleton_routes(inst::Instance)
    N, n, depot = inst.N, inst.n, inst.depot
    top = maximum(inst.levels) * inst.battery
    chargers = collect(depot+1:N)
    leaves(v) = v == depot ? inst.start_battery : top
    ok(v, w, battery) = inst.usable[v, w] && inst.margin * inst.energy[v, w] <= battery + EPS
    # shortest travel depot -> charger (leaving it full) and charger -> depot over charger hops
    function dijkstra(sources::Dict{Int,Float64}, forward::Bool)
        dist = Dict{Int,Float64}(sources)
        prev = Dict{Int,Int}()
        done = Set{Int}()
        while true
            open = [(d, v) for (v, d) in dist if !(v in done)]
            isempty(open) && break
            d, v = minimum(open)
            push!(done, v)
            for w in chargers
                w in done && continue
                (a, b) = forward ? (v, w) : (w, v)
                ok(a, b, leaves(a)) || continue
                nd = d + inst.hours[a, b]
                if nd < get(dist, w, Inf)
                    dist[w] = nd
                    prev[w] = v
                end
            end
        end
        dist, prev
    end
    to_dist, to_prev = dijkstra(Dict(c => inst.hours[depot, c] for c in chargers if ok(depot, c, inst.start_battery)), true)
    from_dist, from_prev = dijkstra(Dict(c => inst.hours[c, depot] for c in chargers if ok(c, depot, top)), false)
    to_dist[depot] = 0.0
    from_dist[depot] = 0.0
    chain_to(c) = c == depot ? [depot] : (haskey(to_prev, c) ? vcat(chain_to(to_prev[c]), c) : [depot, c])
    chain_from(c) = c == depot ? [depot] : (haskey(from_prev, c) ? vcat(c, chain_from(from_prev[c])) : [c, depot])
    routes = Route[]
    stops = vcat(depot, chargers)
    for i in 1:n
        best, pick = Inf, (0, 0)
        for a in stops
            haskey(to_dist, a) || continue
            ok(a, i, leaves(a)) || continue
            arrive = leaves(a) - inst.energy[a, i]
            arrive + EPS >= inst.min_battery_to_leave[i] || continue
            for b in stops
                haskey(from_dist, b) || continue
                ok(i, b, arrive) || continue
                value = to_dist[a] + inst.hours[a, i] + inst.hours[i, b] + from_dist[b]
                value < best && ((best, pick) = (value, (a, b)))
            end
        end
        isfinite(best) || continue
        nodes = vcat(chain_to(pick[1]), i, chain_from(pick[2]))
        r = Route(inst, nodes, [is_charger(inst, v) ? maximum(inst.levels) : 0.0 for v in nodes])
        scheduled = schedule_route(inst, r)
        scheduled === nothing || push!(routes, scheduled)
    end
    routes
end

"""Environment witness steps -> internal route (unused trucks give `nothing`)."""
function route_from_witness(inst::Instance, steps::Vector{Tuple{Int,Float64,Bool}})
    index = Dict(id => i for (i, id) in enumerate(inst.env_id))
    nodes = [inst.depot]
    levels = [0.0]
    for (id, value, charging) in steps
        haskey(index, id) || return nothing
        if charging
            levels[end] = value
        else
            push!(nodes, index[id]); push!(levels, 0.0)
        end
    end
    length(nodes) <= 2 && return nothing
    Route(inst, nodes, levels)
end

"""Internal route -> environment witness steps (node id, value, is_charge)."""
function witness_steps(inst::Instance, r::Route)
    steps = Tuple{Int,Float64,Bool}[]
    for k in 1:length(r.nodes)-1
        r.levels[k] > 0 && push!(steps, (inst.env_id[r.nodes[k]], r.levels[k], true))
        push!(steps, (inst.env_id[r.nodes[k+1]], 0.0, false))
    end
    steps
end
