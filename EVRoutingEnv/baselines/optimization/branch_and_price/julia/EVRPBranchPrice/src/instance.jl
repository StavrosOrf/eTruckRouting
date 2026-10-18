# Nominal instance read from the exporter's JSON.
#
# Internal indices: customers 1..n, depot n+1, available chargers n+2..n+1+m.
# The depot index doubles as the "no customer yet" succession index, so
# succession pairs live in (1..n+1)^2.
#
# All quantities stay in Float64 hours/kWh and every feasibility test uses the
# environment's own 1e-9 tolerance, so a plan accepted here is accepted by the
# simulator's action mask and vice versa.

const EPS = 1e-9

struct Instance
    n::Int
    m::Int
    N::Int
    depot::Int
    env_id::Vector{Int}
    task_id::Vector{Int}
    demand::Vector{Float64}
    service::Vector{Float64}
    tw_open::Vector{Float64}
    tw_close::Vector{Float64}
    energy::Matrix{Float64}
    hours::Matrix{Float64}
    usable::BitMatrix
    battery::Float64
    payload::Float64
    margin::Float64
    horizon::Float64
    fleet::Int
    start_battery::Float64
    levels::Vector{Float64}
    curve::Vector{Int}
    curves::Vector{Vector{Float64}}
    soc_step::Float64
    ports::Vector{Int}
    out::Vector{Vector{Int}}
    min_hours_to_depot::Vector{Float64}
    min_battery_to_leave::Vector{Float64}
    step_budget::Int
    objective::Symbol
    initial_routes::Vector{Vector{Tuple{Int,Float64,Bool}}}
    meta::Dict{String,Any}
end

@inline is_customer(inst::Instance, i::Int) = i <= inst.n
@inline is_charger(inst::Instance, i::Int) = i > inst.depot

"""Hours of station curve `node` from empty to `soc` (linear between grid points)."""
@inline function cumulative_hours(inst::Instance, node::Int, soc::Float64)
    F = inst.curves[inst.curve[node]]
    x = clamp(soc, 0.0, 1.0) / inst.soc_step
    k = min(floor(Int, x), length(F) - 2)
    @inbounds F[k+1] + (x - k) * (F[k+2] - F[k+1])
end

"""Charging hours at `node` from `battery_kwh` up to SoC `level`."""
@inline charge_hours(inst::Instance, node::Int, battery_kwh::Float64, level::Float64) =
    cumulative_hours(inst, node, level) - cumulative_hours(inst, node, battery_kwh / inst.battery)

_float(x) = x === nothing ? Inf : Float64(x)

function read_instance(path::AbstractString)
    raw = JSON3.read(read(path, String))
    raw.schema == "evrp-bp-instance/v1" || error("unsupported schema $(raw.schema)")
    nodes = raw.nodes
    depot_rows = [k for (k, x) in enumerate(nodes) if x.kind == "depot"]
    customer_rows = [k for (k, x) in enumerate(nodes) if x.kind == "customer"]
    charger_entry = Dict(Int(c.node) => c for c in raw.chargers)
    charger_rows = [k for (k, x) in enumerate(nodes)
                    if x.kind == "charger" && Bool(charger_entry[Int(x.id)].available)]
    length(depot_rows) == 1 || error("exactly one depot is required")
    rows = vcat(customer_rows, depot_rows, charger_rows)
    n, m = length(customer_rows), length(charger_rows)
    N = n + 1 + m
    depot = n + 1

    energy = fill(Inf, N, N)
    hours = fill(Inf, N, N)
    for (i, ri) in enumerate(rows), (j, rj) in enumerate(rows)
        energy[i, j] = _float(raw.energy_kwh[ri][rj])
        hours[i, j] = _float(raw.time_h[ri][rj])
    end

    demand = zeros(N); service = zeros(N); tw_open = zeros(N); tw_close = fill(Inf, N)
    task_id = zeros(Int, n)
    for (i, r) in enumerate(customer_rows)
        x = nodes[r]
        demand[i] = Float64(x.demand)
        service[i] = Float64(x.service_h)
        tw_open[i] = Float64(x.tw[1])
        tw_close[i] = _float(x.tw[2])
        task_id[i] = Int(x.task_id)
    end
    env_id = [Int(nodes[r].id) for r in rows]

    curve = zeros(Int, N); ports = zeros(Int, N)
    for i in depot+1:N
        c = charger_entry[env_id[i]]
        curve[i] = Int(c.curve) + 1
        ports[i] = Int(c.ports)
    end
    curves = [Vector{Float64}(c) for c in raw.charge_curves.cumulative_hours]

    vehicle = raw.vehicle
    battery = Float64(vehicle.battery_kwh)
    socs = Float64.(vehicle.start_soc)
    maximum(socs) - minimum(socs) <= 1e-12 ||
        error("trucks must share one initial SoC (start-SoC vehicle classes are not implemented)")
    start_battery = socs[1] * battery
    margin = Float64(raw.energy_margin)

    usable = falses(N, N)
    for i in 1:N, j in 1:N
        i == j && continue
        i == depot && j == depot && continue
        e, h = energy[i, j], hours[i, j]
        (isfinite(e) && isfinite(h) && e >= 0 && h >= 0) || continue
        departure = i == depot ? start_battery : battery
        margin * e <= departure + EPS || continue
        usable[i, j] = true
    end
    out = [Int[j for j in 1:N if usable[i, j]] for i in 1:N]

    initial_routes = Vector{Tuple{Int,Float64,Bool}}[]
    for route in vcat(collect(raw.warm_start.witness), collect(raw.warm_start.extra))
        push!(initial_routes, [(Int(s[1]), Float64(s[2]), Bool(s[3])) for s in route])
    end

    meta = Dict{String,Any}(String(k) => v for (k, v) in pairs(raw.meta))
    inst = Instance(
        n, m, N, depot, env_id, task_id, demand, service, tw_open, tw_close,
        energy, hours, usable, battery, Float64(vehicle.payload), margin,
        Float64(raw.horizon_h), Int(vehicle.count), start_battery,
        sort(Float64.(raw.charge_levels)), curve, curves,
        Float64(raw.charge_curves.soc_step), ports, out,
        zeros(N), zeros(N), Int(raw.step_budget), Symbol(raw.objective),
        initial_routes, meta,
    )
    inst.objective == :travel || error("only objective = travel is implemented")
    _fill_bounds!(inst)
    inst
end

# Two instance-wide bounds used for pruning, both dual independent:
# * shortest hours to the depot over usable arcs (a route cannot finish sooner);
# * the least battery that lets a truck leave a customer and reach a charger
#   or the depot, with the per-leg margin (a label below it is a dead end).
function _fill_bounds!(inst::Instance)
    N, depot = inst.N, inst.depot
    dist = fill(Inf, N); dist[depot] = 0.0
    done = falses(N)
    for _ in 1:N
        v, best = 0, Inf
        for i in 1:N
            if !done[i] && dist[i] < best
                v, best = i, dist[i]
            end
        end
        v == 0 && break
        done[v] = true
        for u in 1:N
            if inst.usable[u, v] && u != depot
                dist[u] = min(dist[u], best + inst.hours[u, v])
            end
        end
    end
    inst.min_hours_to_depot .= dist

    need = [is_customer(inst, i) ? Inf : 0.0 for i in 1:N]
    changed = true
    while changed
        changed = false
        for w in 1:inst.n, j in inst.out[w]
            e = inst.energy[w, j]
            candidate = max(inst.margin * e, e + need[j])
            if candidate < need[w] - 1e-12
                need[w] = candidate
                changed = true
            end
        end
    end
    inst.min_battery_to_leave .= need
    inst
end

"""ng neighbourhoods: each customer and its `size - 1` nearest customers by round-trip hours."""
function ng_neighbourhoods(inst::Instance, ::Val{W}, size::Int) where {W}
    n = inst.n
    map(1:n) do i
        others = sort([j for j in 1:n if j != i]; by = j -> inst.hours[i, j] + inst.hours[j, i])
        nodeset(Val(W), vcat(i, others[1:min(size - 1, length(others))]))
    end
end
