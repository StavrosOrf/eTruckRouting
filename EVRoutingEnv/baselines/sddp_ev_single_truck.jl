# Simple SDDP model for a single-truck EV routing problem.
#
# This is a lightweight adaptation of `sddp_vrp.jl` to the deterministic
# environment structure in EVRoutingEnv. It models one truck with a fixed
# delivery order and optional charging before each leg. Uncertainty can be
# injected via travel-time multipliers per stage.
#
# To run:
#   julia --project=. EVRoutingEnv/baselines/sddp_ev_single_truck.jl
# Make sure the JSON files are available relative to this script:
#   EVRoutingEnv/data/shortest_path_energy_dict.json
#   EVRoutingEnv/data/shortest_path_time_dict.json
#   EVRoutingEnv/data/station_info_dict.json

using SDDP
using JuMP
using Gurobi
using JSON
using Statistics
using Random

println("Loading EVRoutingEnv data…")

# Paths (adjust if needed)
DATA_DIR = joinpath(@__DIR__, "..", "data")
energy_path = joinpath(DATA_DIR, "shortest_path_energy_dict.json")
time_path   = joinpath(DATA_DIR, "shortest_path_time_dict.json")
station_path= joinpath(DATA_DIR, "station_info_dict.json")

energy_raw  = JSON.parse(read(energy_path, String))
time_raw    = JSON.parse(read(time_path, String))
station_raw = JSON.parse(read(station_path, String))

# Helper to parse "(i,j)" keys
function parse_tuple_key(s::String)
    s = strip(s, ['(', ')'])
    a,b = split(s, ',')
    return (parse(Int, strip(a)), parse(Int, strip(b)))
end

# Build node map first to see what's available
all_nodes = Set{Int}()
for (k, _) in energy_raw
    i,j = parse_tuple_key(k)
    push!(all_nodes, i); push!(all_nodes, j)
end
for k in keys(station_raw)
    push!(all_nodes, parse(Int, k))
end

# Print available edges for debugging
available_edges = Tuple{Int,Int}[]
for (k, _) in energy_raw
    i,j = parse_tuple_key(k)
    push!(available_edges, (i,j))
end
println("Sample available edges: ", first(available_edges, 5))

# Fixed delivery order - find a route where start has a charger
# This ensures feasibility
delivery_sequence = Int[]
feasible_route_found = false
for (i, j) in available_edges
    has_charger_start = haskey(station_raw, string(i))
    if has_charger_start  # Start node has a charger, so charging is possible
        delivery_sequence = [i, j]
        feasible_route_found = true
        println("Found route with charger at start: $i -> $j")
        break
    end
end

# If no route with charger found, just use first edge (battery should be sufficient)
if !feasible_route_found
    delivery_sequence = [available_edges[1][1], available_edges[1][2]]
    println("No route with charger found, using first available edge")
end

println("Using delivery sequence: ", delivery_sequence)

# Problem data
K = length(delivery_sequence) - 1  # number of legs
battery_cap = 400.0
init_battery = 200.0  # Start with partial charge to allow for charging decisions
efficiency = 0.9
max_charge_h = 24.0

# Travel-time uncertainty: multiplicative factor per stage
travel_factor = [0.9, 1.0, 1.1]  # low / nominal / high
travel_probs  = [0.2, 0.6, 0.2]

# Shortest-path lookups
sp_energy = Dict{Tuple{Int,Int}, Float64}()
for (k, v) in energy_raw
    i,j = parse_tuple_key(k)
    sp_energy[(i,j)] = v
end
sp_time = Dict{Tuple{Int,Int}, Float64}()
for (k, v) in time_raw
    i,j = parse_tuple_key(k)
    sp_time[(i,j)] = v
end

# Charger info (rate assumptions)
function charger_rate(node::Int)
    info = station_raw[string(node)]
    stype = haskey(info, "station_type") ? info["station_type"] : "Level2"
    return stype == "Level2" ? 12.0 : 133.33  # kW (rough placeholders)
end

# Print info about the route
println("Route information:")
for t in 1:K
    i_node = delivery_sequence[t]
    j_node = delivery_sequence[t+1]
    e_leg = sp_energy[(i_node, j_node)]
    t_leg = sp_time[(i_node, j_node)]
    has_charger = haskey(station_raw, string(i_node))
    println("  Stage $t: $i_node -> $j_node, energy=$e_leg kWh, time=$t_leg h, charger=$has_charger")
end

println("Building SDDP policy graph with $K stages…")

model = SDDP.PolicyGraph(
    SDDP.LinearGraph(K),
    sense = :Min,
    lower_bound = 0.0,
    optimizer = Gurobi.Optimizer,
) do sp, t
    # State variables
    @variable(sp, battery, SDDP.State, initial_value = init_battery)
    @variable(sp, clock, SDDP.State, initial_value = 0.0)
    
    # Set bounds on state variables after creation
    set_lower_bound(battery.out, 0.0)
    set_upper_bound(battery.out, battery_cap)
    set_lower_bound(clock.out, 0.0)

    # Decisions
    @variable(sp, 0 <= charge_h <= max_charge_h)  # hours to charge at start of stage
    @variable(sp, use_charge, Bin)                 # 1 if we actually charge

    # Current leg
    i_node = delivery_sequence[t]
    j_node = delivery_sequence[t+1]

    e_leg = sp_energy[(i_node, j_node)]
    t_leg = sp_time[(i_node, j_node)]

    # Charger availability at start node
    has_charger = haskey(station_raw, string(i_node))
    rate = has_charger ? charger_rate(i_node) : 0.0

    # Tie charge_h to use_charge
    @constraint(sp, charge_h <= max_charge_h * use_charge)

    # Battery update with charging then travel
    @constraint(sp, battery.out == battery.in + efficiency * rate * charge_h - e_leg)
    
    # Stochastic travel time multiplier
    # Create a separate variable for the stochastic realized time
    @variable(sp, actual_travel_time >= 0)
    
    SDDP.parameterize(sp, travel_factor, travel_probs) do ω
        # The actual travel time depends on the realization
        JuMP.fix(actual_travel_time, ω * t_leg; force=true)
    end
    
    # Time update uses the realized travel time
    @constraint(sp, clock.out == clock.in + charge_h + actual_travel_time)

    # Objective: minimize expected total time (charging + travel)
    @stageobjective(sp, charge_h + t_leg)
end

println("Solving policy…")
SDDP.train(model; iteration_limit = 200, print_level = 1)

println("Simulating optimal policy…")
sim = SDDP.simulate(model, 20)

# Calculate average time from simulation
total_times = []
for path in sim
    path_time = 0.0
    for node_data in path
        if haskey(node_data, :stage_objective)
            path_time += node_data[:stage_objective]
        end
    end
    push!(total_times, path_time)
end

avg_time = mean(total_times)
println("Average simulated total time over ", length(sim), " paths: ", avg_time)
