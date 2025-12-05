using SDDP
using JuMP
using Gurobi
using JSON
using DataFrames
using Statistics
using Random
using Plots
using CSV

println("Loading data for EV VRP SDDP model…")

# ─── 1) Read input JSON ─────────────────────────────────────────────────────────
waiting_time_raw = JSON.parse(read("json_data/waiting_time_dict.json", String))
station_info_raw = JSON.parse(read("json_data/station_info_dict.json", String))
time_data_raw    = JSON.parse(read("json_data/shortest_path_time_dict.json", String))
energy_data_raw  = JSON.parse(read("json_data/shortest_path_energy_dict.json", String))

# ─── 2) Helper: parse "(i,j)" → Tuple{Int,Int} ─────────────────────────────────
function parse_tuple_key(s::String)
    s = strip(s, ['(', ')'])
    a,b = split(s, ',')
    return (parse(Int, strip(a)), parse(Int, strip(b)))
end

# ─── 3) Build sparse‐ID mapping for all "long" node IDs ────────────────────────
route_sequence = [
    [5026447875, 65657291, 5433392625],
    [54382864, 90796641],
    [9512913929, 49291774, 90810515],
    [9512913929, 90539746, 90403004],
    [9512913929, 353478871, 9509748626]
    # [9512913929, 258148925, 10920046287, 90796641],
    # [9512913929, 62268500, 62459745, 2565726584],
    # [9512913929, 90403004, 370274261, 90539746],
    # [9512913929, 65463154, 65541114, 65657291],
    # [9512913929, 88719427, 6054431549, 87505264]
    # [9512913929, 10007056897, 11962070326],
    # [9512913929, 239637816, 415194042, 338840749],
    # [9512913929, 6270648007, 56166715],
    # [9512913929, 91912624, 85940419],
    # [9512913929, 4924867069, 62283196, 242460445]
    # [9512913929, 11136422978, 86884900],
    # [9512913929, 9509748626, 90767678, 90427378],
    # [9512913929, 65317565, 65422401],
    # [9512913929, 65657291, 53104575, 353478871],
    # [9512913929, 65657291]
]
V = length(route_sequence)
all_long = Set{Int}()
for seq in route_sequence, n in seq
    push!(all_long, n)
end
for k in keys(station_info_raw)
    push!(all_long, parse(Int,k))
end
for k in keys(time_data_raw)
    i,j = parse_tuple_key(k)
    push!(all_long, i); push!(all_long, j)
end
long2idx = Dict(n => i for (i,n) in enumerate(sort(collect(all_long))))
idx2long = Dict(i => n for (n,i) in long2idx)
N = 1:length(long2idx)

# ─── 4) Remap routes & station set ─────────────────────────────────────────────
route_sequence = [ [ long2idx[n] for n in seq ] for seq in route_sequence ]
S = Set(long2idx[parse(Int,k)] for k in keys(station_info_raw))

# ─── 5) Remap shortest‐path data ───────────────────────────────────────────────
sp_time   = Dict{Tuple{Int,Int},Float64}()
sp_energy = Dict{Tuple{Int,Int},Float64}()
for (k,v) in time_data_raw
    i,j = parse_tuple_key(k)
    sp_time[(long2idx[i], long2idx[j])]   = v
end
for (k,v) in energy_data_raw
    i,j = parse_tuple_key(k)
    sp_energy[(long2idx[i], long2idx[j])] = v
end

println("   sp_time entries:   ", length(sp_time))
println("   sp_energy entries: ", length(sp_energy))
println("   Sample sp_time / sp_energy:")
for ((i,j), _) in first(sp_time, 3)
    println("     time[$i,$j] = ", sp_time[(i,j)],
            ", energy[$i,$j] = ", sp_energy[(i,j)])
end

# ─── 6) Remap waiting‐time dict ────────────────────────────────────────────────
waiting_time = Dict{Int, Dict{String, Float64}}()
for (k, inner) in waiting_time_raw
    sid = long2idx[parse(Int, k)]
    waiting_time[sid] = Dict{String, Float64}(
        level => parse(Float64, string(wait))
        for (level, wait) in inner
    )
    println("Station $sid waiting times: ", waiting_time[sid])
end

println("Total charging stations with waiting time entries: ", length(waiting_time))

# ─── 7) Calculate stages (route legs) ─────────────────────────────────────────
vehicle_stages = Dict(v => length(route_sequence[v]) - 1 for v in 1:V)
max_stages = maximum(values(vehicle_stages))

println("Vehicle stages: ", vehicle_stages)
println("Max stages: ", max_stages)

# ─── 8) Build SDDP scenario tree structure ────────────────────────────────────
# Create a proper SDDP structure with multiple scenarios per stage
# This matches the approach in sddp_3.jl for proper stochastic programming
graph = SDDP.Graph("root")
num_scenarios = 5  # Number of scenarios per stage

# Create scenario nodes for each stage
for stage in 1:max_stages
    for scenario in 1:num_scenarios
        node_name = "stage_$(stage)_scenario_$(scenario)"
        SDDP.add_node(graph, node_name)
        
        # Connect to parent
        if stage == 1
            # First stage: connect to root
            SDDP.add_edge(graph, "root" => node_name, 1.0/num_scenarios)
        else
            # Other stages: connect to all scenarios from previous stage
            for prev_scenario in 1:num_scenarios
                parent_name = "stage_$(stage-1)_scenario_$(prev_scenario)"
                SDDP.add_edge(graph, parent_name => node_name, 1.0/num_scenarios)
            end
        end
    end
end

println("Built SDDP structure with $(length(graph.nodes)) nodes")
println("Total possible paths: $(num_scenarios^max_stages)")
println("SDDP.jl will sample one scenario path per iteration from this tree")

# ─── 9) Build & solve the SDDP.PolicyGraph ───────────────────────────────────
function build_ev_vrp_sddp_model()
    SDDP.PolicyGraph(
        (sp, node) -> begin
            # Extract stage and scenario from node name (e.g., "stage_1_scenario_3" -> stage=1, scenario=3)
            parts = split(node, "_")
            stage = parse(Int, parts[2])
            scenario = parse(Int, parts[4])
            
            println("→ Subproblem $node (stage=$stage, scenario=$scenario)")
            
            # Dynamic sampling from waiting_time_dict using SDDP.jl's capabilities
            # Use both stage and scenario to create diverse but realistic uncertainty patterns
            stage_waiting_times = Dict{Int, Float64}()
            
            # Define transition probabilities for congestion levels (Markov chain)
            # This creates realistic evolution of congestion over time
            congestion_levels = ["low", "busy", "full"]
            
            for s in S
                # Use a combination of stage, scenario, and station to create diverse sampling
                # This ensures each scenario-stage-station combination gets different realizations
                seed_value = hash("$(stage)_$(scenario)_$(s)")
                Random.seed!(seed_value)
                
                # Sample from congestion levels with realistic probabilities
                r = rand()
                if r <= 0.4
                    level = "low"
                elseif r <= 0.7
                    level = "busy"
                else
                    level = "full"
                end
                
                wait_t = waiting_time[s][level]
                stage_waiting_times[s] = wait_t
            end
            
            println("  Stage $stage, Scenario $scenario: Using dynamic scenario sampling for $(length(stage_waiting_times)) stations")
            
            # Show a few sample waiting times for verification
            sample_stations = collect(keys(stage_waiting_times))[1:min(3, length(stage_waiting_times))]
            for station in sample_stations
                println("    Station $station: $(stage_waiting_times[station])")
            end

            # 1) State variables
            @variable(sp, curr_pos[v=1:V], Int, SDDP.State, domain=N, initial_value=route_sequence[v][1])
            @variable(sp, arrival_time[v=1:V], SDDP.State, initial_value=0)
            @variable(sp, arrival_energy[v=1:V], SDDP.State, initial_value=600)
            @variable(sp, energy_slack[v=1:V], SDDP.State, initial_value=0)

            # 2) Decision variables
            # Edge selection: x[v,i,j] = 1 if vehicle v travels from i to j
            @variable(sp, x[v=1:V, i in N, j in N], Bin)
            
            # Node visit: y[v,i] = 1 if vehicle v visits node i
            @variable(sp, y[v=1:V, i in N], Bin)
            
            # Charging time at stations
            @variable(sp, charging_time[v=1:V, s in S] ≥ 0)
            
            # Energy at each node
            @variable(sp, node_energy[v=1:V, i in N] ≥ 0, upper_bound=600.0)
            
            # Energy slack for this stage (additional energy needed beyond what's available)
            @variable(sp, energy_slack_edges[v=1:V, i in N, j in N] ≥ 0)
            
            # Time at each node
            @variable(sp, node_time[v=1:V, i in N] ≥ 0)

            # 3) Parameters
            # Average charging rates
            avg_charging_rate = Dict{Int, Float64}()
            for s in S
                station_type = station_info_raw[string(idx2long[s])]["station_type"]
                avg_charging_rate[s] = station_type == "Level2" ? 12.0 : 133.33
            end

            # Scale factors for better numerical stability
            time_scale = 1.0
            energy_scale = 1.0

            # 4) Constraints
            
            # Disallow self-loops (i→i)
            for v in 1:V, i in N
                @constraint(sp, x[v, i, i] == 0)
            end

            # Proper flow conservation: outflow at start, inflow at end, balance at intermediate nodes
            # Only charging stations are allowed as intermediate nodes
            for v in 1:V, i in N
                route = route_sequence[v]
                # Use stage directly (known from SDDP node) instead of curr_pos[v].in
                if stage <= vehicle_stages[v]
                    start_node = route[stage]
                    end_node = route[stage + 1]
                    
                    if i == start_node
                        # Start node: outflow = 1, inflow = 0
                        @constraint(sp, sum(x[v, i, j] for j in N) == 1)
                        @constraint(sp, sum(x[v, j, i] for j in N) == 0)
                    elseif i == end_node
                        # End node: inflow = 1, outflow = 0 (stop at destination)
                        @constraint(sp, sum(x[v, i, j] for j in N) == 0)
                        @constraint(sp, sum(x[v, j, i] for j in N) == 1)
                    elseif i in S
                        # Charging stations: inflow = outflow (can be used as intermediate nodes)
                        @constraint(sp, sum(x[v, j, i] for j in N if j != i) == sum(x[v, i, k] for k in N if k != i))
                    else
                        # Non-charging stations: cannot be used as intermediate nodes
                        @constraint(sp, sum(x[v, j, i] for j in N) == 0)
                        @constraint(sp, sum(x[v, i, j] for j in N) == 0)
                    end
                end
            end

            # # Limit station visits (at most 2 charging stops per leg)
            # for v in 1:V
            #     @constraint(sp, sum(y[v, s] for s in S) ≤ 2)
            # end

            # Link visit variables to edge variables
            for v in 1:V, i in N
                @constraint(sp, y[v, i] ≥ sum(x[v, i, j] for j in N))
                @constraint(sp, y[v, i] ≥ sum(x[v, j, i] for j in N))
                @constraint(sp, y[v, i] ≤ sum(x[v, i, j] for j in N) + sum(x[v, j, i] for j in N))
            end

            # Energy constraints
            for v in 1:V
                route = route_sequence[v]
                # Use stage directly instead of curr_pos[v].in
                if stage <= vehicle_stages[v]
                    start_node = route[stage]
                    # Initial energy at start node
                    @constraint(sp, node_energy[v, start_node] == arrival_energy[v].in)
                    # Initial time at start node
                    @constraint(sp, node_time[v, start_node] == arrival_time[v].in)
                end
            end

            M = 10000  # Big-M 

            # Energy flow along edges with Big-M constraints (reverted from indicator for compatibility)
            for v in 1:V, i in N, j in N
                if i != j  # Skip self-loops
                    energy_cost = get(sp_energy, (i, j), 0.0)
                    if i in S
                        @constraint(sp, node_energy[v, j] ≥ node_energy[v, i] - energy_cost + avg_charging_rate[i] * charging_time[v, i] + energy_slack_edges[v, i, j] - M * (1 - x[v, i, j]))
                        @constraint(sp, node_energy[v, j] ≤ node_energy[v, i] - energy_cost + avg_charging_rate[i] * charging_time[v, i] + energy_slack_edges[v, i, j] + M * (1 - x[v, i, j]))
                    else
                        @constraint(sp, node_energy[v, j] ≥ node_energy[v, i] - energy_cost + energy_slack_edges[v, i, j] - M * (1 - x[v, i, j]))
                        @constraint(sp, node_energy[v, j] ≤ node_energy[v, i] - energy_cost + energy_slack_edges[v, i, j] + M * (1 - x[v, i, j]))
                    end
                end
            end

            # Energy sufficiency constraint: if we travel on edge (i,j), we must have enough energy
            for v in 1:V, i in N, j in N
                if i != j  # Skip self-loops
                    energy_cost = get(sp_energy, (i, j), 0.0)
                    # If x[v,i,j] = 1, then we must have enough energy to travel
                    @constraint(sp, node_energy[v, i] + energy_slack_edges[v, i, j] ≥ energy_cost - M * (1 - x[v, i, j]))
                end
            end

            # Energy slack constraint: slack can only be used when edge is actually used
            for v in 1:V, i in N, j in N
                if i != j  # Skip self-loops
                    @constraint(sp, energy_slack_edges[v, i, j] ≤ M * x[v, i, j])
                end
            end

            # Time flow along edges with Big-M constraints
            for v in 1:V, i in N, j in N
                if i != j  # Skip self-loops
                    travel_time = get(sp_time, (i, j), 0.0)
                    if i in S
                        wait = get(stage_waiting_times, i, 0.0)
                        @constraint(sp, node_time[v, j] ≥ node_time[v, i] + travel_time + charging_time[v, i] + wait - M * (1 - x[v, i, j]))
                        @constraint(sp, node_time[v, j] ≤ node_time[v, i] + travel_time + charging_time[v, i] + wait + M * (1 - x[v, i, j]))
                    else
                        @constraint(sp, node_time[v, j] ≥ node_time[v, i] + travel_time - M * (1 - x[v, i, j]))
                        @constraint(sp, node_time[v, j] ≤ node_time[v, i] + travel_time + M * (1 - x[v, i, j]))
                    end
                end
            end

            # Charging time constraints: can only charge if we visit the station
            for v in 1:V, s in S

                @constraint(sp, charging_time[v, s] ≤ M * y[v, s])
            end

            # State transitions
            for v in 1:V
                route = route_sequence[v]
                # Use stage directly instead of curr_pos[v].in
                if stage < vehicle_stages[v]
                    # Must progress to next route node
                    @constraint(sp, curr_pos[v].out == stage + 1)
                else
                    # Already at the end, stay there
                    @constraint(sp, curr_pos[v].out == stage)
                end
            end

            # Final energy and time assignments
            for v in 1:V
                route = route_sequence[v]
                # Use stage directly instead of curr_pos[v].in
                if stage <= vehicle_stages[v]
                    end_node = route[stage + 1]
                    # Direct assignment to node_energy at end node
                    @constraint(sp, arrival_energy[v].out == node_energy[v, end_node])
                    # Direct assignment to node_time at end node
                    @constraint(sp, arrival_time[v].out == node_time[v, end_node])
                end
            end

            # Final energy slack: sum all energy slack used by this vehicle in this stage
            for v in 1:V
                # Pre-compute the sum more efficiently
                slack_sum = sum(energy_slack_edges[v, i, j] for i in N for j in N if i != j; init=0.0)
                @constraint(sp, energy_slack[v].out == energy_slack[v].in + slack_sum)
            end

            # Energy budgeting: encourage conservation in early stages
            for v in 1:V
                route = route_sequence[v]
                if stage <= vehicle_stages[v]
                    end_node = route[stage + 1]
                    # If this is not the final stage, encourage energy conservation
                    if stage < vehicle_stages[v]
                        # Estimate energy needed for remaining stages
                        remaining_energy_needed = 0.0
                        for next_stage in (stage+1):vehicle_stages[v]
                            if next_stage <= vehicle_stages[v]
                                start_idx = next_stage
                                end_idx = next_stage + 1
                                if start_idx <= length(route) && end_idx <= length(route)
                                    start_node = route[start_idx]
                                    end_node_next = route[end_idx]
                                    energy_cost = get(sp_energy, (start_node, end_node_next), 0.0)
                                    remaining_energy_needed += energy_cost
                                end
                            end
                        end
                        # Encourage keeping enough energy for future stages (with safety buffer)
                        energy_budget = max(60.0, min(500.0, remaining_energy_needed))
                        @constraint(sp, arrival_energy[v].out >= energy_budget)
                    end
                end
            end

            # 5) Stage objective: minimize total costs + future value approximation
            @stageobjective(sp, 
                # 1) Total travel time over all vehicles & edges
                sum(get(sp_time, (i, j), 0.0) * x[v, i, j] for v in 1:V for i in N for j in N if i != j)
                +
                # 2) Total waiting time at charging stations
                sum(stage_waiting_times[s] * y[v, s] for v in 1:V for s in S if haskey(stage_waiting_times, s))
                +
                # 3) Total charging time
                sum(charging_time[v, s] for v in 1:V for s in S)
                +
                # 4) Total energy slack penalty
                sum(energy_slack_edges[v, i, j] for v in 1:V for i in N for j in N if i != j)
            )
        end,    
        graph,
        sense=:Min, lower_bound=0.0, optimizer=Gurobi.Optimizer
    )
end

function solve_and_record(model;
        max_iterations=50,
        mc_rep=50)

    ### 1) TRAINING history
    counter   = Ref(0)
    iters     = Int[]
    vf_bounds = Float64[]

    function record_iter(res::SDDP.IterationResult)
        counter[] += 1
        it = counter[]
        push!(iters, it)
        push!(vf_bounds, res.bound)
    end

    # Set the global seed for reproducibility (should match Python's seed)
    Random.seed!(42)

    SDDP.train(model;
        iteration_limit         = max_iterations,
        time_limit              = 21600,
        print_level             = 2,
        log_file                = "SDDP_simple.log",
        refine_at_similar_nodes = true,
        cut_deletion_minimum    = 5,
        # parallel_scheme         = SDDP.Threaded(),
        post_iteration_callback = record_iter,
        sampling_scheme         = SDDP.InSampleMonteCarlo()  # Use in-sample Monte Carlo for proper scenario sampling
        # duality_handler         = SDDP.LagrangianDuality()
    )

    extract_decisions_from_sample_path(model)
    
end

function extract_decisions_from_sample_path(model)
    println("Running in-sample MC replications…")
    Random.seed!(42)
    simulations = SDDP.simulate(model, 1, [:x, :charging_time, :y]; sampling_scheme=SDDP.InSampleMonteCarlo())
    
    println("Extracting decisions from final trained model...")
    if length(simulations) > 0
        scenario = simulations[1]
        
        println("\n=== DECISION EXTRACTION ===")
        total_cost = 0.0
        solution = Dict{String, Any}()
        solution["stages"] = []
        
        for (stage_idx, stage_data) in enumerate(scenario)
            println("\nStage $stage_idx:")
            stage_dict = Dict{String, Any}()
            
            # ROUTING DECISIONS x
            x_nonzero = []
            if haskey(stage_data, :x)
                # println("  [DEBUG] x data type: $(typeof(stage_data[:x]))")
                # x_keys = collect(keys(stage_data[:x]))
                # println("  [DEBUG] x data keys: $(first(x_keys, min(5, length(x_keys))))")
                
                # Handle JuMP DenseAxisArrayKey format
                for key in keys(stage_data[:x])
                    val = stage_data[:x][key]
                    if abs(val) > 1e-6
                        try
                            # Extract from JuMP key format: DenseAxisArrayKey{Tuple{Int64, Tuple{Int64, Int64}}}((v, (i, j)))
                            # The key itself contains the tuple data
                            v = key[1]
                            i = key[2]
                            j = key[3]
                            orig_i, orig_j = idx2long[i], idx2long[j]
                            println("    Vehicle $v: $orig_i → $orig_j (value: $val)")
                            push!(x_nonzero, (v, orig_i, orig_j, val))
                        catch e
                            println("[ERROR x] Key: $key, Value: $val, Error: $e")
                        end
                    end
                end
            end
            stage_dict["x"] = x_nonzero
            
            # CHARGING TIME DECISIONS
            charging_nonzero = []
            if haskey(stage_data, :charging_time)
                # println("  [DEBUG] charging_time data type: $(typeof(stage_data[:charging_time]))")
                # charging_keys = collect(keys(stage_data[:charging_time]))
                # println("  [DEBUG] charging_time data keys: $(first(charging_keys, min(5, length(charging_keys))))")
                
                # Handle JuMP DenseAxisArrayKey format
                for key in keys(stage_data[:charging_time])
                    val = stage_data[:charging_time][key]
                    if abs(val) > 1e-6
                        try
                            # Extract from JuMP key format: DenseAxisArrayKey{Tuple{Int64, Int64}}((v, s))
                            # The key itself contains the tuple data
                            v = key[1]
                            s = key[2]
                            orig_s = idx2long[s]
                            println("    Vehicle $v at station $orig_s: $(round(val, digits=2)) time units")
                            push!(charging_nonzero, (v, orig_s, val))
                        catch e
                            println("[ERROR charging_time] Key: $key, Value: $val, Error: $e")
                        end
                    end
                end
            end
            stage_dict["charging_time"] = charging_nonzero
            
            # VISITS y
            y_nonzero = []
            if haskey(stage_data, :y)
                # println("  [DEBUG] y data type: $(typeof(stage_data[:y]))")
                # y_keys = collect(keys(stage_data[:y]))
                # println("  [DEBUG] y data keys: $(first(y_keys, min(5, length(y_keys))))")
                
                # Handle JuMP DenseAxisArrayKey format
                for key in keys(stage_data[:y])
                    val = stage_data[:y][key]
                    if abs(val) > 1e-6
                        try
                            # Extract from JuMP key format: DenseAxisArrayKey{Tuple{Int64, Int64}}((v, i))
                            # The key itself contains the tuple data
                            v = key[1]
                            i = key[2]
                            orig_i = idx2long[i]
                            println("    Vehicle $v visited node $orig_i (value: $val)")
                            push!(y_nonzero, (v, orig_i, val))
                        catch e
                            println("[ERROR y] Key: $key, Value: $val, Error: $e")
                        end
                    end
                end
            end
            stage_dict["y"] = y_nonzero
            
            # STAGE OBJECTIVE
            if haskey(stage_data, :stage_objective)
                stage_cost = stage_data[:stage_objective]
                total_cost += stage_cost
                println("  Stage objective: $(round(stage_cost, digits=2))")
                stage_dict["stage_objective"] = stage_cost
            end
            
            push!(solution["stages"], stage_dict)
        end
        
        println("\nTotal scenario cost: $(round(total_cost, digits=2))")
        solution["total_cost"] = total_cost
        
        open("sample_solution.json", "w") do io
            JSON.print(io, solution, 2)
        end
        println("Sample solution saved to sample_solution.json")
    else
        println("No scenario path found for decision extraction")
    end
end

# ─── Run if script is called directly ─────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    # Set the same seed as Python for reproducibility
    Random.seed!(42)
    
    model = build_ev_vrp_sddp_model()
    solve_and_record(model; max_iterations=30, mc_rep=1)
    
end 