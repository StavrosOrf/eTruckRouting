using JuMP
using Gurobi
using Suppressor
using StatsBase: mean
using Printf

function generate_artificial_paths(
    data::EVRPData,
)
    artificial_paths = Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    }()
    start_depots = zeros(Int, data.n_vehicles)
    for (k, v_list) in pairs(data.V)
        for v in v_list
            start_depots[v] = k
        end
    end
    end_depots = Int[]
    for k in data.N_depots
        append!(end_depots, repeat([k], data.v_end[k]))
    end
    append!(end_depots,
        repeat(
            data.N_depots, 
            outer = Int(ceil((data.n_vehicles - sum(values(data.v_end))) / data.n_depots))
        )
    )
    end_depots = end_depots[1:data.n_vehicles]
    
    for (v, (starting_node, current_node)) in enumerate(zip(start_depots, end_depots))
        starting_time = 0.0
        starting_charge = data.B
        key = (
            (starting_node, starting_time, starting_charge),  
            (current_node, starting_time, starting_charge)
        )
        # initialise a proportion of the customers to be served
        served = zeros(Int, data.n_customers)
        for i in 1:length(served)
            if mod1(i, data.n_vehicles) == v
                served[i] = 1
            end
        end
        s = Subpath(
            n_customers = data.n_customers,
            starting_node = starting_node,
            starting_time = starting_time,
            starting_charge = starting_charge,
            current_node = current_node,
            arcs = [(starting_node, current_node)],
            current_time = starting_time,
            current_charge = starting_charge,
            load = 0,
            served = served,
            artificial = true,
        )
        p = Path(
            subpaths = [s],
            charging_arcs = ChargingArc[],
            served = served,
            load = 0,
            arcs = [(starting_node, current_node)],
            artificial = true,
        )
        if !(key in keys(artificial_paths))
            artificial_paths[key] = Path[]
        end
        push!(artificial_paths[key], p)
    end
    return artificial_paths
end

function add_empty_paths!(
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    data::EVRPData,
)
    for starting_node in data.N_depots
        starting_time = 0.0
        starting_charge = data.B
        current_node = starting_node
        key = (
            (starting_node, starting_time, starting_charge),  
            (current_node, starting_time, starting_charge)
        )
        s = Subpath(
            n_customers = data.n_customers,
            starting_node = starting_node,
            starting_time = starting_time,
            starting_charge = starting_charge,
            current_node = current_node,
            arcs = [(starting_node, current_node)],
            current_time = starting_time,
            current_charge = starting_charge,
            load = 0,
            served = zeros(Int, data.n_customers),
            artificial = false,
        )
        p = Path(
            subpaths = [s],
            charging_arcs = ChargingArc[],
            served = zeros(Int, data.n_customers),
            load = 0,
            arcs = [(starting_node, current_node)],
            artificial = false,
        )
        if !(key in keys(some_paths))
            some_paths[key] = Path[]
        end
        push!(some_paths[key], p)
    end
    return
end

function add_path_to_generated_paths!(
    generated_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Tuple{Float64, Path}},
    },
    c::Float64,
    p::Path,
)
    state_pair = (
        (p.subpaths[1].starting_node, p.subpaths[1].starting_time, p.subpaths[1].starting_charge),
        (p.subpaths[end].current_node, p.subpaths[end].current_time, p.subpaths[end].current_charge)
    )
    if state_pair in keys(generated_paths)
        if !any(isequal(p, p2) for p2 in generated_paths[state_pair])
            push!(generated_paths[state_pair], (c, p))
        end
    else
        generated_paths[state_pair] = [(c, p)]
    end
    return
end


function path_formulation_build_model(
    data::EVRPData,
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Path},
    },
    path_costs::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Int},
    },
    path_service::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
            Int,
        }, 
        Vector{Int}
    },
    ;
    Env = nothing,
)
    if isnothing(Env)
        model = @suppress Model(Gurobi.Optimizer)
    else
        model = @suppress Model(() -> Gurobi.Optimizer(Env))
    end
    JuMP.set_attribute(model, "MIPGapAbs", 1e-3)
    JuMP.set_string_names_on_creation(model, false)

    z = Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    }(
        (key, p) => @variable(model, lower_bound = 0)
        for key in keys(some_paths)
            for p in 1:length(some_paths[key])
    )
    @constraint(
        model, 
        κ[n1 in data.N_depots],
        sum(
            sum(
                z[(state1, state2),p]
                for p in 1:length(some_paths[(state1, state2)])
            )
            for (state1, state2) in keys(some_paths)
                if state1[1] == n1 && state1[2] == 0 && state1[3] == data.B
        )
        == data.v_start[n1]
    )
    @constraint(
        model,
        μ[n2 in data.N_depots],
        sum(
            sum(
                z[(state1, state2),p]
                for p in 1:length(some_paths[(state1, state2)])
            )
            for (state1, state2) in keys(some_paths)
                if state2[1] == n2
        )
        ≥ data.v_end[n2]
    )
    @constraint(
        model,
        ν[j in data.N_customers],
        sum(
            sum(
                path_service[((state1, state2), j)][p] * z[(state1, state2), p]
                for p in 1:length(some_paths[(state1, state2)])
            )
            for (state1, state2) in keys(some_paths)
        )
        >= 1
    )
    @expression(
        model, 
        path_costs_expr,
        sum(
            sum(
                path_costs[state_pair][p] * z[state_pair,p]
                for p in 1:length(some_paths[state_pair])
            )
            for state_pair in keys(some_paths)
        )
    )
    @objective(model, Min, path_costs_expr)

    return model, z
end

function check_path_in_SR3_constraint(
    path::Path,
    S::NTuple{3, Int},
)
    return Int(floor(sum(path.served[collect(S)]) / 2))
end

function compute_path_coefficient_in_lmSRnk_constraint(
    path::Path,
    S::Tuple{Vararg{Int}},
    M::Tuple{Vararg{Int}},
    k::Int,
)
    nodes = get_nodes(path)
    coeff = 0
    state = 0
    for node in nodes
        if !(node in M)
            state = 0
        elseif node in S
            state += 1
            if state ≥ k
                coeff += 1
                state -= k
            end
        end
    end
    return coeff
end

function add_paths_to_path_model!(
    model::Model,
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
            Int,
        }, 
        VariableRef,
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    path_costs::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Int},
    },
    path_service::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        Vector{Int},
    },
    generated_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Tuple{Float64, Path}},
    },
    SR3_constraints::Dict{
        T, 
        ConstraintRef, 
    },
    data::EVRPData,
) where {T}
    mp_constraint_start_time = time()
    for state_pair in keys(generated_paths)
        if !(state_pair in keys(some_paths))
            some_paths[state_pair] = Path[]
            path_costs[state_pair] = Int[]
            for i in 1:data.n_customers
                path_service[(state_pair, i)] = Int[]
            end
            count = 0
        else
            count = length(some_paths[state_pair])
        end
        for (c, p_new) in generated_paths[state_pair]
            if state_pair in keys(some_paths)
                for p in some_paths[state_pair]
                    if isequal(p_new, p)
                        throw(ErrorException("""
                        Generated path already in `some_paths!`
                        Reduced cost: $c
                        Generated path: $p_new
                        `some_paths`: $(some_paths[state_pair])
                        """))
                        break
                    end
                end
            end

            count += 1
            # 1: include in some_paths
            push!(some_paths[state_pair], p_new)
            # 2: add path cost
            push!(
                path_costs[state_pair], 
                compute_path_cost(data, p_new)
            )
            # 3: add path service
            for i in 1:data.n_customers
                push!(path_service[(state_pair, i)], p_new.served[i])
            end
            # 4: create variable
            z[(state_pair, count)] = @variable(model, lower_bound = 0)
            (state1, state2) = state_pair
            # 5: modify constraints starting from depot, ending at depot
            set_normalized_coefficient(model[:κ][state1[1]], z[state_pair,count], 1)
            set_normalized_coefficient(model[:μ][state2[1]], z[state_pair,count], 1)
            # 6: modify customer service constraints
            for l in data.N_customers
                set_normalized_coefficient(model[:ν][l], z[state_pair, count], p_new.served[l])
            end
            # 7: modify objective
            set_objective_coefficient(model, z[state_pair, count], path_costs[state_pair][count])
            # 8: add variable to violated SR3 constraints (if applicable)
            if keytype(SR3_constraints) == NTuple{3, Int}
                for (SR3_key, SR3_con) in pairs(SR3_constraints)
                    val = check_path_in_SR3_constraint(p_new, SR3_key)
                    if val > 0
                        set_normalized_coefficient(SR3_con, z[state_pair,count], val)
                    end
                end
            elseif keytype(SR3_constraints) == Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}}
                for ((S, M), SR3_con) in pairs(SR3_constraints)
                    val = compute_path_coefficient_in_lmSRnk_constraint(p_new, S, M, 2)
                    if val > 0
                        set_normalized_coefficient(SR3_con, z[state_pair,count], val)
                    end
                end
            end
        end
    end
    mp_constraint_end_time = time()
    mp_constraint_time = round(mp_constraint_end_time - mp_constraint_start_time, digits = 3)
    return mp_constraint_time
end

function path_formulation_column_generation_solve_linear_relaxation!(
    model::Model,
    CG_params::Dict{String, Any},
    printlist::Vector{String},
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    SR3_constraints::Dict{
        T,
        ConstraintRef,
    },
    data::EVRPData,
    artificial_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    ;
    verbose::Bool = true,
) where {T}
    mp_solution_start_time = time()
    @suppress optimize!(model)
    mp_solution_end_time = time()
    if JuMP.termination_status(model) in [
        JuMP.INFEASIBLE,
        JuMP.INFEASIBLE_OR_UNBOUNDED,
        JuMP.TIME_LIMIT,
        JuMP.MEMORY_LIMIT,
    ]
        CGLP_results = Dict(
            "errored" => true,
            "artificial" => false,
        )
        add_message!(printlist, "CGLP model errored.\n", verbose)
        return CGLP_results
    end
    CGLP_results = Dict(
        "errored" => false,
        "objective" => objective_value(model),
        "z" => Dict(
            (key, p) => value.(z[(key, p)])
            for (key, p) in keys(z)
        ),
        "κ" => Dict(zip(data.N_depots, dual.(model[:κ]).data)),
        "μ" => Dict(zip(data.N_depots, dual.(model[:μ]).data)),
        "ν" => dual.(model[:ν]).data,
        "λ" => Dict{keytype(SR3_constraints), Float64}(
            S => dual(SR3_constraints[S])
            for S in keys(SR3_constraints)
        ),
    )
    CGLP_results["artificial"] = any(
        value.(z[(key, p)]) > 1e-3
        for key in keys(artificial_paths)
            for p in 1:length(artificial_paths[key])
    )
    push!(CG_params["objective"], CGLP_results["objective"])
    push!(CG_params["κ"], CGLP_results["κ"])
    push!(CG_params["μ"], CGLP_results["μ"])
    push!(CG_params["ν"], CGLP_results["ν"])
    push!(CG_params["λ"], CGLP_results["λ"])
    push!(CG_params["lp_relaxation_solution_time_taken"], round(mp_solution_end_time - mp_solution_start_time, digits = 3))

    return CGLP_results
end

function path_formulation_column_generation_find_nondominated_paths(
    data::EVRPData,
    graph::EVRPGraph,
    CG_params::Dict{String, Any},
    CGLP_results::Dict{String, Any},
    printlist::Vector{String},
    ;
    method::String = "ours",
    use_load::Bool = false,
    use_time_windows::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    prune_graph_outdegree_sequence::Vector{Int} = Int[],
    exploration_method::String = "bestfirst",
    use_ngroute::Bool = false,
    ng_neighborhoods::Union{Nothing, BitMatrix} = nothing,
    verbose::Bool = false,
    time_limit::Float64 = Inf,
)
    """
    For a given iteration, run either the "ours" or "benchmark" algorithm 
    to find nondominated paths with negative reduced cost.
    """

    start_time = time()

    modified_costs = compute_arc_modified_costs(
        data,
        CGLP_results["κ"], 
        CGLP_results["μ"], 
        CGLP_results["ν"],
    )

    subpath_labels_total_time = 0.0
    path_labels_total_time = 0.0

    found_paths_with_pruned_graph = false
    prune_graph_outdegree_sequence = [
        k for k in prune_graph_outdegree_sequence
            if k < data.n_customers
    ]

    for k in prune_graph_outdegree_sequence
        pruned_graph = prune_graph(graph, modified_costs; k = k)

        if method == "ours"
            (timed_out, generated_paths, generated_paths_count, subpath_labels_time, path_labels_time) = subproblem_iteration_ours(
                data, pruned_graph, 
                modified_costs,
                CGLP_results["λ"], 
                ;
                use_load = use_load,
                use_time_windows = use_time_windows,
                use_nonlinear_charging = use_nonlinear_charging,
                charging_function = charging_function,
                exploration_method = exploration_method,
                use_ngroute = use_ngroute,
                ng_neighborhoods = ng_neighborhoods,
                time_limit = time_limit - (time() - start_time),
            )
        elseif method == "benchmark"
            subpath_labels_time = 0.0
            (timed_out, generated_paths, generated_paths_count, path_labels_time) = subproblem_iteration_benchmark(
                data, pruned_graph, 
                modified_costs,
                CGLP_results["λ"], 
                ;
                use_load = use_load,
                use_time_windows = use_time_windows,
                use_nonlinear_charging = use_nonlinear_charging,
                charging_function = charging_function,
                exploration_method = exploration_method,
                use_ngroute = use_ngroute,
                ng_neighborhoods = ng_neighborhoods,
                time_limit = time_limit - (time() - start_time),
            )
        end
        subpath_labels_total_time += subpath_labels_time
        path_labels_total_time += path_labels_time
        add_message!(
            printlist, 
            @sprintf(
                "              |            |            |           | %9.3f | %9.3f |           | %10d %s %2d\n", 
                subpath_labels_time,
                path_labels_time,
                generated_paths_count,
                "- H",
                k,
            ),
            verbose,
        )
        if timed_out
            push!(CG_params["sp_base_time_taken"], subpath_labels_total_time)
            push!(CG_params["sp_full_time_taken"], path_labels_total_time)
            push!(CG_params["sp_total_time_taken"], subpath_labels_total_time + path_labels_total_time)
            push!(CG_params["number_of_new_paths"], generated_paths_count)
            return (true, generated_paths, generated_paths_count)
        end
        if generated_paths_count > 0
            found_paths_with_pruned_graph = true
            break
        end
    end

    if !found_paths_with_pruned_graph
        if method == "ours"
            (timed_out, generated_paths, generated_paths_count, subpath_labels_time, path_labels_time) = subproblem_iteration_ours(
                data, graph, 
                modified_costs,
                CGLP_results["λ"], 
                ;
                use_load = use_load,
                use_time_windows = use_time_windows,
                use_nonlinear_charging = use_nonlinear_charging,
                charging_function = charging_function,
                exploration_method = exploration_method,
                use_ngroute = use_ngroute,
                ng_neighborhoods = ng_neighborhoods,
                time_limit = time_limit - (time() - start_time),
            )
        elseif method == "benchmark"
            subpath_labels_time = 0.0
            (timed_out, generated_paths, generated_paths_count, path_labels_time) = subproblem_iteration_benchmark(
                data, graph, 
                modified_costs,
                CGLP_results["λ"], 
                ;
                use_load = use_load,
                use_time_windows = use_time_windows,
                use_nonlinear_charging = use_nonlinear_charging,
                charging_function = charging_function,
                exploration_method = exploration_method,
                use_ngroute = use_ngroute,
                ng_neighborhoods = ng_neighborhoods,
                time_limit = time_limit - (time() - start_time),
            )
        end
        subpath_labels_total_time += subpath_labels_time
        path_labels_total_time += path_labels_time
        add_message!(
            printlist, 
            @sprintf(
                "              |            |            |           | %9.3f | %9.3f |           | %10d %s\n", 
                subpath_labels_time,
                path_labels_time,
                generated_paths_count,
                (found_paths_with_pruned_graph ? "- H" : ""),
            ),
            verbose,
        )
        if timed_out
            push!(CG_params["sp_base_time_taken"], subpath_labels_total_time)
            push!(CG_params["sp_full_time_taken"], path_labels_total_time)
            push!(CG_params["sp_total_time_taken"], subpath_labels_total_time + path_labels_total_time)
            push!(CG_params["number_of_new_paths"], generated_paths_count)
            return (true, generated_paths, generated_paths_count)
        end
    end

    push!(CG_params["sp_base_time_taken"], subpath_labels_total_time)
    push!(CG_params["sp_full_time_taken"], path_labels_total_time)
    push!(CG_params["sp_total_time_taken"], subpath_labels_total_time + path_labels_total_time)
    push!(CG_params["number_of_new_paths"], generated_paths_count)
    return (false, generated_paths, generated_paths_count)
end

function path_formulation_column_generation!(
    model::Model,
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    SR3_constraints::Dict{
        T,
        ConstraintRef,
    },
    data::EVRPData,
    graph::EVRPGraph,
    artificial_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    path_costs::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Int},
    },
    path_service::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        Vector{Int},
    },
    printlist::Vector{String},
    ;
    method::String = "ours",
    use_load::Bool = false,
    use_time_windows::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    use_pruned_graph::Bool = false,
    prune_graph_outdegree_sequence::Vector{Int} = Int[3, 8],
    exploration_method::String = "bestfirst",
    use_ngroute::Bool = false,
    ng_neighborhoods::Union{Nothing, BitMatrix} = nothing,
    verbose::Bool = true,
    time_limit::Float64 = Inf,
    max_iters::Float64 = Inf,
) where {T}

    start_time = time()
    counter = 0
    converged = false
    local CGLP_results = Dict{String, Any}()
    timed_out = false

    CG_params = Dict{String, Any}()
    CG_params["number_of_paths"] = [sum(length(v) for v in values(some_paths))]
    CG_params["objective"] = Float64[]
    CG_params["κ"] = Dict{Int, Float64}[]
    CG_params["μ"] = Dict{Int, Float64}[]
    CG_params["ν"] = Vector{Float64}[]
    CG_params["λ"] = Dict{keytype(SR3_constraints), Float64}[]
    CG_params["lp_relaxation_solution_time_taken"] = Float64[]
    CG_params["sp_base_time_taken"] = Float64[]
    CG_params["sp_full_time_taken"] = Float64[]
    CG_params["sp_total_time_taken"] = Float64[]
    CG_params["lp_relaxation_constraint_time_taken"] = Float64[]
    CG_params["number_of_new_paths"] = Int[]
    CG_params["converged"] = false
    CG_params["cycled"] = false

    while (
        !converged
        && max_iters > counter
    )
        counter += 1

        CGLP_results = path_formulation_column_generation_solve_linear_relaxation!(
            model,
            CG_params,
            printlist,
            z,
            SR3_constraints,
            data,
            artificial_paths,
            ;
            verbose = verbose,
        )
        if CGLP_results["errored"]
            break
        end

        (timed_out, generated_paths, generated_paths_count) = path_formulation_column_generation_find_nondominated_paths(
            data,
            graph,
            CG_params,
            CGLP_results,
            printlist,
            ;
            method = method,
            use_load = use_load,
            use_time_windows = use_time_windows,
            use_nonlinear_charging = use_nonlinear_charging,
            charging_function = charging_function,
            prune_graph_outdegree_sequence = (
                use_pruned_graph 
                ? prune_graph_outdegree_sequence
                : Int[]
            ),
            exploration_method = exploration_method,
            use_ngroute = use_ngroute,
            ng_neighborhoods = ng_neighborhoods,
            verbose = verbose,
            time_limit = time_limit - (time() - start_time),
        )
        if timed_out
            break
        end
        if generated_paths_count == 0
            converged = true
        end

        mp_constraint_time = add_paths_to_path_model!(
            model,
            z,
            some_paths, 
            path_costs,
            path_service,
            generated_paths,
            SR3_constraints,
            data, 
        )

        push!(
            CG_params["number_of_paths"], 
            sum(length(v) for v in values(some_paths))
        )
        push!(
            CG_params["lp_relaxation_constraint_time_taken"],
            mp_constraint_time,
        )
        add_message!(
            printlist, 
            @sprintf(
                "Iteration %3d | %.4e | %10d | %9.3f | %9.3f | %9.3f | %9.3f | %10d %s\n", 
                counter,
                CG_params["objective"][counter],
                CG_params["number_of_paths"][counter],
                CG_params["lp_relaxation_solution_time_taken"][counter],
                CG_params["sp_base_time_taken"][counter],
                CG_params["sp_full_time_taken"][counter],                
                CG_params["lp_relaxation_constraint_time_taken"][counter],
                CG_params["number_of_new_paths"][counter],
                (use_pruned_graph ? "- H" : ""),
            ),
            verbose,
        )

        # cycling detection
        if (
            length(CG_params["number_of_paths"]) >= 2
            && CG_params["number_of_paths"][end] == CG_params["number_of_paths"][end-1]
            && !converged
        )
            CG_params["cycled"] = true
            add_message!(printlist, "CG cycling error.\n", verbose)
            break
        end
    end

    CG_params["converged"] = converged
    CG_params["counter"] = counter
    end_time = time() 
    time_taken = round(end_time - start_time, digits = 3)
    CG_params["time_taken"] = time_taken
    CG_params["time_limit_reached"] = timed_out
    CG_params["errored"] = CGLP_results["errored"]
    CG_params["artificial"] = CGLP_results["artificial"]
    CG_params["lp_relaxation_time_taken"] = sum.(zip(CG_params["lp_relaxation_constraint_time_taken"], CG_params["lp_relaxation_solution_time_taken"]))
    CG_params["lp_relaxation_time_taken_total"] = sum(CG_params["lp_relaxation_time_taken"])
    CG_params["sp_base_time_taken_total"] = sum(CG_params["sp_base_time_taken"])
    CG_params["sp_full_time_taken_total"] = sum(CG_params["sp_full_time_taken"])
    CG_params["sp_time_taken_total"] = CG_params["sp_base_time_taken_total"] + CG_params["sp_full_time_taken_total"]
    CG_params["lp_relaxation_time_taken_mean"] = CG_params["lp_relaxation_time_taken_total"] / length(CG_params["lp_relaxation_time_taken"])
    CG_params["sp_base_time_taken_mean"] = CG_params["sp_base_time_taken_total"] / length(CG_params["sp_base_time_taken"])
    CG_params["sp_full_time_taken_mean"] = CG_params["sp_full_time_taken_total"] / length(CG_params["sp_full_time_taken"])
    CG_params["sp_time_taken_mean"] = CG_params["sp_base_time_taken_mean"] + CG_params["sp_full_time_taken_mean"]

    add_message!(
        printlist, 
        @sprintf(
            "Total         |            | %10d | %9.3f | %9.3f | %9.3f | %9.3f | \n", 
            CG_params["number_of_paths"][end],
            sum(CG_params["lp_relaxation_solution_time_taken"]),
            sum(CG_params["sp_base_time_taken"]),
            sum(CG_params["sp_full_time_taken"]),
            sum(CG_params["lp_relaxation_constraint_time_taken"]),
        ),
        verbose,
    )

    return (timed_out, CGLP_results, CG_params)
end

function path_formulation_solve_integer_model!(
    model::Model,
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    artificial_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    printlist::Vector{String},
)
    for (key, p) in keys(z)
        JuMP.set_integer(z[key, p])
    end

    CGIP_solution_start_time = time()
    @suppress optimize!(model)
    CGIP_solution_end_time = time()
    if JuMP.termination_status(model) in [
        JuMP.INFEASIBLE,
        JuMP.INFEASIBLE_OR_UNBOUNDED,
        JuMP.TIME_LIMIT,
        JuMP.MEMORY_LIMIT,
    ]
        CGIP_results = Dict(
            "errored" => true,
            "artificial" => false,
        )
        add_message!(printlist, "CGIP model errored.", verbose)
    else
        CGIP_results = Dict(
            "errored" => false,
            "objective" => objective_value(model),
            "z" => Dict(
                (key, p) => value.(z[key, p])
                for (key, p) in keys(z)
            ),
            "time_taken" => round(CGIP_solution_end_time - CGIP_solution_start_time, digits = 3),
        )
        CGIP_results["artificial"] = any(
            value.(z[(key, p)]) > 1e-3
            for key in keys(artificial_paths)
                for p in 1:length(artificial_paths[key])
        )
    end

    for (key, p) in keys(z)
        JuMP.unset_integer(z[key, p])
    end

    return CGIP_results
end


function detect_cycle_in_path(
    p::Path,
    node::Int,
)
    # Assumes that `node` is a customer.

    # finds first and last occurrences of node
    nodes = get_nodes(p)
    start_ind = findfirst(x -> x == node, nodes)
    end_ind = findlast(x -> x == node, nodes)
    return nodes[start_ind:end_ind]
end


function detect_cycles_in_path_solution(
    path_results::Vector{Path},
)
    cycles = Dict{Int, Set{Int}}()
    for p in path_results
        for node in findall(x -> x > 1, p.served)
            if !(node in keys(cycles))
                cycles[node] = Set{Int}()
            end
            union!(cycles[node], detect_cycle_in_path(p, node))
        end
    end
    return cycles
end


function modify_ng_neighborhoods_with_found_cycles!(
    ng_neighborhoods::BitMatrix,
    cycles_lookup::Dict{Int, Set{Int}},
)
    entries_changed = 0
    for (node, changed_nodes) in pairs(cycles_lookup)
        entries_changed += sum(.!ng_neighborhoods[node, collect(changed_nodes)])
        ng_neighborhoods[node, collect(changed_nodes)] .= 1
    end
    return entries_changed
end


function delete_paths_with_found_cycles_from_model!(
    model::Model,
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
            Int,
        }, 
        VariableRef,
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    path_costs::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Int},
    },
    path_service::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        Vector{Int},
    },
    cycles_lookup::Dict{
        Int,
        Set{Int},
    },
    data::EVRPData,
)
    for state_pair in keys(some_paths)
        delete_inds = Int[]
        for (count, path) in enumerate(some_paths[state_pair])
            # check if path needs to be removed
            removed = false
            for (node, changed_nodes) in pairs(cycles_lookup)
                if path.served[node] ≤ 1
                    continue
                end
                cycle = detect_cycle_in_path(path, node)
                if !isempty(setdiff(intersect(cycle, changed_nodes), [node]))
                    removed = true
                    # println("""
                    # Deleted path due to $node, $changed_nodes: 
                    # $cycle; \t $(path.arcs)
                    # """)
                    break
                end
            end
            if removed
                push!(delete_inds, count)
            end
        end
        # delete variables from model
        orig_inds = 1:length(some_paths[state_pair])
        for count in delete_inds
            JuMP.delete(model, z[(state_pair, count)])
            pop!(z, (state_pair, count))
        end
        # rename variable bindings in dictionary `z`
        for (i, ind) in enumerate(sort(setdiff(orig_inds, delete_inds)))
            z[(state_pair, i)] = pop!(z, (state_pair, ind))
        end
        # delete paths, their costs, and service info via `delete_inds`
        deleteat!(some_paths[state_pair], delete_inds)
        deleteat!(path_costs[state_pair], delete_inds)
        for node in data.N_customers
            deleteat!(path_service[(state_pair, node)], delete_inds)
        end
    end
end


function enumerate_violated_path_SR3_inequalities(
    paths::Vector{Tuple{Float64, Path}},
    data::EVRPData,
)
    S_list = Tuple{Float64, NTuple{3, Int}}[]
    for S in combinations(data.N_customers, 3)
        violation = sum(
            val * floor(sum(p.served[S]) / 2)
            for (val, p) in values(paths)
        ) - 1.0
        if violation > 1e-6
            push!(S_list, (violation, Tuple(S)))
        end
    end
    sort!(S_list, by = x -> x[1], rev = true)
    return S_list
end

function select_representative_violated_path_SR3_inequalities(
    SR3_list_in::Vector{Tuple{Float64, NTuple{3, Int}}},
    data::EVRPData,
)

    function find_closest_triplet(
        clique::Tuple{Float64, Tuple{Vararg{Int}}, Tuple{Vararg{Int}}, Tuple{Vararg{Int}}},
        data::EVRPData,
    )
        @assert length(clique[2]) ≥ 1
        @assert length(clique[3]) ≥ 1
        @assert length(clique[4]) ≥ 1
        (_, S) = minimum([
            (
                data.q[i,j] + data.q[j,k] + data.q[k,i],
                Tuple(sort([i,j,k]))
            )
            for i in clique[2] for j in clique[3] for k in clique[4]
        ])
        return S
    end

    found_cliques = Tuple{Float64, Tuple{Vararg{Int}}, Tuple{Vararg{Int}}, Tuple{Vararg{Int}}}[]
    SR3_list = copy(SR3_list_in)
    while length(SR3_list) != 0
        f_found = 0.0
        local found_triples
        local clique
        for (i, (f, S)) in enumerate(SR3_list)
            c1 = Tuple(vcat([S[3]],
                [setdiff(S_, S)[1] for (f_, S_) in union(SR3_list[1:i-1], SR3_list[i+1:end])
                if isapprox(f, f_, atol=1e-12) && S[1] in S_ && S[2] in S_],
            ))
            c2 = Tuple(vcat([S[2]],
                [setdiff(S_, S)[1] for (f_, S_) in union(SR3_list[1:i-1], SR3_list[i+1:end])
                if isapprox(f, f_, atol=1e-12) && S[1] in S_ && S[3] in S_],
            ))
            c3 = Tuple(vcat([S[1]],
                [setdiff(S_, S)[1] for (f_, S_) in union(SR3_list[1:i-1], SR3_list[i+1:end])
                if isapprox(f, f_, atol=1e-12) && S[2] in S_ && S[3] in S_],
            ))
            clique = (f, c1, c2, c3)
            found_triples = [
                Tuple(sort([i, j, k]))
                for i in clique[2], j in clique[3], k in clique[4]
            ]
            matches_in_SR3_list = [
                (f_, S_) for (f_, S_) in SR3_list
                    if (S_ in found_triples && isapprox(f, f_, atol=1e-12))
            ]
            if length(found_triples) == length(matches_in_SR3_list)
                f_found = f
                break
            end
        end
        if f_found == 0.0
            break
        end
        SR3_list = [
            (f_, S_) for (f_, S_) in SR3_list
                if !(S_ in found_triples && f_found ≈ f_)
        ]
        push!(found_cliques, clique)
    end
    select_triples = [
        find_closest_triplet(clique, data)
        for clique in found_cliques
    ]
    SR3_list_final = vcat(
        Tuple{Float64, NTuple{3, Int}}[
            (f, S) for (f, S) in SR3_list_in
                if S in select_triples
        ],
        SR3_list,
    )
    return sort(SR3_list_final, by = x -> x[1], rev = true)
end

function enumerate_violated_path_SRnk_inequalities(
    paths::Vector{Tuple{Float64, Path}},
    data::EVRPData,
    n::Int,
    k::Int,
)
    S_list = Tuple{Float64, NTuple{n, Int}}[]
    for S in combinations(data.N_customers, n)
        violation = sum(
            val * floor(sum(p.served[S]) / k)
            for (val, p) in values(paths)
        ) - floor(n / k)
        if violation > 1e-6
            push!(S_list, (violation, Tuple(S)))
        end
    end
    sort!(S_list, by = x -> x[1], rev = true)
    return S_list
end

function compute_memory_set_of_lmSRnk_inequality(
    paths::Vector{Tuple{Float64, Path}},
    S::Tuple{Vararg{Int}},
    k::Int,
)
    M = Set(S)
    for (_, p) in paths
        M_add = Set{Int}()
        state = 0
        nodes = get_nodes(p)
        for node in nodes
            if node in S
                state += 1
                if state ≥ k
                    union!(M, M_add)
                    M_add = Set{Int}()
                    state -= k
                end
            elseif state > 0
                union!(M_add, node)
            end
        end
    end
    return Tuple{Vararg{Int}}(sort(collect(M)))
end


function add_SR3_constraints_to_path_model!(
    model::Model, 
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    SR3_constraints::Dict{
        NTuple{3, Int},
        ConstraintRef
    }, 
    generated_SR3_list::Vector{Tuple{Float64, NTuple{3, Int}}}, 
)
    for item in generated_SR3_list
        S = item[2]
        SR3_constraints[S] = @constraint(
            model,
            sum(
                sum(
                    floor(sum(p.served[collect(S)]) / 2) * z[state_pair, p_ind]
                    for (p_ind, p) in enumerate(some_paths[state_pair])
                )
                for state_pair in keys(some_paths)
            ) ≤ 1
        )
    end
end

function update_lmSR3_constraint_list!(
    lmSR3_list::Vector{Tuple{Float64, NTuple{3, Int}, T}},
    implemented_lmSR3_list::Vector{Tuple{Float64, NTuple{3, Int}, T}},
) where {T <: Tuple{Vararg{Int}}}
    lmSR3_list_todelete = Tuple{Float64, NTuple{3, Int}, Tuple{Vararg{Int}}}[]
    lmSR3_list_toadd = Tuple{Float64, NTuple{3, Int}, Tuple{Vararg{Int}}}[]
    for item in implemented_lmSR3_list
        S, M = item[2], item[3]
        big_M = collect(M)
        lmSR3_list_todeleteinds = Int[]
        for (i, item_) in enumerate(lmSR3_list)
            S_, M_ = item_[2], item_[3]
            if S == S_
                append!(big_M, collect(M_))
                push!(lmSR3_list_todeleteinds, i)
                push!(lmSR3_list_todelete, item_)
            end
        end
        deleteat!(lmSR3_list, lmSR3_list_todeleteinds)
        big_M = Tuple(sort(unique(big_M)))
        push!(lmSR3_list_toadd, (item[1], S, big_M))
    end
    append!(lmSR3_list, lmSR3_list_toadd)
    return lmSR3_list_todelete, lmSR3_list_toadd
end

function add_lmSR3_constraints_to_path_model!(
    model::Model, 
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    lmSR3_constraints::Dict{
        Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}},
        ConstraintRef,
    }, 
    lmSR3_list_todelete::Vector{
        Tuple{Float64, NTuple{3, Int}, T}
    }, 
    lmSR3_list_toadd::Vector{
        Tuple{Float64, NTuple{3, Int}, T}
    }, 
) where {T <: Tuple{Vararg{Int}}}
    for item in lmSR3_list_todelete
        S, M = item[2], item[3]
        JuMP.delete(model, lmSR3_constraints[(S, M)])
        pop!(lmSR3_constraints, (S, M))
    end
    for item in lmSR3_list_toadd
        S, M = item[2], item[3]
        lmSR3_constraints[(S, M)] = @constraint(
            model,
            sum(
                sum(
                    compute_path_coefficient_in_lmSRnk_constraint(p, S, M, 2) * z[state_pair, p_ind]
                    for (p_ind, p) in enumerate(some_paths[state_pair])
                )
                for state_pair in keys(some_paths)
            ) ≤ 1
        )
    end
    return
end

function add_lmSR3_constraints_to_path_model!(
    model::Model, 
    z::Dict{
        Tuple{
            Tuple{NTuple{3, Int}, NTuple{3, Int}},
            Int,
        },
        VariableRef,
    },
    some_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    lmSR3_constraints::Dict{
        Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}},
        ConstraintRef,
    }, 
    lmSR3_list_toadd::Vector{
        Tuple{Float64, NTuple{3, Int}, T}
    }, 
) where {T <: Tuple{Vararg{Int}}}
    for item in lmSR3_list_toadd
        S, M = item[2], item[3]
        lmSR3_constraints[(S, M)] = @constraint(
            model,
            sum(
                sum(
                    compute_path_coefficient_in_lmSRnk_constraint(p, S, M, 2) * z[state_pair, p_ind]
                    for (p_ind, p) in enumerate(some_paths[state_pair])
                )
                for state_pair in keys(some_paths)
            ) ≤ 1
        )
    end
    return
end


function path_formulation_column_generation_with_adaptive_ngroute_SR3_cuts(
    data::EVRPData, 
    graph::EVRPGraph,
    ;
    Env = nothing,
    method::String = "ours",
    use_time_windows::Bool = false,
    use_load::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    use_pruned_graph::Bool = false,
    prune_graph_outdegree_sequence::Vector{Int} = Int[3, 8],
    exploration_method::String = "bestfirst",
    use_ngroute::Bool = true,
    ng_neighborhoods::Union{Nothing, BitMatrix} = nothing,
    ngroute_neighborhood_size::Int = Int(ceil(sqrt(data.n_customers))),
    ngroute_neighborhood_charging_size::Int = 1,
    use_adaptive_ngroute::Bool = true,
    use_SR3_cuts::Bool = true,
    use_lmSR3_cuts::Bool = true,
    max_SR3_cuts::Int = 5, 
    randomize_cuts::Bool = false,
    verbose::Bool = true,
    time_limit::Float64 = Inf,
    max_iters::Float64 = Inf,
)
    start_time = time()

    if use_ngroute && isnothing(ng_neighborhoods)
        ng_neighborhoods = compute_ngroute_neighborhoods(
            graph,
            ngroute_neighborhood_size, 
            ngroute_neighborhood_charging_size,
        )
    end


    printlist = String[]

    start_printlist = String[]
    push!(start_printlist, 
        @sprintf(
            """
            Starting column generation on the path formulation.
            # customers:                    %3d
            # depots:                       %3d
            # charging stations:            %3d
            # vehicles:                     %3d
            use load?:                      %s
            use time windows?:              %s

            method:                         %s
            exploration method:             %S
            """,
            data.n_customers,
            data.n_depots,
            data.n_charging,
            data.n_vehicles,
            use_load,
            use_time_windows,
            method,
            exploration_method,
        )
    )
    if use_ngroute
        push!(start_printlist, 
            @sprintf(
                """
                use ngroute?:                   %s
                ngroute neighborhood size:
                    customers                   %3d
                    depots                      %3d
                    charging                    %3d
                """,
                use_ngroute,
                ngroute_neighborhood_size,
                1,
                ngroute_neighborhood_charging_size,
            )
        )
        push!(start_printlist,
            @sprintf(
                "use_adaptive_ngroute:           %s\n",
                use_adaptive_ngroute
            )
        )
        if use_SR3_cuts
            push!(start_printlist,
                @sprintf(
                    """
                    use_SR3_cuts?:                  %s
                        use_lmSR3_cuts?:            %s
                        max_SR3_cuts:               %3d
                    """,
                    use_SR3_cuts,
                    use_lmSR3_cuts,
                    max_SR3_cuts,
                )
            )
        else
            push!(start_printlist,
                @sprintf(
                    """
                    use_SR3_cuts?:                  %s
                    """,
                    use_SR3_cuts,
                )
            )
        end
    else
        push!(start_printlist,
            @sprintf(
                """
                use ngroute?:                   %s
                """,
                use_ngroute,
            )
        )
    end

    for message in start_printlist
        add_message!(printlist, message, verbose)
    end

    artificial_paths = generate_artificial_paths(data)
    some_paths = deepcopy(artificial_paths)
    add_empty_paths!(some_paths, data)

    path_costs = compute_path_costs(data, some_paths)
    path_service = compute_path_service(data, some_paths)

    add_message!(
        printlist,
        @sprintf("              |  Objective | #    paths | Time (LP) | Time (SP) | Time (SP) | Time (LP) | #    paths \n"),
        verbose,
    )
    add_message!(
        printlist,
        @sprintf("              |            |            |           |    (base) |    (full) |    (cons) |      (new) \n"),
        verbose,
    )

    model, z = path_formulation_build_model(
        data, some_paths, path_costs, path_service,
        ; 
        Env = Env,
    )

    local CGLP_results
    local CG_params
    local CGIP_results
    local converged = false

    if use_lmSR3_cuts
        SR3_constraints = Dict{
            Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}},
            ConstraintRef,
        }()
        SR3_list = Tuple{Float64, NTuple{3, Int}, Tuple{Vararg{Int}}}[]
    else
        SR3_constraints = Dict{NTuple{3, Int}, ConstraintRef}()
        SR3_list = Tuple{Float64, NTuple{3, Int}}[]
    end
    CGLP_all_results = Dict[]
    CGIP_all_results = Dict[]
    all_params = Dict[]
    CG_all_params = Dict[]
    if use_ngroute
        CG_all_ng_neighborhoods = BitMatrix[]
    else
        CG_all_ng_neighborhoods = nothing
    end

    while true
        continue_flag = false
        iteration_params = Dict{String, Any}()

        # Setup
        iteration_params["errored"] = false
        iteration_params["method"] = "none"
        iteration_params["ngroute_neighborhoods_expanded"] = 0
        iteration_params["ngroute_neighborhood_size"] = use_ngroute ? (data.n_customers + data.n_charging) * mean(ng_neighborhoods[data.N_customers, vcat(data.N_customers, data.N_charging)]) : 0
        iteration_params["cycles_lookup_length"] = 0
        iteration_params["implemented_SR3_cuts_count"] = 0
        iteration_params["converged"] = false
        iteration_params["time_limit_reached"] = false

        # Run column generation
        (timed_out, CGLP_results, CG_params) = path_formulation_column_generation!(
            model, z, SR3_constraints,
            data, graph,
            artificial_paths,
            some_paths, path_costs, path_service,
            printlist,
            ;
            method = method,
            use_load = use_load,
            use_time_windows = use_time_windows,
            use_nonlinear_charging = use_nonlinear_charging,
            charging_function = charging_function,
            use_pruned_graph = use_pruned_graph,
            prune_graph_outdegree_sequence = prune_graph_outdegree_sequence,
            exploration_method = exploration_method,
            use_ngroute = use_ngroute,
            ng_neighborhoods = ng_neighborhoods,
            verbose = verbose,
            time_limit = time_limit - (time() - start_time),
            max_iters = max_iters,
        )
        push!(CGLP_all_results, CGLP_results)
        push!(CG_all_params, CG_params)
        if use_ngroute
            push!(CG_all_ng_neighborhoods, copy(ng_neighborhoods))
        end

        if CGLP_results["errored"] || CG_params["cycled"]
            add_message!(printlist, "Terminating column generation...\n", verbose)
            iteration_params["errored"] = true
            break
        end

        iteration_params["CGLP_objective"] = CGLP_results["objective"]
        iteration_params["CG_time_taken"] = CG_params["time_taken"]
        iteration_params["CGLP_artificial"] = CGLP_results["artificial"]
        iteration_params["CG_sp_time_taken_mean"] = CG_params["sp_time_taken_mean"]
        iteration_params["n_CG_iterations"] = CG_params["counter"]
        iteration_params["CG_time_limit_reached"] = CG_params["time_limit_reached"]
        iteration_params["time_limit_reached"] = CG_params["time_limit_reached"]
        CGLP_results["paths"] = collect_path_solution_support(
            CGLP_results, some_paths, data,
        )

        CGIP_results = path_formulation_solve_integer_model!(
            model,
            z,
            artificial_paths,
            printlist,
        )
        push!(CGIP_all_results, CGIP_results)
        if CGIP_results["errored"] 
            add_message!(printlist, "Terminating column generation...\n", verbose)
            iteration_params["errored"] = true
            break
        end

        # Termination criteria
        iteration_params["CGIP_objective"] = CGIP_results["objective"]
        iteration_params["CGIP_time_taken"] = CGIP_results["time_taken"]
        iteration_params["CGIP_artificial"] = CGIP_results["artificial"]
        iteration_params["CG_LP_IP_gap"] = 1.0 - CGLP_results["objective"] / CGIP_results["objective"]
        CGIP_results["paths"] = collect_path_solution_support(
            CGIP_results, some_paths, data, 
        )
        for message in [
            @sprintf("\n"),
            @sprintf("Time taken (CG) (s):  %9.3f s\n", iteration_params["CG_time_taken"]),
            @sprintf("(CGLP) Objective:         %.4e\n", iteration_params["CGLP_objective"]),

            @sprintf("Time taken (IP) (s):  %9.3f s\n", iteration_params["CGIP_time_taken"]),
            @sprintf("(CGIP) Objective:         %.4e\n", iteration_params["CGIP_objective"]),
            @sprintf("%% gap:                %9.3f %%\n", iteration_params["CG_LP_IP_gap"] * 100.0),
        ]
            add_message!(printlist, message, verbose)
        end

        # check if converged
        if (
            !CG_params["time_limit_reached"]
            && CGIP_results["objective"] ≈ CGLP_results["objective"]
        )
            converged = true
            iteration_params["converged"] = true
        end

        if CG_params["converged"] && !continue_flag && !converged && use_ngroute && use_adaptive_ngroute
            # see if any path in solution was non-elementary
            cycles_lookup = detect_cycles_in_path_solution([p for (val, p) in CGLP_results["paths"]])
            entries_changed = 0
            if length(cycles_lookup) > 0 
                delete_paths_with_found_cycles_from_model!(model, z, some_paths, path_costs, path_service, cycles_lookup, data)
                entries_changed = modify_ng_neighborhoods_with_found_cycles!(ng_neighborhoods, cycles_lookup)
                continue_flag = true
                iteration_params["method"] = "use_adaptive_ngroute"
                iteration_params["ngroute_neighborhoods_expanded"] = entries_changed
            end
            add_message!(printlist, "Expanded ng-route ng_neighborhoods by $(entries_changed)\n", verbose)
            add_message!(printlist, "\n", verbose)
        end

        if CG_params["converged"] && !continue_flag && !converged && use_ngroute && use_SR3_cuts
            # if no path in solution was non-elementary, 
            # generate violated SR3 inequalities
            generated_SR3_list = enumerate_violated_path_SR3_inequalities(CGLP_results["paths"], data)
            add_message!(printlist, "Found SR3 cuts:\t\t\t$(length(generated_SR3_list))\n", verbose)
            if length(generated_SR3_list) != 0
                generated_SR3_list = select_representative_violated_path_SR3_inequalities(
                    generated_SR3_list, data,
                )
                add_message!(printlist, "Selected SR3 cuts:\t\t$(length(generated_SR3_list))\n", verbose)
            end

            # sample cuts if too many
            if length(generated_SR3_list) ≤ max_SR3_cuts
                implemented_SR3_list = generated_SR3_list
            elseif randomize_cuts
                implemented_SR3_list = sample(
                    generated_SR3_list, 
                    Weights([val for (val, _) in generated_SR3_list]),
                    max_SR3_cuts, 
                    replace = false,
                )
            else
                implemented_SR3_list = generated_SR3_list[1:max_SR3_cuts]
            end
            add_message!(printlist, "Sampled SR3 cuts:\t\t$(length(implemented_SR3_list))\n", verbose)
            
            if length(implemented_SR3_list) != 0
                if use_lmSR3_cuts
                    implemented_SR3_list = Tuple{Float64, Tuple{Int64, Int64, Int64}, Tuple{Vararg{Int64}}}[
                        (val, S, compute_memory_set_of_lmSRnk_inequality(CGLP_results["paths"], S, 2))
                        for (val, S) in implemented_SR3_list
                    ]
                    SR3_list_todelete, SR3_list_toadd = update_lmSR3_constraint_list!(
                        SR3_list, implemented_SR3_list,
                    )
                    add_lmSR3_constraints_to_path_model!(
                        model, z, some_paths, 
                        SR3_constraints, SR3_list_todelete, SR3_list_toadd,
                    )
                    # append!(SR3_list, implemented_SR3_list)
                    # add_lmSR3_constraints_to_path_model!(
                    #     model, z, some_paths, 
                    #     SR3_constraints, implemented_SR3_list,
                    # )
                    iteration_params["method"] = "use_lmSR3_cuts"
                    iteration_params["implemented_SR3_cuts_count"] = length(implemented_SR3_list)
                    continue_flag = true
                    add_message!(printlist, "Imposed lm-SR3 cuts:\t\t$(length(implemented_SR3_list))\n", verbose)
                    for (val, S, M) in implemented_SR3_list
                        add_message!(printlist, "$S, $M: $val\n", verbose)
                    end
                    add_message!(printlist, "\n", verbose)
                else
                    append!(SR3_list, implemented_SR3_list)
                    # Add violated inequalities to master problem
                    add_SR3_constraints_to_path_model!(
                        model, z, some_paths, 
                        SR3_constraints, implemented_SR3_list, 
                    )
                    iteration_params["method"] = "use_SR3_cuts"
                    iteration_params["implemented_SR3_cuts_count"] = length(implemented_SR3_list)
                    continue_flag = true
                    add_message!(printlist, "Imposed SR3 cuts:\t\t$(length(implemented_SR3_list))\n", verbose)
                    for (val, S) in implemented_SR3_list
                        add_message!(printlist, "$S: $val\n", verbose)
                    end
                    add_message!(printlist, "\n", verbose)
                end
            end
        end


        current_time_taken = time() - start_time
        add_message!(
            printlist,
            @sprintf("Total time taken (s): %9.3f s\n\n", current_time_taken),
            verbose,
        )
        iteration_params["time_taken"] = current_time_taken
        push!(all_params, iteration_params)

        if !(time_limit > current_time_taken)
            iteration_params["time_limit_reached"] = true
            break
        end
        
        if !continue_flag
            break
        end
    end

    return (
        CGLP_all_results, CGIP_all_results, CG_all_params, CG_all_ng_neighborhoods, all_params, printlist, 
        some_paths, model, z, SR3_constraints
    )
end


function collect_path_solution_support(
    results, 
    paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}},
        Vector{Path},
    },
    data::EVRPData,
    ;
)
    results_paths = Tuple{Float64, Path}[]
    for ((key, p), val) in pairs(results["z"])
        if key in keys(paths)
            if p <= length(paths[key])
                if val > 1e-5
                    push!(results_paths, (val, paths[key][p]))
                end
            end
        end
    end

    sort!(
        results_paths,
        by = x -> (-compute_path_cost(data, x[2]), -x[1]),
    )
    return results_paths
end

function get_path_solution_metrics(
    sol_paths::Vector{Tuple{Float64, Path}},
    data::EVRPData,
)
    metrics = Dict{String, Any}()

    sol_paths = [(Int(round(i)), p) for (i, p) in sol_paths if length(p.arcs) > 1]
    sol_paths = vcat([repeat([p], i) for (i, p) in sol_paths]...)

    metrics["artificial"] = any([p.artificial for p in sol_paths])

    metrics["n_vehicles_used"] = length(sol_paths)
    metrics["path_lengths"] = [length(p.arcs) for p in sol_paths]
    metrics["mean_path_length"] = mean(metrics["path_lengths"])
    metrics["subpath_lengths"] = [length(s.arcs) for p in sol_paths for s in p.subpaths]
    metrics["mean_subpath_length"] = mean(metrics["subpath_lengths"])
    metrics["ps_lengths"] = [length(p.subpaths) for p in sol_paths]
    metrics["mean_ps_length"] = mean(metrics["ps_lengths"])
    
    metrics["n_recharges"] = sum(metrics["ps_lengths"] .- 1, init=0)

    metrics["path_costs"] = [compute_path_cost(data, p) for p in sol_paths]
    metrics["total_cost"] = sum(metrics["path_costs"])
    metrics["mean_path_cost"] = mean(metrics["path_costs"])

    metrics["path_driving_times"] = [
        sum(data.t[a...] for a in p.arcs)
        for p in sol_paths
    ]
    metrics["path_charging_times"] = [
        sum([a.time_diff for a in p.charging_arcs], init=0)
        for p in sol_paths
    ]
    metrics["path_total_times"] = [
        p.subpaths[end].current_time
        for p in sol_paths
    ]
    metrics["path_waiting_times"] = [
        l - dt - ct
        for (l, dt, ct) in zip(
            metrics["path_total_times"],
            metrics["path_driving_times"],
            metrics["path_charging_times"],
        )
    ]
    metrics["mean_path_driving_time"] = mean(metrics["path_driving_times"]) / data.T
    metrics["mean_path_charging_time"] = mean(metrics["path_charging_times"]) / data.T
    metrics["mean_path_total_time"] = mean(metrics["path_total_times"]) / data.T
    metrics["mean_path_waiting_time"] = mean(metrics["path_waiting_times"]) / data.T

    metrics["path_nodes"] = [
        vcat(p.arcs[1][1], [a[2] for a in p.arcs])
        for p in sol_paths
    ]
    metrics["path_nodes_print"] = [
        [data.node_labels[i] for i in nodes]
        for nodes in metrics["path_nodes"]
    ]

    return metrics
end

function write_path_solution_metrics!(
    metrics::Dict{String, Any},
    fp::String,
)
    open(fp, "w+") do io
        @printf(io, "Total cost:          %12d\n", metrics["total_cost"])
        @printf(io, "No. of recharges:    %12d\n", metrics["n_recharges"])
        @printf(io, "No. of vehicles:     %12d\n", metrics["n_vehicles_used"])
        print(io, "\n")
        for i in 1:length(metrics["path_costs"])
            @printf(io, "Route %2d: Cost %12d\n", i, metrics["path_costs"][i])
            for n in metrics["path_nodes_print"][i]
                @printf(io, "  %4s", n)
                if !(startswith(n, "C"))
                    print(io, "\n")
                end
            end
            print(io, "\n")
        end
    end
    return
end
