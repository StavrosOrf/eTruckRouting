
using JuMP
using Gurobi
using Accessors
using Graphs

function arc_formulation_preprocessing!(
    data::EVRPData,
    graph::EVRPGraph,
)
    @reset graph.A = Set([
        (i, j) 
        for (i, j) in graph.A
            if !(
                j == data.N_depots[1]
                || i == data.N_depots[2]
            )
    ])
    push!(graph.A, (data.N_depots[1], data.N_depots[2]))

    G = SimpleDiGraph{Int}(data.n_nodes)
    for (i, j) in graph.A
        add_edge!(G, i, j)
    end
    @reset graph.G = G
    return data, graph
end

function arc_formulation(
    data::EVRPData,
    graph::EVRPGraph,
    n_vehicles::Int,
    ;
    Env::Gurobi.Env = Gurobi.Env(),
    impose_time_windows::Bool = true,
    impose_charge::Bool = true,
    impose_load::Bool = true,
    time_limit_sec::Float64 = 60.0,
    MIPGap::Float64 = 1e-2,
    n_threads::Int = 1,
)

    model = Model(() -> Gurobi.Optimizer(Env))
    JuMP.set_optimizer_attribute(model, "MIPGap", MIPGap)
    JuMP.set_time_limit_sec(model, time_limit_sec)
    JuMP.set_optimizer_attribute(model, "Threads", n_threads) # 0 is default
    
    @variable(model, x[graph.A, 1:n_vehicles], Bin)
    @variable(model, 0 ≤ t[1:data.n_nodes, 1:n_vehicles] ≤ data.T)
    @variable(model, 0 ≤ b[1:data.n_nodes, 1:n_vehicles] ≤ data.B)
    @variable(model, 0 ≤ d[1:data.n_nodes, 1:n_vehicles] ≤ data.C)
    @variable(model, τ[1:data.n_nodes, 1:n_vehicles] ≥ 0)

    @constraint(
        model, 
        [k in 1:n_vehicles],
        sum(x[(i, j), k] for i in data.N_depots, j in outneighbors(graph.G, i)) == 1
    )
    @constraint(
        model, 
        [k in 1:n_vehicles],
        sum(x[(i, j), k] for j in data.N_depots, i in inneighbors(graph.G, j)) == 1
    )
    @constraint(
        model,
        [k in 1:n_vehicles, i in union(data.N_customers, data.N_charging)],
        sum(x[(j, i), k] for j in inneighbors(graph.G, i))
        == sum(x[(i, j), k] for j in outneighbors(graph.G, i))
    )
    @constraint(
        model, 
        [i in data.N_customers],
        sum(x[(i, j), k] for j in outneighbors(graph.G, i) for k in 1:n_vehicles)
        == 1
    )
    if impose_load
        @constraint(
            model,
            [k in 1:n_vehicles, (i, j) in graph.A],
            d[j, k] ≥ d[i, k] + data.d[i] - (1 - x[(i, j), k]) * data.C
        )
        @constraint(
            model,
            [k in 1:n_vehicles],
            d[data.N_depots[1], k] == 0
        )
    end
    @constraint(
        model,
        [k in 1:n_vehicles, (i, j) in graph.A],
        t[j, k] ≥ t[i, k] + τ[i, k] + data.t[i, j] - (1 - x[(i, j), k]) * data.T
    )
    @constraint(
        model,
        [k in 1:n_vehicles, i in union(data.N_customers, data.N_depots)],
        τ[i, k] == 0
    )
    if impose_time_windows
        @constraint(
            model,
            [k in 1:n_vehicles, i in data.N_nodes],
            data.α[i] ≤ t[i, k] ≤ data.β[i]
        )
    end

    if impose_charge
        @constraint(
            model,
            [k in 1:n_vehicles, (i, j) in graph.A],
            b[j, k] ≤ b[i, k] + τ[i, k] - data.q[i, j] + (1 - x[(i, j), k]) * data.B
        )
        @constraint(
            model,
            [k in 1:n_vehicles],
            b[data.N_depots[1], k] == data.B
        )
    end

    @objective(
        model, 
        Min, 
        sum(
            data.c[i, j] * x[(i, j), k] 
            for (i, j) in graph.A 
            for k in 1:n_vehicles
        )
    )

    optimize!(model)

    return Dict(
        "objective_value" => JuMP.objective_value(model),
        "objective_bound" => JuMP.objective_bound(model),
        "time_taken" => JuMP.solve_time(model),
        "x" => [
            (i, j, k) 
            for (i, j) in graph.A 
            for k in 1:n_vehicles
            if JuMP.value(x[(i, j), k]) > 0.5
        ],
    )
end

