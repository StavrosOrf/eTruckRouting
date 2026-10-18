using DataStructures
using Random
using Combinatorics
using StatsBase
using Distributions
using Distances
using Printf
using Graphs
using LinearAlgebra
using Plots
using ColorSchemes
using CSV
using DataFrames

abstract type Label end

Base.@kwdef mutable struct Subpath
    n_customers::Int
    starting_node::Int
    starting_time::Int
    starting_charge::Int
    current_node::Int = starting_node
    arcs::Vector{NTuple{2, Int}} = NTuple{2, Int}[]
    current_time::Int = starting_time
    current_charge::Int = starting_charge
    load::Int = 0
    served::Vector{Int} = zeros(Int, n_customers)
    artificial::Bool = false
end

Base.copy(s::Subpath) = Subpath(
    n_customers = s.n_customers,
    starting_node = s.starting_node,
    starting_time = s.starting_time,
    starting_charge = s.starting_charge,
    current_node = s.current_node,
    arcs = copy(s.arcs),
    current_time = s.current_time,
    current_charge = s.current_charge,
    load = s.load,
    served = copy(s.served),
    artificial = s.artificial,
)

Base.show(io::IO, s::Subpath) = begin
    if s.artificial
        message = """Subpath (artificial):
        """
    else
        message = """Subpath:
        """
    end
    message = message * """
    ($(s.starting_node), $(s.starting_time), $(s.starting_charge)) -> ($(s.current_node), $(s.current_time), $(s.current_charge))
    arcs:           $(s.arcs)
    served:         $(s.served)
    load:           $(s.load)
    """
    print(io, message)
end

Base.isequal(s1::Subpath, s2::Subpath) = begin 
    (
        s1.n_customers == s2.n_customers
        && s1.starting_node == s2.starting_node
        && s1.starting_time == s2.starting_time
        && s1.starting_charge == s2.starting_charge
        && s1.current_node == s2.current_node
        && s1.arcs == s2.arcs
        && s1.current_time == s2.current_time
        && s1.current_charge == s2.current_charge
        && s1.load == s2.load
        && s1.served == s2.served
        && s1.artificial == s2.artificial
    )
end

Base.@kwdef struct ChargingArc
    node::Int
    time_start::Int
    time_end::Int
    time_diff::Int
    charge_start::Int
    charge_end::Int
    charge_diff::Int
end

Base.isequal(a1::ChargingArc, a2::ChargingArc) = (
    a1.node == a2.node
    && a1.time_start == a2.time_start
    && a1.time_end == a2.time_end
    && a1.time_diff == a2.time_diff
    && a1.charge_start == a2.charge_start
    && a1.charge_end == a2.charge_end
    && a1.charge_diff == a2.charge_diff
)

Base.copy(a::ChargingArc) = ChargingArc(
    node = a.node,
    time_start = a.time_start,
    time_end = a.time_end,
    time_diff = a.time_diff,
    charge_start = a.charge_start,
    charge_end = a.charge_end,
    charge_diff = a.charge_diff,
)

Base.@kwdef mutable struct Path
    subpaths::Vector{Subpath}
    charging_arcs::Vector{ChargingArc}
    served::Vector{Int} = sum(s.served for s in subpaths)
    load::Int = sum(s.load for s in subpaths)
    arcs::Vector{NTuple{2, Int}} = vcat([s.arcs for s in subpaths]...)
    artificial::Bool = false
end

Base.isequal(p1::Path, p2::Path) = (
    all(isequal(s1, s2) for (s1, s2) in zip(p1.subpaths, p2.subpaths))
    && all(isequal(a1, a2) for (a1, a2) in zip(p1.charging_arcs, p2.charging_arcs))
    && p1.served == p2.served
    && p1.load == p2.load
    && p1.arcs == p2.arcs
    && p1.artificial == p2.artificial
)

Base.copy(p::Path) = Path(
    subpaths = [copy(s) for s in p.subpaths],
    charging_arcs = [copy(a) for a in p.charging_arcs],
    served = copy(p.served),
    load = copy(p.load),
    arcs = copy(p.arcs),
    artificial = p.artificial,
)

function get_nodes(
    p::Path,
)
    return vcat(
        [a[1] for a in p.arcs],
        [p.arcs[end][2]]
    )
end

struct EVRPData
    n_depots::Int
    n_customers::Int
    n_vehicles::Int
    n_charging::Int
    n_depots_charging::Int
    n_nodes::Int
    N_customers::UnitRange{Int}
    N_depots::UnitRange{Int}
    N_vehicles::UnitRange{Int}
    N_charging::UnitRange{Int}
    N_depots_charging::UnitRange{Int}
    N_nodes::UnitRange{Int}
    node_labels::Dict{Int, String}
    depot_pattern::String
    customer_pattern::String
    charging_pattern::String
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    customer_coords::Array{Float64, 2}
    depot_coords::Array{Float64, 2}
    charging_coords::Array{Float64, 2}
    coords::Array{Float64, 2}
    distances::Array{Float64, 2}
    V::Dict{Int, Vector{Int}}
    v_start::Dict{Int, Int}
    v_end::Dict{Int, Int}
    c::Array{Int, 2}
    t::Array{Int, 2}
    q::Array{Int, 2}
    d::Vector{Int}
    C::Int
    T::Int
    α::Vector{Int}
    β::Vector{Int}
    inverse_refueling_rate::Float64
    B::Int
    travel_cost_coeff::Int
end

struct EVRPGraph
    G::SimpleDiGraph{Int}
    node_labels::Dict{Int, String}
    c::Array{Int, 2}
    t::Array{Int, 2}
    q::Array{Int, 2}
    d::Vector{Int}
    C::Int
    N_customers::UnitRange{Int}
    n_customers::Int
    N_depots::UnitRange{Int}
    n_depots::Int
    N_charging::UnitRange{Int}
    n_charging::Int
    N_depots_charging::UnitRange{Int}
    n_depots_charging::Int
    N_nodes::UnitRange{Int}
    n_nodes::Int
    A::Set{Tuple{Int, Int}}
    T::Int
    B::Int
    inverse_refueling_rate::Float64
    α::Vector{Int}
    β::Vector{Int}
    min_t::Vector{Int}
    min_q::Vector{Int}
end

function add_message!(
    printlist::Vector{String}, 
    message::String, 
    verbose::Bool,
)
    push!(printlist, message)
    if verbose
        print(message)
    end
end

function compute_subpath_cost(
    data::EVRPData,
    s::Subpath,
    M::Int = Int(1e10),
    ;
)
    if s.artificial 
        return M
    elseif length(s.arcs) == 0
        return 0
    end

    travel_cost = data.travel_cost_coeff * sum(
        data.c[a...] for a in s.arcs
    )
    return travel_cost
end

function compute_subpath_modified_cost(
    data::EVRPData,
    s::Subpath,
    κ::Dict{Int, Float64},
    μ::Dict{Int, Float64},
    ν::Vector{Float64}, 
    ;
    verbose = false,
)
    reduced_cost = compute_subpath_cost(data, s)
    verbose && @printf("Subpath cost: \t\t%11.3f\n", reduced_cost)

    service_cost = 0.0
    for (j, c) in enumerate(s.served)
        service_cost += (c * -ν[j])
    end
    verbose && @printf("Service cost: \t\t%11.3f\n", service_cost)
    reduced_cost += service_cost

    if s.starting_node in data.N_depots
        if s.starting_time == 0.0 && s.starting_charge == data.B
            verbose && @printf("Starting depot cost: \t%11.3f\n", (- κ[s.starting_node]))
            reduced_cost = reduced_cost - κ[s.starting_node]
        end
    end

    if s.current_node in data.N_depots
        verbose && @printf("Ending depot cost: \t%11.3f\n", ( - μ[s.current_node]))
        reduced_cost = reduced_cost - μ[s.current_node]
    end

    verbose && @printf("Total modified cost: \t%11.3f\n\n", reduced_cost)

    return reduced_cost
end

function compute_subpath_costs(
    data::EVRPData,
    all_subpaths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Subpath},
    },
    M::Int = Int(1e10),
    ;
)
    subpath_costs = Dict(
        key => Int[
            compute_subpath_cost(data, s, M;)
            for s in all_subpaths[key]
        ]
        for key in keys(all_subpaths)
    )
    return subpath_costs
end

function compute_subpath_service(
    data::EVRPData,
    all_subpaths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Subpath},
    },
)
    subpath_service = Dict(
        (key, i) => Int[
            s.served[i]
            for s in all_subpaths[key]
        ]
        for key in keys(all_subpaths), i in 1:data.n_customers
    )
    return subpath_service
end

function compute_path_cost(
    data::EVRPData,
    p::Path,
    M::Int = Int(1e10),
    ;
    verbose = false,
)
    subpath_costs = length(p.subpaths) > 0 ? sum(compute_subpath_cost(data, s, M) for s in p.subpaths) : 0
    verbose && @printf("Subpath costs: \t\t%11.3f\n", subpath_costs)
    
    return subpath_costs
end

function compute_path_modified_cost(
    data::EVRPData,
    p::Path,
    κ::Dict{Int, Float64},
    μ::Dict{Int, Float64},
    ν::Vector{Float64}, 
    ;
    verbose = false,
)
    reduced_cost = 0.0
    for s in p.subpaths
        reduced_cost += compute_subpath_modified_cost(data, s, κ, μ, ν, verbose = verbose)
    end

    verbose && @printf("Total modified cost: \t%11.3f\n\n", reduced_cost)

    return reduced_cost
end

function compute_path_modified_cost(
    data::EVRPData,
    p::Path,
    κ::Dict{Int, Float64},
    μ::Dict{Int, Float64},
    ν::Vector{Float64}, 
    λ::Dict{NTuple{3, Int}, Float64},
    ;
    verbose = false,
)
    reduced_cost = 0.0
    for s in p.subpaths
        reduced_cost += compute_subpath_modified_cost(data, s, κ, μ, ν, verbose = verbose)
    end
    SR3_costs = sum(
        [val * check_path_in_SR3_constraint(p, S)
        for (S, val) in pairs(λ)],
        init=0.0,
    )
    verbose && @printf("SR3 costs: \t\t%11.3f\n", SR3_costs)
    reduced_cost += SR3_costs

    verbose && @printf("Total modified cost: \t%11.3f\n\n", reduced_cost)

    return reduced_cost
end

function compute_path_modified_cost(
    data::EVRPData,
    p::Path,
    κ::Dict{Int, Float64},
    μ::Dict{Int, Float64},
    ν::Vector{Float64}, 
    λ::Dict{Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}}, Float64},
    ;
    verbose = false,
)
    reduced_cost = 0.0
    for s in p.subpaths
        reduced_cost += compute_subpath_modified_cost(data, s, κ, μ, ν, verbose = verbose)
    end
    lmSR3_costs = sum(
        [val * compute_path_coefficient_in_lmSRnk_constraint(p, S, M, 2)
        for ((S, M), val) in pairs(λ)],
        init=0.0,
    )
    verbose && @printf("lm-SR3 costs: \t%11.3f\n", lmSR3_costs)
    reduced_cost += lmSR3_costs

    verbose && @printf("Total modified cost: \t%11.3f\n\n", reduced_cost)

    return reduced_cost
end

function compute_path_costs(
    data::EVRPData,
    all_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Path},
    },
    M::Int = Int(1e10),
    ;
)
    path_costs = Dict(
        key => Int[
            compute_path_cost(data, p, M;)
            for p in all_paths[key]
        ]
        for key in keys(all_paths)
    )
    return path_costs
end

function compute_path_service(
    data::EVRPData,
    all_paths::Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Path},
    },
)
    path_service = Dict(
        (key, i) => Int[
            p.served[i]
            for p in all_paths[key]
        ]
        for key in keys(all_paths), i in 1:data.n_customers
    )
    return path_service
end

function generate_locations(
    ;
    n_depots::Int, 
    n_customers::Int,
    n_charging::Int,
    depot_pattern::String,
    customer_pattern::String,
    charging_pattern::String,
    customer_spread::Float64 = 1e-3,
    xmin::Float64, 
    xmax::Float64,
    ymin::Float64,
    ymax::Float64,
    seed::Int,
    separate_source_sink::Bool = false,
    data_dir::String = "data/",
    pr::Float64 = 100.0,
)
    function complex_coords(
        n::Int,
        xmin::Float64 = -1.0,
        xmax::Float64 = 1.0,
        ymin::Float64 = -1.0,
        ymax::Float64 = 1.0,
    )
        coords = hcat(
            [round.(collect(reim(exp(2 * pi * j * im / n))), digits = 5)
            for j in 1:n]...
        )
        unit_coords = (coords .+ 1) ./ 2 # fit into 0-1 box
        return [xmin; ymin] .+ unit_coords .* [xmax - xmin; ymax - ymin]
    end

    function get_rectangle(
        n::Int,
    )
        a = Int(ceil(sqrt(n)))
        b = n ÷ a
        while b * a != n
            a -= 1
            b = n ÷ a
        end
        return a, b
    end

    function grid_coords(
        a::Int,
        b::Int,
        xmin::Float64 = -1.0,
        xmax::Float64 = 1.0,
        ymin::Float64 = -1.0,
        ymax::Float64 = 1.0,
    )
        return hcat(
            [
                [
                    xmin + ((xmax - xmin) * i) / (a - 1), 
                    ymin + ((ymax - ymin) * j) / (b - 1),
                ]
                for i in 0:a-1, j in 0:b-1
            ]...
        )
    end

    function circle_packing_coords(
        n::Int,
        xmin::Float64 = -1.0,
        xmax::Float64 = 1.0,
        ymin::Float64 = -1.0,
        ymax::Float64 = 1.0,
        ;
        data_dir::String = "data/",
    )
        lines = readlines(joinpath(data_dir, "cci_coords/cci$n.txt"))
        coords = hcat(
            [
                [parse(Float64, x[2]), parse(Float64, x[3])]
                for x in split.(lines)
            ]...
        )
        unit_coords = (coords .+ 1) ./ 2 # fit into 0-1 box
        return [xmin; ymin] .+ unit_coords .* [xmax - xmin; ymax - ymin]
    end

    function complex_coords_random(
        n::Int, 
        seed::Int,
        xmin::Float64 = -1.0,
        xmax::Float64 = 1.0,
        ymin::Float64 = -1.0,
        ymax::Float64 = 1.0,
        customer_spread::Float64 = 0.0,
    )
        Random.seed!(seed)
        unit_coords = zeros(Float64, 2, n)
        while true
            deg = rand(n) * 2 * pi
            scale = rand(n)
            coords = hcat(
                [round.(collect(reim(scale[j] * exp(deg[j] * im))), digits = 5)
                for j in 1:n]...
            )
            unit_coords = (coords .+ 1) ./ 2 # fit into 0-1 box
            d = Distances.pairwise(Euclidean(), unit_coords, dims = 2)
            if minimum(d + I * customer_spread) ≥ customer_spread
                break
            end
        end
        return [xmin; ymin] .+ unit_coords .* [xmax - xmin; ymax - ymin]
    end

    function uniform_coords_random(        
        n::Int, 
        seed::Int,
        xmin::Float64 = -1.0,
        xmax::Float64 = 1.0,
        ymin::Float64 = -1.0,
        ymax::Float64 = 1.0,
        customer_spread::Float64 = 0.0,
    )
        Random.seed!(seed)
        coords = zeros(Float64, 2, n)
        n_blocks = Int(round((xmax - xmin) * (ymax - ymin)))
        while true
            for (i, (y, x)) in enumerate(Iterators.product(
                ymin:ymax-1,
                xmin:xmax-1,
            ))
                coords[:, i:n_blocks:n] .= [x; y] .+ rand(Float64, 2, length(i:n_blocks:n)) 
            end
            d = Distances.pairwise(Euclidean(), coords, dims = 2)
            if minimum(d + I * customer_spread) ≥ customer_spread
                break
            end
        end
        return coords
    end

    if customer_pattern == "random_box"
        customer_coords = uniform_coords_random(n_customers, seed, xmin, xmax, ymin, ymax, customer_spread)
    elseif customer_pattern == "random_uniform_polar"
        customer_coords = complex_coords_random(n_customers, seed, xmin, xmax, ymin, ymax, customer_spread)
    end
    if depot_pattern == "circular"
        depot_coords = complex_coords(n_depots, xmin, xmax, ymin, ymax)
    elseif depot_pattern == "grid"
        (a, b) = get_rectangle(n_depots)
        depot_coords = grid_coords(a, b, xmin, xmax, ymin, ymax)
    elseif depot_pattern == "circular_packing"
        depot_coords = circle_packing_coords(n_depots, xmin, xmax, ymin, ymax, data_dir = data_dir)
    end
    if separate_source_sink
        depot_coords = hcat(depot_coords, depot_coords)
    end
    
    if charging_pattern == "circular"
        charging_coords = complex_coords(n_depots, xmin, xmax, ymin, ymax)
    elseif charging_pattern == "grid"
        (a, b) = get_rectangle(n_charging)
        charging_coords = grid_coords(a, b, xmin, xmax, ymin, ymax)
    elseif charging_pattern == "grid_clipped"
        a = Int(xmax - xmin + 1)
        b = Int(ymax - ymin + 1)
        charging_coords = grid_coords(a, b, xmin, xmax, ymin, ymax)
        charging_coords = hcat(setdiff(eachcol(charging_coords), [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])...)
    elseif charging_pattern == "circular_packing"
        charging_coords = circle_packing_coords(n_depots, xmin, xmax, ymin, ymax, data_dir = data_dir)
    end

    customer_coords .*= pr 
    depot_coords .*= pr 
    charging_coords .*= pr 

    coords = hcat(
        customer_coords,
        depot_coords,
        charging_coords,
    )

    distances = Distances.pairwise(Euclidean(), coords, dims=2)

    return (
        customer_coords,
        depot_coords,
        charging_coords,
        coords,
        distances,
    )
end


function generate_instance(
    ;
    n_depots::Int, 
    n_customers::Int,
    n_charging::Int,
    n_vehicles::Int,
    depot_pattern::String,
    customer_pattern::String,
    charging_pattern::String,
    customer_spread::Float64,
    xmin::Float64,
    xmax::Float64,
    ymin::Float64,
    ymax::Float64,
    T::Int, # Length of time horizon
    seed::Int,
    B::Int, # Battery capacity
    inverse_refueling_rate::Float64, # Inverse refueling rate
    travel_cost_coeff::Int,
    load_tolerance::Float64,
    load_precision::Int,
    vehicle_locations::String = "random",
    time_windows_distribution::String = "random",
    time_windows_min_width::Float64 = 0.0,
    time_windows_max_width::Float64 = 1.0,
    separate_source_sink::Bool = false,
    data_dir::String = "data/",
)

    n_depots_ = (separate_source_sink ? 2 * n_depots : n_depots)
    n_nodes = n_depots_ + n_customers + n_charging

    seeds = abs.(rand(MersenneTwister(seed), Int, 6))

    N_customers = 1:n_customers
    N_depots = n_customers+1:n_customers+n_depots_
    N_vehicles = 1:n_vehicles
    N_charging = n_customers+n_depots_+1:n_customers+n_depots_+n_charging
    N_depots_charging = n_customers+1:n_customers+n_depots_+n_charging
    N_nodes = 1:n_customers+n_depots_+n_charging

    node_labels = merge(Dict(
        i => "Depot $ind" for (ind, i) in enumerate(N_depots)
    ), Dict(
        i => "Customer $ind" for (ind, i) in enumerate(N_customers)
    ), Dict(
        i => "Charging $ind" for (ind, i) in enumerate(N_charging)
    ))

    (
        customer_coords,
        depot_coords,
        charging_coords,
        coords,
        distances,
    ) = generate_locations(
        n_depots = n_depots, 
        n_customers = n_customers,
        n_charging = n_charging,
        depot_pattern = depot_pattern,
        customer_pattern = customer_pattern,
        charging_pattern = charging_pattern,
        customer_spread = customer_spread,
        xmin = xmin,
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        seed = seeds[1],
        separate_source_sink = separate_source_sink,
        data_dir = data_dir,
    )

    if vehicle_locations == "random"
        start_depots = StatsBase.sample(MersenneTwister(seeds[4]), N_depots, n_vehicles, replace = true)
        V = Dict(i => findall(x -> x==i, start_depots) for i in N_depots)
        v_start = Dict(i => length(V[i]) for i in N_depots)
        v_end  = Dict(i => 1 for i in N_depots)
    elseif vehicle_locations == "gradiated"
        @assert depot_pattern == "grid"
        @assert n_depots == 4
        @assert time_windows_distribution == "gradiated"
        V = Dict(
            N_depots[1] => collect(1:(n_vehicles÷2)),
            N_depots[3] => collect((n_vehicles÷2)+1:n_vehicles),
            N_depots[2] => Int[],
            N_depots[4] => Int[],
        )
        v_start = Dict(i => length(V[i]) for i in N_depots)
        v_end = Dict(
            N_depots[1] => 0,
            N_depots[3] => 0,
            N_depots[2] => 1,
            N_depots[4] => 1,
        )
    end
    
    # c = Int.(round.(100 .* distances))
    # t = Int.(round.(100 .* distances)) # travel times are integer
    # q = Int.(round.(100 .* distances)) # charge costs are integer
    c = Int.(round.(distances .* 100))
    t = Int.(round.(distances .* 100))
    q = Int.(round.(distances .* 100 .* inverse_refueling_rate))

    Random.seed!(seeds[2])
    d = (rand(Truncated(Geometric(0.5), 0, 4), n_customers) .+ 1) * load_precision
    d = vcat(d, repeat([0], n_depots_ + n_charging))
    C = sum(d) * load_tolerance / n_vehicles
    C = Int(round(C / load_precision) * load_precision)

    if time_windows_distribution == "random"
        (α, β) = generate_time_windows(
            0, T, n_customers, seeds[5],
            time_windows_min_width,
            time_windows_max_width,
        )
    elseif time_windows_distribution == "gradiated"
        n_blocks = Int(round((xmax - xmin) * (ymax - ymin)))
        n_customers_block = Int(n_customers / ((xmax - xmin) * (ymax - ymin)))
        Random.seed!(seeds[5])
        seeds_TW = abs.(rand(Int, n_blocks))
        T_div = Int(round(T / (xmax - xmin)))
        α = zeros(Int, n_customers)
        β = zeros(Int, n_customers)
        for (i, (y, x)) in enumerate(Iterators.product(
            ymin:ymax-1,
            xmin:xmax-1,
        ))
            α_, β_ = generate_time_windows(
                Int(T_div * x), Int(T_div * (x+1)), n_customers_block,
                seeds_TW[i],
                time_windows_min_width,
                time_windows_max_width,
            )
            α[i:n_blocks:n_customers] .= α_
            β[i:n_blocks:n_customers] .= β_
        end
    end
    α_charge = vcat(α, repeat([0], n_depots_ + n_charging))
    β_charge = vcat(β, repeat([T], n_depots_ + n_charging))

    data = EVRPData(
        n_depots_,
        n_customers,
        n_vehicles,
        n_charging,
        n_depots_ + n_charging,
        n_nodes,
        N_customers,
        N_depots,
        N_vehicles,
        N_charging,
        N_depots_charging,
        N_nodes,
        node_labels,
        depot_pattern,
        customer_pattern,
        charging_pattern,
        xmin, 
        xmax, 
        ymin,
        ymax, 
        customer_coords,
        depot_coords,
        charging_coords,
        coords,
        distances,
        V,
        v_start,
        v_end,
        c,
        t,
        q,
        d,
        C,
        T,
        α_charge,
        β_charge,
        inverse_refueling_rate,
        Int(round(B * inverse_refueling_rate)),
        travel_cost_coeff,
    )
    return data
end


function write_instance_evrptw_format(
    data::EVRPData,
    fp::String,
    ;
    multidepot::Bool = false,
    pr::Float64 = 100.0,
)

    depot_df = DataFrame()
    depot_df[!, :StringID] = ["D$(i-data.n_customers)" for i in data.N_depots]
    depot_df[!, :Type] = ["d" for i in data.N_depots]
    depot_coords = Int.(floor.(data.depot_coords))'
    depot_df[!, :x] .= depot_coords[:,1]
    depot_df[!, :y] .= depot_coords[:,2]
    depot_df[!, :demand] .= 0
    depot_df[!, :ReadyTime] .= Int.(floor.(data.α[data.N_depots] ./ pr))
    depot_df[!, :DueDate] .= Int.(floor.(data.β[data.N_depots] ./ pr))
    depot_df[!, :ServiceTime] .= 0

    charging_df = DataFrame()
    charging_df[!, :StringID] = ["S$(i-data.n_customers-data.n_depots)" for i in data.N_charging]
    charging_df[!, :Type] = ["f" for i in data.N_charging]
    charging_coords = Int.(floor.(data.charging_coords))'
    charging_df[!, :x] .= charging_coords[:,1]
    charging_df[!, :y] .= charging_coords[:,2]
    charging_df[!, :demand] .= 0
    charging_df[!, :ReadyTime] .= Int.(floor.(data.α[data.N_charging] ./ pr))
    charging_df[!, :DueDate] .= Int.(floor.(data.β[data.N_charging] ./ pr))
    charging_df[!, :ServiceTime] .= 0

    customer_df = DataFrame()
    customer_df[!, :StringID] = ["C$i" for i in data.N_customers]
    customer_df[!, :Type] = ["c" for i in data.N_customers]
    customer_coords = Int.(floor.(data.customer_coords))'
    customer_df[!, :x] .= customer_coords[:,1]
    customer_df[!, :y] .= customer_coords[:,2]
    customer_df[!, :demand] .= 0
    customer_df[!, :ReadyTime] .= Int.(floor.(data.α[data.N_customers] ./ pr))
    customer_df[!, :DueDate] .= Int.(floor.(data.β[data.N_customers] ./ pr))
    customer_df[!, :ServiceTime] .= 0

    df = vcat(
        depot_df,
        charging_df,
        customer_df,
    )

    # Write the DataFrame to a file with fixed-width columns
    open(fp, "w") do io
        for col in names(df)
            print(io, lpad(col, 10))
        end
        print(io, "\n")
        for row in eachrow(df)
            for col in names(row)
                print(io, lpad(row[col], 10))
            # println(io, rpad(row[:StringID], 10), rpad(row[:Type], 5), 
            #             lpad(row[:x], 10), lpad(row[:y], 10), 
            #             lpad(row[:demand], 10), lpad(row[:ReadyTime], 10), 
            #             lpad(row[:DueDate], 10), lpad(row[:ServiceTime], 10))
            end
            print(io, "\n")
        end
    end

    B_ = data.B ./ (pr * data.inverse_refueling_rate)
    C = data.C

    open(fp, "a+") do io
        println(io, "")
        println(io, "Vehicle fuel tank capacity /$B_/")
        println(io, "Vehicle load capacity /$C/")
        println(io, "Fuel consumption rate /1.0/")
        println(io, "Inverse refueling rate /$(inverse_refueling_rate)/")
        println(io, "Average Velocity /1.0/")
    end

    if !multidepot
        return
    end

    open(fp, "a+") do io
        println(io, "")
        println(io, "Vehicle start locations")
        for i in data.N_depots
            @printf(io, "D%d:   %2d\n", i-data.n_customers, data.v_start[i])
        end
        println(io, "Vehicle end locations")
        for i in data.N_depots
            @printf(io, "D%d:   %2d\n", i-data.n_customers, data.v_end[i])
        end
    end

    return
end

function generate_graph_from_data(
    data::EVRPData,
    ;
    sparse::Bool = false,
    sparse_prob::String = "linear",
    sparse_linear_max_q::Float64 = 2.0 * Float64(data.B),
    sparse_linear_min_q::Float64 = Float64(minimum(data.q)),
)

    tc_depot = vec(minimum(data.t[:,data.N_depots], dims = 2))
    cc_depot_charging = vec(minimum(data.q[:,data.N_depots_charging], dims = 2))

    A = Set{Tuple{Int, Int}}()
    Random.seed!(0)
    for i in data.N_nodes, j in data.N_nodes
        if i == j && !(i in data.N_depots)
            continue
        end
        min_charge = data.q[i,j]
        min_time = data.t[i,j]
        if i in data.N_customers
            min_charge += cc_depot_charging[i]
            min_time += tc_depot[i]
        elseif i in data.N_charging
            min_time += tc_depot[i]
        end
        if j in data.N_customers
            min_charge += cc_depot_charging[j]
            min_time += tc_depot[j]
        elseif j in data.N_charging
            min_time += tc_depot[j]
        end
        if min_charge > data.B || min_time > data.T
            continue
        end
        if sparse
            threshold_prob = (sparse_linear_max_q - data.q[i,j]) / (sparse_linear_max_q - sparse_linear_min_q)
            if rand() < threshold_prob
                push!(A, (i, j))
            elseif !(i in data.N_customers || j in data.N_customers)
                push!(A, (i, j))
            end
        else
            if min_time == 0 || min_charge == 0
                continue
            end
            push!(A, (i, j))
        end
    end

    G = SimpleDiGraph{Int}(data.n_nodes)
    for (i, j) in A
        add_edge!(G, i, j)
    end

    return EVRPGraph(
        G,
        copy(data.node_labels),
        copy(data.c),
        copy(data.t),
        copy(data.q),
        copy(data.d),
        data.C,
        data.N_customers,
        data.n_customers,
        data.N_depots,
        data.n_depots,
        data.N_charging,
        data.n_charging,
        data.N_depots_charging,
        data.n_depots_charging,
        data.N_nodes,
        data.n_nodes,
        A,
        data.T,
        data.B,
        data.inverse_refueling_rate,
        copy(data.α), 
        copy(data.β),
        tc_depot,
        cc_depot_charging,
    )

end

function create_pruned_edges(
    graph::EVRPGraph,
    k::Int,
)
    A = Set{Tuple{Int, Int}}()
    # 1. include self-loops in depots
    union!(A, [(i, i) for i in graph.N_depots])
    # 2. include all pairs between depots and charging stations
    union!(A, [(i, j) for (i, j) in permutations(graph.N_depots_charging, 2)])
    # 3. for each depot or charging station, include edges to and from nearest k customers
    for node in graph.N_depots_charging
        closest_customers = sortperm(graph.t[node, graph.N_customers])[1:k]
        union!(A, [(node, i) for i in closest_customers])
        union!(A, [(i, node) for i in closest_customers])
    end
    # 4. for each customer, include edges to and from nearest k customers (excluding themselves)
    for node in graph.N_customers
        closest_customers = setdiff(sortperm(graph.t[node, graph.N_customers])[1:k], node)
        union!(A, [(node, i) for i in closest_customers])
        union!(A, [(i, node) for i in closest_customers])
    end

    return A
end

function create_pruned_edges(
    graph,
    k,
    modified_costs::Matrix{Float64},
)
    A = Set{Tuple{Int, Int}}()
    # 1. include self-loops in depots
    union!(A, [(i, i) for i in graph.N_depots])
    # 2. include all nodes incoming and outgoing from depots
    union!(A, [(i, j) for i in graph.N_depots for j in graph.N_nodes])
    union!(A, [(i, j) for i in graph.N_nodes for j in graph.N_depots])
    # 3. at customers and charging stations, include edges to and from nearest k customers based on modified costs
    N_customers_charging = union(graph.N_customers, graph.N_charging)
    for node in N_customers_charging
        closest_nodes = sortperm(modified_costs[node, N_customers_charging])[1:k+1]
        union!(A, [(node, i) for i in closest_nodes])
        closest_nodes = sortperm(modified_costs[N_customers_charging, node])[1:k+1]
        union!(A, [(i, node) for i in closest_nodes])
    end

    return A
end

function prune_graph(
    graph::EVRPGraph,
    modified_costs::Matrix{Float64},
    ;
    k::Int = 5,
)
    A = create_pruned_edges(graph, k, modified_costs)

    G = SimpleDiGraph{Int}(graph.n_nodes)
    for (i, j) in A
        add_edge!(G, i, j)
    end
    t_ds = dijkstra_shortest_paths(G, collect(graph.N_depots), graph.t)
    q_ds = dijkstra_shortest_paths(G, collect(graph.N_depots_charging), graph.q)

    node_labels = merge(Dict(
        i => "Depot $ind" for (ind, i) in enumerate(graph.N_depots)
    ), Dict(
        i => "Customer $ind" for (ind, i) in enumerate(graph.N_customers)
    ), Dict(
        i => "Charging $ind" for (ind, i) in enumerate(graph.N_charging)
    ))

    return EVRPGraph(
        G,
        node_labels,
        copy(graph.c),
        copy(graph.t),
        copy(graph.q),
        copy(graph.d),
        graph.C,
        graph.N_customers,
        graph.n_customers,
        graph.N_depots,
        graph.n_depots,
        graph.N_charging,
        graph.n_charging,
        graph.N_depots_charging,
        graph.n_depots_charging,
        graph.N_nodes,
        graph.n_nodes,
        A,
        graph.T,
        graph.B,
        graph.inverse_refueling_rate,
        copy(graph.α), 
        copy(graph.β),
        t_ds.dists,
        q_ds.dists,
    )
end

function generate_time_windows(
    T_start::Int,
    T_end::Int,
    n_customers::Int,
    seed::Int,
    time_windows_min_width::Float64,
    time_windows_max_width::Float64,
)
    if !(
        0 <= time_windows_min_width ≤ time_windows_max_width <= 1
    )
        error("`time_windows_min_width` and `time_windows_max_width` out of bounds!")
    end
    Random.seed!(seed)
    if time_windows_min_width < time_windows_max_width
        time_window_dist = Uniform(time_windows_min_width * (T_end - T_start), time_windows_max_width * (T_end - T_start))
        time_window_widths = Int.(round.(rand(time_window_dist, n_customers)))
    else
        time_window_widths = Int.(round.(time_windows_min_width * (T_end - T_start)))
    end
    time_window_posdist = Uniform(0.0, 1.0)
    time_window_pos = rand(time_window_posdist, n_customers)
    α = T_start .+ Int.(round.(time_window_pos .* ((T_end - T_start) .- time_window_widths)))
    β = α .+ time_window_widths

    return (α, β)
end

function generate_times(
    T::Int,
    n_customers::Int,
    seed::Int,
    batch::Int,
    permissiveness::Float64 = 0.4,
)
    if n_customers % batch != 0
        error()
    end
    times_dist = Uniform(0.0, T)
    α = zeros(Int, n_customers)
    β = zeros(Int, n_customers)

    Random.seed!(seed)
    for batch_ind in 1:(n_customers ÷ batch)
        inds = collect((batch_ind-1)*batch+1:batch_ind*batch)
        while true
            times = Int.(round.(sort(rand(times_dist, 2 * batch))))
            start_times = times[1:end÷2]
            end_times = times[end÷2+1:end]
            if all(
                (end_times .- start_times) ./ T .> permissiveness
            )
                α[inds] = start_times
                β[inds] = end_times
                break
            end
        end
    end
    return α, β
end

function charge_to_specified_level(
    starting_charge::Int, 
    desired_end_charge::Int, 
    starting_time::Int, 
)
    """
    For a given starting charge, desired end charge, and starting time,
    compute the charged amount, end time, and end charge after charging.

    Note: assumes that 1 unit of charge takes 1 unit of time.
    """
    if desired_end_charge ≤ starting_charge
        return (0, starting_time, starting_charge)
    end
    delta = desired_end_charge - starting_charge
    end_time = starting_time + delta
    return (delta, end_time, desired_end_charge)
end

function compute_ngroute_neighborhoods(
    graph::EVRPGraph,
    ngroute_neighborhood_size::Int,
    ngroute_neighborhood_charging_size::Int,
    ;
)
    """
    neighborhoods[i,j] == 1 iff node i is in the ng-neighborhood of node j 
    """
    if !(1 ≤ ngroute_neighborhood_size ≤ graph.n_customers)
        error()
    end
    if !(0 ≤ ngroute_neighborhood_charging_size ≤ graph.n_customers)
        error()
    end
    neighborhoods = falses(graph.n_nodes, graph.n_nodes)
    for i in graph.N_customers
        neighborhoods[sortperm(graph.c[i, graph.N_customers])[1:ngroute_neighborhood_size], i] .= true
        # do not include any charging stations / depots in the neighborhoods of customers,
        # since there is no limit on repeat visits to charging stations / depots
    end
    for i in graph.N_depots
        # Include itself
        neighborhoods[i,i] = true
    end
    for i in graph.N_charging
        neighborhoods[i,i] = true
        neighborhoods[sortperm(graph.c[i, graph.N_customers])[1:ngroute_neighborhood_charging_size], i] .= true
    end
    return neighborhoods
end

function ngroute_update_fset!(
    l::Label,
    next_node::Int,
    neighborhoods::BitMatrix,
)
    l.ng_fset .&= neighborhoods[:, next_node]
    l.ng_fset[next_node] = true
    return
end

function ngroute_update_bset!(
    l::Label,
    next_node::Int,
)
    if l.ng_residue[next_node]
        l.ng_bset[next_node] = true
    end
    return
end

function ngroute_update_residue!(
    l::Label,
    next_node::Int,
    neighborhoods::BitMatrix,
)
    l.ng_residue .&= neighborhoods[:, next_node]
    return
end

function SR3_update_cost!(
    l::Label, # BPathLabel or SubpathLabel
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
)
    l.cost -= sum(λvals[l.cut_flabels .& λcust[:, next_node]])
    return
end

function SR3_update_cut_flabels!(
    l::Label,
    next_node::Int,
    λcust::BitMatrix,
)
    l.cut_flabels .⊻= λcust[:, next_node]
    return
end

function lmSR3_update_cost!(
    l::Label,
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
)
    l.cost -= sum(λvals[(
        l.cut_flabels
        .& λmemory[:, next_node]
        .& λcust[:, next_node]
    )])
    return
end

function lmSR3_update_cut_flabels!(
    l::Label,
    next_node::Int,
    λcust::BitMatrix,
    λmemory::BitMatrix,
)
    l.cut_flabels .&= λmemory[:, next_node]
    l.cut_flabels .⊻= λcust[:, next_node]
    return
end

function lmSR3_update_cut_blabels!(
    l::Label,
    next_node::Int,
    λcust::BitMatrix,
)
    l.cut_blabels .⊻= (l.cut_mlabels .& λcust[:, next_node])
    return
end

function lmSR3_update_cut_mlabels!(
    l::Label,
    next_node::Int,
    λmemory::BitMatrix,
)
    l.cut_mlabels .&= λmemory[:, next_node]
    return
end

function compute_arc_modified_costs(
    data::EVRPData,
    κ::Dict{Int, Float64},
    μ::Dict{Int, Float64},
    ν::Vector{Float64},
    ;
)
    modified_costs = data.travel_cost_coeff * Float64.(copy(data.c))
    for j in data.N_customers
        modified_costs[:,j] .-= ν[j]
    end
    for i in data.N_depots
        modified_costs[i,:] .-= κ[i]
        modified_costs[:,i] .-= μ[i]
    end

    return modified_costs
end


function prepare_lambda(
    λ::Dict{NTuple{3, Int}, Float64},
    n_nodes::Int,
)
    λvals = collect(values(λ))
    λcust = falses(length(λ), n_nodes)
    for (i, k) in enumerate(keys(λ))
        λcust[i, collect(k)] .= true
    end
    λmemory = falses(length(λ))
    return λvals, λcust
end

function prepare_lambda(
    λ::Dict{Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}}, Float64},
    n_nodes::Int,
)
    λvals = collect(values(λ))
    λcust = falses(length(λ), n_nodes)
    λmemory = falses(length(λ), n_nodes)
    for (i, k) in enumerate(keys(λ))
        (S, M) = k
        λcust[i, collect(S)] .= true
        λmemory[i, collect(M)] .= true
    end
    return λvals, λcust, λmemory
end

function bitvector_dominates(
    v1::BitVector,
    v2::BitVector,
) 
    """
    Returns true if v2[i] is true (i.e. = 1) whenever v1[i] is true,
    and false otherwise.
    """
    for i in eachindex(v1)
        if v1[i] && !v2[i]
            return false
        end
    end
    return true
end

# Note: the all() function uses short-circuiting (returns false if any value is false)
dominates(v1::T, v2::T) where {T <: Real} = v1 ≤ v2
dominates(v1::BitVector, v2::BitVector) = all(v1 .≤ v2)
dominates(v1::T, v2::T) where {T <: Vector{Int}} = all(v1 .≤ v2)
dominates(k1::T, k2::T) where {T <: Tuple} = all(dominates(v1, v2) for (v1, v2) in zip(k1, k2))
function dominates(
    k1::T, 
    k2::T, 
    λvals::Vector{Float64}, 
) where {T <: Tuple{Float64, BitVector, Vararg{Any}}}
    return (
        k1[1] - sum(λvals[k1[2] .& .~k2[2]]) ≤ k2[1]
        && all(dominates(v1, v2) for (v1, v2) in zip(k1[3:end], k2[3:end]))
    )
end

function dominates_lmSR3_01(
    k1::T, 
    k2::T, 
    λvals::Vector{Float64}, 
) where {T <: Tuple{Float64, BitVector, BitVector, BitVector, Vararg{Any}}}
    return (
        (
            k1[1] 
            - sum(λvals[.~k1[4] .& .~k2[4] .& k1[2]   .& .~k2[2]])
            - sum(λvals[.~k1[4] .& .~k2[4] .& k1[3]   .& .~k2[3]])
            - sum(λvals[k1[4]   .& .~k2[4] .& .~k2[2] .& .~k2[3]])
            - sum(λvals[k1[4]   .& .~k2[4] .& k1[2]   .& (k2[2] .⊻ k2[3])])
            - sum(λvals[.~k1[4] .& k2[4]   .& k1[2]   .& k1[3]])
            - sum(λvals[.~k1[4] .& k2[4]   .& .~k2[2] .& (k1[2] .⊻ k1[3])])
            - sum(λvals[k1[4]   .& k2[4]   .& k1[2]   .& .~k2[2]])
        ) ≤ k2[1]
        && all(dominates(v1, v2) for (v1, v2) in zip(k1[5:end], k2[5:end]))
    )
end



function add_label_to_collection!(
    collection::SortedDict{
        T,
        L,
        Base.Order.ForwardOrdering,
    },
    k1::T,
    v1::L,
    ;
) where {
    T <: Tuple, 
    L <: Label,
}
"""
# Fix: check each attribute lazily
# Fix: find the index of the incoming label in the collection, in terms of reduced cost
"""
    added = true
    for (k2, v2) in pairs(collection)
        if dominates(k2, k1)
            added = false
            break
        end
        if dominates(k1, k2)
            pop!(collection, k2)
        end
    end
    if added
        insert!(collection, k1, v1)
    end
    return added
end


function add_label_to_collection_cuts!(
    collection::SortedDict{
        T,
        L,
        Base.Order.ForwardOrdering,
    },
    k1::T,
    v1::L,
    λvals::Vector{Float64},
    ;
) where {
    T <: Tuple{Float64, BitVector, Vararg{Any}}, 
    L <: Label,
}
    added = true
    for (k2, v2) in pairs(collection)
        if dominates(k2, k1, λvals)
            added = false
            break
        end
        if dominates(k1, k2, λvals)
            pop!(collection, k2)
        end
    end
    if added
        insert!(collection, k1, v1)
    end
    return added
end


function add_label_to_collection_lmSR3_subpath!(
    collection::SortedDict{
        T,
        L,
        Base.Order.ForwardOrdering,
    },
    k1::T,
    v1::L,
    λvals::Vector{Float64},
    ;
) where {
    T <: Tuple{Float64, BitVector, BitVector, BitVector, Vararg{Any}}, 
    L <: Label,
}
    added = true
    for (k2, v2) in pairs(collection)
        if dominates_lmSR3_01(k2, k1, λvals)
            added = false
            break
        end
        if dominates_lmSR3_01(k1, k2, λvals)
            pop!(collection, k2)
        end
    end
    if added
        insert!(collection, k1, v1)
    end
    return added
end


function get_next_state!(
    unexplored_states::Vector{T},
) where {T <: Tuple}
    return popfirst!(unexplored_states)
end

function get_next_state!(
    unexplored_states::SortedSet{T},
) where {T <: Tuple}
    return pop!(unexplored_states)
end

function add_state_to_unexplored_states!(
    unexplored_states::Vector{T},
    state::T,
) where {T <: Tuple}
    push!(unexplored_states, state)
end

function add_state_to_unexplored_states!(
    unexplored_states::SortedSet{T},
    state::T,
) where {T <: Tuple}
    push!(unexplored_states, state)
end


function plot_instance(
    data::EVRPData,
    ;
    plot_edges::Bool = false,
    graph::Union{EVRPGraph, Nothing} = nothing,
    fontsize::Int = 11,
    markersize::Int = 4,
    alpha::Float64 = 0.7,
    legend::Union{Symbol, Bool} = :outerright,
    add_text_labels::Bool = true,
)

    xrange = data.coords[1,:] |> extrema |> x -> x[2] - x[1]
    yrange = data.coords[2,:] |> extrema |> x -> x[2] - x[1]
    offset = 0.02 * max(xrange, yrange)
    xlim = extrema(data.coords[1,:]) .+ [-3 * offset, 5 * offset]
    ylim = extrema(data.coords[2,:]) .+ [-3 * offset, 5 * offset]
    p = Plots.plot(
        # xlim = (0, 1), ylim = (0, 1),
        xlim = xlim,
        ylim = ylim,
        aspect_ratio = :equal, 
        fmt = :png, 
    )
    Plots.plot!(
        data.customer_coords[1,:], data.customer_coords[2,:],
        seriestype = :scatter, 
        label = "Customer",
        markersize = markersize,
        alpha = alpha,
        color = :green,
    )
    if add_text_labels
        annotate!.(
            data.customer_coords[1,:] .+ offset, data.customer_coords[2,:] .+ offset, 
            Plots.text.(
                collect(string(i) for i in 1:data.n_customers), 
                :green, :left, fontsize,
            )
        )
    end

    Plots.plot!(
        data.depot_coords[1,:], data.depot_coords[2,:],
        seriestype = :scatter, 
        label = "Depots",
        markershape = :utriangle,
        markersize = markersize,
        # alpha = alpha,
        alpha = 1.0,
        color = :black,
    )
    if add_text_labels
        Plots.annotate!.(
            data.depot_coords[1,:] .+ offset, data.depot_coords[2,:] .+ offset, 
            Plots.text.(
                collect("M" * string(i) for i in 1:data.n_depots), 
                :black, :left, fontsize,
            )
        )
    end

    Plots.plot!(
        data.charging_coords[1,:], data.charging_coords[2,:],
        seriestype = :scatter, 
        label = "Charging stations",
        markersize = markersize,
        markershape = :rect,
        alpha = alpha,
        color = :grey,
    )
    if add_text_labels
        Plots.annotate!.(
            data.charging_coords[1,:] .+ offset, data.charging_coords[2,:] .+ offset, 
            Plots.text.(
                collect("R" * string(i) for i in 1:data.n_charging), 
                :grey, :left, fontsize,
            )
        )
    end

    Plots.plot!(legend = legend)

    if plot_edges
        for e in edges(graph.G)
            Plots.plot!(
                data.coords[1,[e.src,e.dst]],
                data.coords[2,[e.src,e.dst]],
                label = false,
                color = :gray,
                alpha = 0.3,
            )
        end
    end
    return p
end

function compute_path_metrics(
    some_paths::Dict{T, Vector{Path}},
) where T
    total_subpath_length = 0.0
    num_subpaths = 0.0
    total_path_length = 0.0
    num_paths = 0.0
    total_ps_length = 0.0
    for path_l in values(some_paths)
        for p in path_l
            if (
                length(p.subpaths) == 1 
                && (
                    p.subpaths[1].artificial # artificial path
                    # || length(p.subpaths[1].arcs) == 1 # path from depot to depot
                )
            )
                continue
            end
            total_subpath_length += sum(sum(s.served) + 1 for s in p.subpaths) 
            num_subpaths += length(p.subpaths)
            total_path_length += sum(p.served) + length(p.subpaths)
            num_paths += 1
            total_ps_length += length(p.subpaths)
        end
    end

    return Dict(
        "mean_subpath_length" => total_subpath_length / num_subpaths,
        "mean_path_length" => total_path_length / num_paths,
        "mean_ps_length" => total_ps_length / num_paths,
    )
end
