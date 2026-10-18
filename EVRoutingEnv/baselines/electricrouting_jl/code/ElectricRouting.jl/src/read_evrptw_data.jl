using CSV
using DataFrames
using Accessors

function transform_floats(
    x,
    pr::Float64 = 100.0,
    eps::Float64 = 1e-9,
)
    if (x * pr - floor.(x) * pr) < eps
        return Int.(floor.(x * pr))
    else
        return Int.(ceil.(x * pr))
    end
end

function read_evrptw_instance(
    instance_name::String,
    n_vehicles::Int,
    travel_cost_coeff::Int,
    ;
    n_charging::Int = 0, # If 0, read from file
    n_customers::Int = 0, # If 0, read from file
    scale_time_horizon::Float64 = 1.0,
    scale_charge_capacity::Float64 = 1.0,
    multidepot::Bool = false,
    separate_source_sink::Bool = false,
    data_dir::String = "data/evrptw",
)
    # Read instance
    instance_fp = joinpath(data_dir, "Instances/", instance_name * ".txt")
    instance_lines = readlines(instance_fp)
    sep = findfirst(x -> occursin(r"^\s*$", x), instance_lines)
    table_str = join(instance_lines[1:sep-1], "\n")
    df = CSV.read(
        IOBuffer(table_str), 
        DataFrame, 
        header=["StringID", "Type", "x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"],
        skipto=2,
        delim=' ', 
        ignorerepeated=true,
    )
    
    (
        B_, # Unscaled battery capacity
        C,  # Load capacity
        _,
        inverse_refueling_rate,  # Inverse battery refueling rate
        _
    ) = [parse(Float64, match(r"/([\d.]+)/", x)[1]) for x in instance_lines[sep+1:sep+5]]

    # Scale load capacity
    C = Int(round(C))

    # Scaled battery capacity
    B = transform_floats(B_ * scale_charge_capacity * inverse_refueling_rate)

    # DataFrames for depots, customers, charging stations
    Random.seed!(0)
    depots_df = filter(r -> r.Type == "d", df)
    customers_df = filter(r -> r.Type == "c", df)
    charging_df = filter(r -> r.Type == "f", df)

    if n_customers > 0
        customers_df = customers_df[1:n_customers, :]
    end
    if n_charging > 0
        charging_df = charging_df[1:n_charging, :]
    end

    # Index sets
    n_depots = nrow(depots_df)
    n_customers = nrow(customers_df)
    n_charging = nrow(charging_df)

    n_depots_ = (separate_source_sink ? 2 * n_depots : n_depots)

    n_nodes = n_depots_ + n_customers + n_charging

    N_customers = 1:n_customers
    N_depots = n_customers+1:n_customers+n_depots_
    N_vehicles = 1:n_vehicles
    N_charging = n_customers+n_depots_+1:n_customers+n_depots_+n_charging
    N_depots_charging = n_customers+1:n_customers+n_depots_+n_charging
    N_nodes = 1:n_customers+n_depots_+n_charging

    if separate_source_sink
        node_labels = merge(Dict(
            N_depots .=> repeat(depots_df[!, :StringID], 2)
        ), Dict(
            N_customers .=> customers_df[!, :StringID]
        ), Dict(
            N_charging .=> charging_df[!, :StringID]
        ))
    else 
        node_labels = merge(Dict(
            N_depots .=> depots_df[!, :StringID]
        ), Dict(
            N_customers .=> customers_df[!, :StringID]
        ), Dict(
            N_charging .=> charging_df[!, :StringID]
        ))
    end

    # Locations 
    depot_coords = depots_df[:, [:x, :y]] |> Matrix{Float64} |> transpose
    if separate_source_sink
        depot_coords = hcat(depot_coords, depot_coords)
    end
    customer_coords = customers_df[:, [:x, :y]] |> Matrix{Float64} |> transpose
    charging_coords = charging_df[:, [:x, :y]] |> Matrix{Float64} |> transpose

    coords = hcat(customer_coords, depot_coords, charging_coords)
    (xmin, xmax) = extrema(coords[1, :])
    (ymin, ymax) = extrema(coords[2, :])

    distances = Distances.pairwise(Euclidean(), coords; dims=2)
    
    # Starting locations of vehicles
    if multidepot
        # @assert sort(keys(multidepot_v_start)) == N_depots
        # @assert sort(keys(multidepot_v_end)) == N_depots
        # @assert sum(values(multidepot_v_start)) == n_vehicles
        # @assert sum(values(multidepot_v_end)) <= n_vehicles
        # v_start = multidepot_v_start
        # v_end = multidepot_v_end
        sep_start = findlast(x -> occursin(r"^Vehicle start locations", x), instance_lines)
        V = Dict{Int, Vector{Int}}()
        v_start = Dict{Int, Int}()
        nv = 0
        for i in 1:n_depots
            m = match(r"^D(\d+):\s+(\d+)", instance_lines[sep_start + i])
            depot = parse(Int, m[1]) + n_customers
            val = parse(Int, m[2])
            v_start[depot] = val
            V[depot] = collect(nv+1:nv+val)
            nv += val
        end
        sep_end = findlast(x -> occursin(r"^Vehicle end locations", x), instance_lines)
        v_end = Dict{Int, Int}()
        for i in 1:n_depots
            m = match(r"^D(\d+):\s+(\d+)", instance_lines[sep_end + i])
            depot = parse(Int, m[1]) + n_customers
            val = parse(Int, m[2])
            v_end[depot] = val
        end
    else
        @assert n_depots == 1
        start_depots = repeat(N_depots, n_vehicles)
        V = Dict(N_depots[1] => collect(1:n_vehicles))
        v_start = v_end = Dict(N_depots[1] => n_vehicles)
    end

    # Travel cost
    c = transform_floats.(distances)
    # (Scaled) travel time

    travel_times = copy(distances)
    for i in 1:n_customers
        travel_times[i, :] .+= customers_df[i, "ServiceTime"]
    end
    t = transform_floats.(travel_times)

    # Battery consumption
    q = transform_floats.(distances * inverse_refueling_rate)

    # Demand
    if separate_source_sink
        d = vcat(
            customers_df[!, :demand],
            depots_df[!, :demand],
            depots_df[!, :demand],
            charging_df[!, :demand],
        )
        # Time windows
        α = vcat(
            customers_df[!, :ReadyTime],
            depots_df[!, :ReadyTime],
            depots_df[!, :ReadyTime],
            charging_df[!, :ReadyTime],
        )
        α = transform_floats.(α * scale_time_horizon)
        β = vcat(
            customers_df[!, :DueDate],
            depots_df[!, :DueDate],
            depots_df[!, :DueDate],
            charging_df[!, :DueDate],
        )
        β = transform_floats.(β * scale_time_horizon)
        T = Int(round(maximum(β)))
    else
        d = vcat(
            customers_df[!, :demand],
            depots_df[!, :demand],
            charging_df[!, :demand],
        )
        # Time windows
        α = vcat(
            customers_df[!, :ReadyTime],
            depots_df[!, :ReadyTime],
            charging_df[!, :ReadyTime],
        )
        α = transform_floats.(α * scale_time_horizon)
        β = vcat(
            customers_df[!, :DueDate],
            depots_df[!, :DueDate],
            charging_df[!, :DueDate],
        )
        β = transform_floats.(β * scale_time_horizon)
        T = Int(round(maximum(β)))
    end

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
        "evrptw", # depot_pattern
        "evrptw", # customer_pattern
        "evrptw", # charging_pattern
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
        α,
        β,
        inverse_refueling_rate,
        B,
        travel_cost_coeff,
    )

    return data
end

function stretch_time_windows!(
    data::EVRPData,
    stretch_factor::Float64,
)
    if !(0.0 <= stretch_factor <= 1.0)
        error("stretch_factor must be in [0, 1]")
    end
    data.α .= Int.(round.(
        # stretch_factor .* 0 .+ 
        (1 - stretch_factor) .* data.α 
    ))
    data.β .= Int.(round.(
        stretch_factor .* data.T
        .+ (1 - stretch_factor) .* data.β
    ))
    return
end

function remove_time_windows!(
    data::EVRPData,
    remove_factor::Float64,
    ;
    seed::Union{Int, Nothing} = nothing
)
    if !(0.0 <= remove_factor <= 1.0)
        error("remove_factor must be in [0, 1]")
    end
    n_TWs_remove = Int(round(remove_factor * data.n_customers * 2))
    
    if !isnothing(seed)
        Random.seed!(seed)
    end
    rand_inds = Random.shuffle(1:2*data.n_customers)[1:n_TWs_remove]

    for i in rand_inds
        if i <= data.n_customers
            j = i
            data.α[j] = 0
        else
            j = i - data.n_customers
            data.β[j] = data.T
        end
    end
    return
end

function duplicate_time_horizon!(
    data::EVRPData,
    batch_size::Int,
    ;
    seed::Int = 0,
)
    @assert data.n_customers % batch_size == 0

    Random.seed!(seed)

    shuffled_customers = shuffle(1:data.n_customers)
    n_batches = data.n_customers ÷ batch_size
    batches = [
        shuffled_customers[(k-1)*batch_size+1:k*batch_size] 
        for k in 1:n_batches
    ]
    for (k, batch) in enumerate(batches)
        data.α[batch] .+= (k-1) * data.T
        data.β[batch] .+= (k-1) * data.T
    end
    @reset data.T = data.T * n_batches
    data.β[data.N_depots_charging] .= data.T
    return data
end
