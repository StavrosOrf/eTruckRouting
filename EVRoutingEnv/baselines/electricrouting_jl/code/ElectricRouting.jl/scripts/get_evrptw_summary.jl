using Pkg
Pkg.activate("$(@__DIR__)/..")

using DataFrames, CSV, Glob


instance_type_range = [
    # Tight time windows
    "r1", 
    "c1", 
    "rc1", 
    # Loose time windows
    "r2", 
    "c2", 
    "rc2",
]

n_customers_range = [25, 50, 100]
data_dir = "$(@__DIR__)/../data/"

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


records = []


for n_customers in n_customers_range, 
    instance_type in instance_type_range
    instance_name_list = [
        basename(x)[1:end-4]
        for x in Glob.glob(
            "$(instance_type)*_*_$(n_customers).txt",
            joinpath(data_dir, "evrptw/Instances/"),
        )
    ]
    println(instance_name_list)
    for instance_name in instance_name_list

        instance_fp = joinpath(data_dir, "evrptw/Instances/", instance_name * ".txt")
        instance_lines = readlines(instance_fp)
        sep = findfirst(x -> occursin(r"^\s*$", x), instance_lines)
        table_str = join(instance_lines[1:sep-1], "\n")
        df = CSV.read(IOBuffer(table_str), DataFrame, delim=' ', ignorerepeated=true)
        (
            B_, # Unscaled battery capacity
            C,  # Load capacity
            _,
            inverse_refueling_rate,  # Inverse battery refueling rate
            _
        ) = [parse(Float64, match(r"/([\d.]+)/", x)[1]) for x in instance_lines[sep+1:end]]
        
        C = Int(round(C))
        B = transform_floats(B_ * inverse_refueling_rate)


        β = df[!, :DueDateServiceTime]
        β = transform_floats.(β)
        T = Int(round(maximum(β)))

        depots_df = filter(r -> r.Type == "d", df)
        customers_df = filter(r -> r.Type == "c", df)
        charging_df = filter(r -> r.Type == "f", df)

        n_customers = nrow(customers_df)
        n_charging = nrow(charging_df)

        depot_x = depots_df[1, :x]
        depot_y = depots_df[1, :y]
        customer_x_min = minimum(customers_df[!, :x])
        customer_x_max = maximum(customers_df[!, :x])
        customer_y_min = minimum(customers_df[!, :y])
        customer_y_max = maximum(customers_df[!, :y])
        charging_x_min = minimum(charging_df[!, :x])
        charging_x_max = maximum(charging_df[!, :x])
        charging_y_min = minimum(charging_df[!, :y])
        charging_y_max = maximum(charging_df[!, :y])

        println("Instance $instance_name: ($depot_x, $depot_y, $customer_x_min, $customer_x_max, $customer_y_min, $customer_y_max)")
        push!(
            records, 
            (
                instance_name = instance_name,
                n_customers = n_customers,
                n_charging = n_charging,
                depot_x = depot_x,
                depot_y = depot_y,
                customer_x_min = customer_x_min,
                customer_x_max = customer_x_max,
                customer_y_min = customer_y_min,
                customer_y_max = customer_y_max,
                charging_x_min = charging_x_min,
                charging_x_max = charging_x_max,
                charging_y_min = charging_y_min,
                charging_y_max = charging_y_max,
                B = B,
                T = T,
            )
        )
    end
end

records_df = DataFrame(records)
(
    records_df
    |> x -> sort(x, [:n_customers, :instance_name])
    |> x -> filter(
        r -> r.n_customers == 100,
        x
    )
    |> x -> println(x)
)
