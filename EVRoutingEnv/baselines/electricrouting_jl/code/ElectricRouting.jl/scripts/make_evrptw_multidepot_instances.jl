using Pkg
Pkg.activate("$(@__DIR__)/..")

include("$(@__DIR__)/../src/utils.jl")
include("$(@__DIR__)/../src/read_evrptw_data.jl")

using Glob
using Plots

n_depots = 4
depot_pattern = "grid"
customer_pattern = "random_box"
charging_pattern = "grid_clipped"
customer_spread = 0.1
load_tolerance = 2.0

inverse_refueling_rate = 0.2
(ymin, ymax) = (0.0, 4.0)
xmin = 0.0
B = 20000
n_vehicles = 10

xmax_k_range = [
    (2.0, 6.0),
    (3.0, 7.0),
    (4.0, 8.0),
    (5.0, 9.0),
]
# density = 5.0
density_range = [
    2.0,
    3.0,
    4.0,
    5.0, 
]
# time_windows_min_width, time_windows_max_width = 0.5, 0.5
time_windows_min_max_type_width_range = [
    (0.5, 0.5, "small"),
    (1.0, 1.0, "big"),
]
seed_range = 1:5

for (
    (xmax, k),
    density,
    (
        time_windows_min_width, 
        time_windows_max_width,
        time_windows_type,
    ),
    seed,
) in Iterators.product(
    xmax_k_range,
    density_range,
    time_windows_min_max_type_width_range,
    seed_range,
)
    T = Int(B * k * (1 + inverse_refueling_rate))
    n_customers = Int(density * (xmax - xmin) * (ymax - ymin))
    n_charging = Int((xmax - xmin + 1)*(ymax - ymin + 1) - 4)


    filename = @sprintf(
        "%03d_%02d_%s_%02d", 
        n_customers,
        Int(xmax * ymax),
        time_windows_type,
        seed,
    )

    if (
        n_customers == 60
        && time_windows_type == "small"
        && seed == 4
        && xmax == 5.0
    )
        seed += 5
    end

    data = generate_instance(
        n_depots = n_depots,
        n_customers = n_customers,
        n_charging = n_charging,
        n_vehicles = n_vehicles,
        depot_pattern = depot_pattern,
        customer_pattern = customer_pattern,
        charging_pattern = charging_pattern,
        customer_spread = customer_spread,
        xmin = xmin,
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        T = T,
        seed = seed,
        B = B,
        inverse_refueling_rate = inverse_refueling_rate,
        travel_cost_coeff = 1,
        load_tolerance = load_tolerance,
        load_precision = 10,
        vehicle_locations = "gradiated",
        time_windows_distribution = "gradiated",
        time_windows_min_width = time_windows_min_width,
        time_windows_max_width = time_windows_max_width,
    )

    write_instance_evrptw_format(
        data,
        "$(@__DIR__)/../data/evrptw_multidepot/Instances/$filename.txt",
        ;
        multidepot = true,
    )
end
