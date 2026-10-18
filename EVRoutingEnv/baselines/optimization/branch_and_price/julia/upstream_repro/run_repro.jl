# Ladder step 1: run the unmodified ElectricRouting.jl snapshot on its bundled
# multi-depot instances and compare with the shipped Solutions/*.txt costs.
#
#   julia --project=. run_repro.jl [time_limit_s] [instance ...]
#
# The archived snapshot is read in place and never modified.

const UPSTREAM = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
    "electricrouting_jl", "code", "ElectricRouting.jl"))
const DATA_DIR = joinpath(UPSTREAM, "data", "evrptw_multidepot")

include(joinpath(UPSTREAM, "src", "utils.jl"))
include(joinpath(UPSTREAM, "src", "nonlinear_charging.jl"))
include(joinpath(UPSTREAM, "src", "subpath_stitching.jl"))
include(joinpath(UPSTREAM, "src", "desaulniers_benchmark.jl"))
include(joinpath(UPSTREAM, "src", "path_formulation.jl"))
include(joinpath(UPSTREAM, "src", "read_evrptw_data.jl"))

function shipped_cost(name)
    line = first(filter(l -> startswith(l, "Total cost"),
        readlines(joinpath(DATA_DIR, "Solutions", name * ".txt"))))
    parse(Int, split(line)[end])
end

function run_one(name, env; time_limit = 600.0)
    data = read_evrptw_instance(name, 10, 1; multidepot = true, data_dir = DATA_DIR)
    graph = generate_graph_from_data(data)
    t0 = time()
    lp, ip, _, _, _, _, _, _, _, _ = path_formulation_column_generation_with_adaptive_ngroute_SR3_cuts(
        data, graph; Env = env, method = "ours",
        use_time_windows = true, use_load = true,
        verbose = false, time_limit = time_limit,
    )
    (lp_obj = last(lp)["objective"], ip_obj = last(ip)["objective"], seconds = time() - t0)
end

function main(args)
    time_limit = isempty(args) ? 600.0 : parse(Float64, args[1])
    names = length(args) > 1 ? args[2:end] : ["016_08_small_01", "016_08_big_01", "032_12_small_01"]
    env = Gurobi.Env()
    run_one(names[1], env; time_limit = 30.0)  # compile before timing
    println("instance,lp,ip,shipped,ip_minus_shipped,seconds")
    for name in names
        r = run_one(name, env; time_limit = time_limit)
        ref = shipped_cost(name)
        println(join((name, r.lp_obj, r.ip_obj, ref, r.ip_obj - ref, round(r.seconds; digits = 2)), ","))
        flush(stdout)
    end
end

main(ARGS)
