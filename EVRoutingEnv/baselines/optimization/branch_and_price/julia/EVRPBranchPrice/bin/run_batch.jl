# julia --project=<pkg> bin/run_batch.jl LIST.txt [--key value ...]
#
# LIST.txt holds one "IN.json OUT.json" pair per line; all are solved in one
# process (one compilation, one solver environment) with the same parameters.

using EVRPBranchPrice

include(joinpath(@__DIR__, "run_bp.jl"))  # parse_parameters; its main() is guarded below

function batch(args)
    params = parse_parameters(args[2:end])
    env = Backend(Symbol(params.lp_solver))
    for line in eachline(args[1])
        isempty(strip(line)) && continue
        input, output = split(strip(line))
        inst = read_instance(input)
        res = solve(inst; params = params, env = env)
        write_result(output, inst, res, params)
        println(input, " status=", res.status, " objective=", res.upper, " bound=", res.lower,
                " seconds=", round(res.seconds; digits = 2))
        flush(stdout)
    end
end

batch(ARGS)
