# julia --project=<pkg> bin/run_bp.jl IN.json OUT.json [--key value ...]
#
# Any BPParameters field can be set, e.g. --time_limit 3600 --threads 8
# --ng_size 10 --verbose false.

using EVRPBranchPrice

function parse_parameters(args)
    kwargs = Dict{Symbol,Any}()
    types = Dict(zip(fieldnames(BPParameters), fieldtypes(BPParameters)))
    k = 1
    while k <= length(args)
        key = Symbol(replace(lstrip(args[k], '-'), '-' => '_'))
        haskey(types, key) || error("unknown parameter $(args[k])")
        T = types[key]
        kwargs[key] = T === Bool ? parse(Bool, args[k+1]) : parse(T, args[k+1])
        k += 2
    end
    BPParameters(; kwargs...)
end

function main(args)
    length(args) >= 2 || error("usage: run_bp.jl IN.json OUT.json [--key value ...]")
    params = parse_parameters(args[3:end])
    inst = read_instance(args[1])
    res = solve(inst; params = params)
    write_result(args[2], inst, res, params)
    println("status=", res.status, " objective=", res.upper, " bound=", res.lower)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
