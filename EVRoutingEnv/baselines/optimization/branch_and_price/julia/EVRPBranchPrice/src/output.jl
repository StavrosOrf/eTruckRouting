# `evrp-bp-result/v1` certificate: bounds, solver evidence and, per route, the
# schedule under nominal rules plus the environment witness steps.

function route_record(inst::Instance, r::Route)
    ok, why, stops = simulate(inst, r)
    Dict(
        "feasible" => ok,
        "reason" => why,
        "travel_h" => r.travel,
        "load" => sum(inst.demand[c] for c in r.customers; init = 0.0),
        "customers" => [inst.task_id[c] for c in r.customers],
        "witness" => [[id, v, c] for (id, v, c) in witness_steps(inst, r)],
        "stops" => [Dict(
            "node" => inst.env_id[x.node],
            "kind" => is_customer(inst, x.node) ? "customer" : x.node == inst.depot ? "depot" : "charger",
            "arrival_h" => x.arrival,
            "arrival_kwh" => x.arrival_battery,
            "level" => x.level,
            "charge_h" => x.charge_hours,
            "departure_h" => x.departure,
            "departure_kwh" => x.departure_battery,
            "load_after" => x.load_after,
        ) for x in stops],
    )
end

function result_dict(inst::Instance, res::BPResult, params::BPParameters)
    gap = isfinite(res.upper) && isfinite(res.lower) && res.upper > 0 ?
        max(0.0, (res.upper - res.lower) / res.upper) : nothing
    num(x) = isfinite(x) ? x : nothing
    Dict(
        "schema" => "evrp-bp-result/v1",
        "status" => res.status,
        "proven_optimal" => res.status == "OPTIMAL",
        "objective" => String(inst.objective),
        "objective_hours" => num(res.upper),
        "best_bound_hours" => num(res.lower),
        "gap" => gap,
        "phases" => res.phases,
        "relaxed_optimum_hours" => num(res.relaxed_optimum),
        "root_lp_hours" => num(res.root_lp),
        "root_bound_hours" => num(res.root_bound),
        "nodes_explored" => res.nodes,
        "nodes_open" => res.open,
        "columns" => res.columns,
        "pricing_rounds" => res.rounds,
        "exact_pricing_rounds" => res.exact_rounds,
        "wall_seconds" => res.seconds,
        "pricing_seconds" => res.pricing_seconds,
        "master_seconds" => res.master_seconds,
        "routes" => [route_record(inst, r) for r in res.routes],
        "unused_trucks" => inst.fleet - length(res.routes),
        "assumptions" => Dict(
            "nominal" => "mean leg times/energies, base service times, deterministic charging curve",
            "charger_ports" => "relaxed (uncapacitated); verify and repair downstream",
            "charge_curve_interpolation_step" => inst.soc_step,
            "tolerance" => EPS,
        ),
        "parameters" => Dict(String(k) => getfield(params, k) for k in fieldnames(BPParameters)),
        "instance_meta" => inst.meta,
        "versions" => Dict("julia" => string(VERSION), "gurobi" => string(Gurobi.GRB_VERSION_MAJOR, ".",
            Gurobi.GRB_VERSION_MINOR, ".", Gurobi.GRB_VERSION_TECHNICAL)),
        "log" => res.log,
    )
end

write_result(path::AbstractString, inst::Instance, res::BPResult, params::BPParameters) =
    open(io -> JSON3.pretty(io, result_dict(inst, res, params)), path, "w")
