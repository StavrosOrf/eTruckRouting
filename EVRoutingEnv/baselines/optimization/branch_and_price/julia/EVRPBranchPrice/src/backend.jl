# LP/MIP backend of the master problem. Gurobi is the default; HiGHS needs no
# licence, which lets tests and extra solves run while every Gurobi WLS
# session is taken. Pricing, bounds and certificates do not depend on it.

struct Backend
    kind::Symbol                     # :gurobi or :highs
    env::Union{Nothing,Gurobi.Env}
end

Backend(kind::Symbol) = kind == :gurobi ? Backend(:gurobi, Gurobi.Env()) :
                        kind == :highs ? Backend(:highs, nothing) :
                        error("unknown lp_solver $(kind); use gurobi or highs")

function new_model(b::Backend; threads::Int, method::Int = -1)
    if b.kind == :gurobi
        model = direct_model(Gurobi.Optimizer(b.env))
        set_silent(model)
        set_attribute(model, "Threads", threads)
        set_attribute(model, "Method", method)
    else
        model = direct_model(HiGHS.Optimizer())
        set_silent(model)
    end
    model
end

"""Stop a MIP once it cannot beat `value` (Gurobi Cutoff, HiGHS objective bound)."""
function set_cutoff!(model, b::Backend, value::Float64)
    b.kind == :gurobi ? set_attribute(model, "Cutoff", value) : set_attribute(model, "objective_bound", value)
end
