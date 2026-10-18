abstract type PiecewiseLinearIncreasingFunction end 

struct PiecewiseLinearIncreasingConcaveFunction <: PiecewiseLinearIncreasingFunction
    N::Int
    x_intervals::Vector{<:Real}
    x_breakpoints::Vector{<:Real}
    y_intervals::Vector{<:Real}
    y_breakpoints::Vector{<:Real}
    slopes::Vector{<:Real}
    inv_slopes::Vector{<:Real}
    max_val::Real
    inv_max_val::Real
    function PiecewiseLinearIncreasingConcaveFunction(
        N::Int,
        x_intervals::Vector{<:Real}, 
        x_breakpoints::Vector{<:Real},
        y_intervals::Vector{<:Real},
        y_breakpoints::Vector{<:Real},
        slopes::Vector{<:Real},
        inv_slopes::Vector{<:Real},
        max_val::Real,
        inv_max_val::Real,
    )
        @assert (
            length(x_intervals) == length(x_breakpoints) - 1
            == length(y_intervals) == length(y_breakpoints) - 1
            == length(slopes) == length(inv_slopes)
            == N
        )
        @assert all(x_intervals .> 0)
        @assert all(y_intervals .> 0)
        @assert issorted(x_breakpoints)
        @assert issorted(y_breakpoints)
        @assert all(slopes .> 0)
        @assert issorted(slopes; rev=true)
        @assert all(inv_slopes .== 1 ./ slopes)
        @assert max_val == Inf
        @assert inv_max_val == sum(y_intervals) == y_breakpoints[end]
        return new(
            N,
            x_intervals,
            x_breakpoints,
            y_intervals,
            y_breakpoints,
            slopes,
            inv_slopes,
            max_val,
            inv_max_val,
        )
    end
end

function PiecewiseLinearIncreasingConcaveFunction(
    x_intervals::Vector{<:Real}, 
    slopes::Vector{<:Real},
)
    @assert length(x_intervals) == length(slopes)
    @assert issorted(slopes; rev=true)
    @assert all(slopes .> 0)
    @assert all(x_intervals .> 0)
    y_breakpoints = vcat(0, cumsum(x_intervals .* slopes))
    y_intervals = y_breakpoints[2:end] .- y_breakpoints[1:end-1]
    return PiecewiseLinearIncreasingConcaveFunction(
        length(x_intervals),
        x_intervals, 
        vcat(0, cumsum(x_intervals)),
        y_intervals,
        y_breakpoints,
        slopes, 
        1 ./ slopes,
        Inf,
        sum(y_intervals),
    )
end

struct PiecewiseLinearIncreasingConvexFunction <: PiecewiseLinearIncreasingFunction
    N::Int
    x_intervals::Vector{<:Real}
    x_breakpoints::Vector{<:Real}
    y_intervals::Vector{<:Real}
    y_breakpoints::Vector{<:Real}
    slopes::Vector{<:Real}
    inv_slopes::Vector{<:Real}
    max_val::Real
    inv_max_val::Real
    function PiecewiseLinearIncreasingConvexFunction(
        N::Int,
        x_intervals::Vector{<:Real}, 
        x_breakpoints::Vector{<:Real},
        y_intervals::Vector{<:Real},
        y_breakpoints::Vector{<:Real},
        slopes::Vector{<:Real},
        inv_slopes::Vector{<:Real},
        max_val::Real,
        inv_max_val::Real,
    )
        @assert (
            length(x_intervals) == length(x_breakpoints) - 1
            == length(y_intervals) == length(y_breakpoints) - 1
            == length(slopes) == length(inv_slopes)
            == N
        )
        @assert all(x_intervals .> 0)
        @assert all(y_intervals .> 0)
        @assert issorted(x_breakpoints)
        @assert issorted(y_breakpoints)
        @assert all(slopes .> 0)
        @assert issorted(slopes)
        @assert all(inv_slopes .== 1 ./ slopes)
        @assert max_val == sum(x_intervals) == x_breakpoints[end]
        @assert inv_max_val == Inf
        return new(
            N,
            x_intervals,
            x_breakpoints,
            y_intervals,
            y_breakpoints,
            slopes,
            inv_slopes,
            max_val,
            inv_max_val,
        )
    end
end

function PiecewiseLinearIncreasingConvexFunction(
    x_intervals::Vector{<:Real}, 
    slopes::Vector{<:Real},
)
    @assert length(x_intervals) == length(slopes)
    @assert issorted(slopes)
    @assert all(slopes .> 0)
    @assert all(x_intervals .> 0)
    y_breakpoints = vcat(0, cumsum(x_intervals .* slopes))
    y_intervals = y_breakpoints[2:end] .- y_breakpoints[1:end-1]
    return new(
        length(x_intervals),
        x_intervals, 
        vcat(0, cumsum(x_intervals)),
        y_intervals,
        y_breakpoints,
        slopes, 
        1 ./ slopes,
        sum(x_intervals),
        Inf,
    )
end

function (g::PiecewiseLinearIncreasingFunction)(x::T) where {T <: Real}
    for i in 1:g.N
        if x > g.x_breakpoints[i+1]
            continue
        end
        return g.y_breakpoints[i+1] - g.slopes[i] * (g.x_breakpoints[i+1] - x)
    end
    if x > g.max_val
        error("Input $x exceeds maximum value $(g.max_val) of piecewise linear function.")
    end
    return g.y_breakpoints[end]
end

function invert(g::PiecewiseLinearIncreasingConcaveFunction)
    return PiecewiseLinearIncreasingConvexFunction(
        g.N,
        g.y_intervals,
        g.y_breakpoints, 
        g.x_intervals, 
        g.x_breakpoints,
        g.inv_slopes,
        g.slopes,
        sum(g.y_intervals),
        Inf,
    )
end

function invert(g::PiecewiseLinearIncreasingConvexFunction)
    return PiecewiseLinearIncreasingConcaveFunction(
        g.N,
        g.y_intervals,
        g.y_breakpoints, 
        g.x_intervals, 
        g.x_breakpoints,
        g.inv_slopes,
        g.slopes,
        Inf,
        sum(g.x_intervals),
    )
end

function compute_charging_duration(
    g::PiecewiseLinearIncreasingConcaveFunction,
    charge_lower::Real,
    charge_upper::Real,
)
    if charge_upper <= charge_lower
        return zero(charge_lower)
    end
    if charge_upper > g.inv_max_val
        error("Input $charge_upper exceeds maximum value $(g.inv_max_val) of piecewise linear function.")
    end
    val = zero(charge_lower)
    iL = 0
    for i in 1:g.N
        if charge_lower > g.y_breakpoints[i+1]
            continue
        end
        val -= (charge_lower - g.y_breakpoints[i]) * g.inv_slopes[i]
        iL = i
        break
    end
    for i in iL:g.N
        if charge_upper > g.y_breakpoints[i+1]
            continue
        end
        val -= (g.y_breakpoints[i+1] - charge_upper) * g.inv_slopes[i]
        return val + g.x_breakpoints[i+1] - g.x_breakpoints[iL]
    end
end

# function compute_charging_duration(
#     g::PiecewiseLinearIncreasingConcaveFunction,
#     ginv::PiecewiseLinearIncreasingConvexFunction,
#     charge_lower::Real,
#     charge_upper::Real,
# )
#     @assert charge_lower ≤ ginv.max_val "Input $charge_lower exceeds maximum value $(ginv.max_val) of piecewise linear function."
#     @assert charge_upper ≤ g.inv_max_val "Input $charge_upper exceeds maximum value $(g.inv_max_val) of piecewise linear function."
#     return ginv(charge_upper) - ginv(charge_lower)
# end

function compute_end_charge(
    g::PiecewiseLinearIncreasingConcaveFunction,
    charging_duration::Real,
    initial_charge::Real,
)
    charging_duration_L = zero(charging_duration)
    iL = 0
    iR = 0
    for i in 1:g.N
        if g.y_breakpoints[i+1] < initial_charge
            continue
        end
        # Here, g.y_breakpoints[i] < initial_charge <= g.y_breakpoints[i+1]
        charging_duration_L = (g.y_breakpoints[i+1] - initial_charge) * g.inv_slopes[i] 
        if charging_duration_L ≥ charging_duration
            return initial_charge + charging_duration * g.slopes[i]
        end
        iL = i
        break
    end
    charging_duration -= charging_duration_L
    x_breakpoint_R = g.x_breakpoints[iL+1] + charging_duration
    # println(x_breakpoint_L + charging_duration)
    for i in iL+1:g.N
        if x_breakpoint_R ≥ g.x_breakpoints[i+1]
            charging_duration -= g.x_intervals[i]
            continue
        end
        # Here, g.x_breakpoints[i] <= (x_breakpoint_L + charging_duration) <= g.x_breakpoints[i+1]
        return g.y_breakpoints[i] + (charging_duration * g.slopes[i])
    end
    return g.y_breakpoints[end]
end
