mutable struct BPathLabel <: Label
    cost::Float64
    nodes::Vector{Int}
    excesses::Vector{Int}
    slacks::Vector{Int}
    recharged::Int
    time_mincharge::Int
    time_maxcharge::Int
    time_maxrecharge::Int
    load::Int
    served::BitVector
    ng_fset::BitVector
    cut_flabels::BitVector
end

Base.copy(p::BPathLabel) = BPathLabel(
    p.cost,
    copy(p.nodes),
    copy(p.excesses),
    copy(p.slacks),
    p.recharged,
    p.time_mincharge,
    p.time_maxcharge,
    p.time_maxrecharge,
    p.load,
    copy(p.served),
    copy(p.ng_fset),
    copy(p.cut_flabels),
)

# Note: get_vkey() is a key, not for comparison
get_vkey(p::BPathLabel) = (
    # Necessary for sorting purposes
    p.cost,
    # Helpful for sorting purposes
    p.time_mincharge,
    p.time_mincharge + p.time_maxrecharge,
    p.time_mincharge + p.time_maxrecharge - p.time_maxcharge,
    p.load,
    # Necessary to uniquely determine a path
    p.nodes,
)

BPATH_VKEY_TYPE = Tuple{
    Float64,
    Int, 
    Int, 
    Int,
    Int,
    Vector{Int},
}

function create_new_bpath_label(
    starting_node::Int,
    data::EVRPData,
    ;
    n_cuts::Int = 0,
)

    ng_fset = falses(data.n_nodes)
    ng_fset[starting_node] = true

    return BPathLabel(
        0.0,
        [starting_node],
        Int[], Int[],
        0,
        # 0, 0, data.B, data.B, -data.B,
        0, 0, 0, 
        0,
        falses(data.n_customers),
        ng_fset,
        falses(n_cuts),
    )
end


function update_path_ngroute!(
    new_path::BPathLabel,
    next_node::Int,
    ng_neighborhoods::BitMatrix
)
    # update forward ng-set
    ngroute_update_fset!(new_path, next_node, ng_neighborhoods)
    return 
end

function update_path_cut_labels_SR3!(
    new_path::BPathLabel,
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
)
    ## IMPORTANT: update cost before flabels
    ## since flabels affect cost
    SR3_update_cost!(new_path, next_node, λvals, λcust)
    SR3_update_cut_flabels!(new_path, next_node, λcust)
    return
end

function update_path_cut_labels_LmSR3!(
    new_path::BPathLabel,
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
)
    ## IMPORTANT: update cost before flabels
    ## since flabels affect cost
    lmSR3_update_cost!(new_path, next_node, λvals, λcust, λmemory)
    lmSR3_update_cut_flabels!(new_path, next_node, λcust, λmemory)
    return
end


function compute_new_bpath(
    current_path::BPathLabel,
    data::EVRPData,
    graph::EVRPGraph,
    modified_costs::Matrix{Float64},
    current_node::Int,
    next_node::Int,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    ng_neighborhoods::BitMatrix,
    cuts::String,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
    ;
)
    @debug "Computing new path from nodes: $(current_path.nodes) to next_node: $next_node"

    # customer service
    if use_ngroute
        if current_path.ng_fset[next_node]
            @debug "  Infeasible extension from node $current_node to node $next_node (ng-route): $(current_path.ng_fset)"
            return (false, current_path)
        end
    else
        # elementary
        if next_node in data.N_customers && current_path.served[next_node]
            @debug "  Infeasible extension from node $current_node to node $next_node (elementary): $(current_path.served)"
            return (false, current_path)
        end
    end

    

    # load feasibility
    if use_load
        @debug "  Current load: $(current_path.load), next customer load: $(data.d[next_node]), Capacity: $(data.C)"
        if current_path.load + data.d[next_node] > data.C
            @debug "  Infeasible extension (load): $(current_path.load), $(data.d[next_node]), $(data.C)"
            return (false, current_path)
        end
    end


    new_path = copy(current_path)
    push!(new_path.nodes, next_node)
    if next_node in data.N_charging
        new_path.recharged += 1
    end

    @debug "  Time windows: [$(data.α[next_node]), $(data.β[next_node])]"
    @debug "  current_path.time_mincharge: $(current_path.time_mincharge)"
    @debug "  current_path.time_maxcharge: $(current_path.time_maxcharge)"
    @debug "  data.t[current_node, next_node]: $(data.t[current_node,next_node])"

    if current_node in data.N_charging
        slack = max(
            0,
            min(
                data.α[next_node] - current_path.time_mincharge - data.t[current_node,next_node],
                current_path.time_maxrecharge,
            )
        )
    else
        slack = max(
            0,
            min(
                data.α[next_node] - current_path.time_mincharge - data.t[current_node,next_node],
                current_path.time_maxcharge - current_path.time_mincharge,
            )
        )
    end
    @debug "  slack: $slack"
    push!(new_path.slacks, slack)

    @debug "  current_path.time_maxrecharge: $(current_path.time_maxrecharge)"
    @debug "  data.q[current_node, next_node]: $(data.q[current_node,next_node])"

    excess = max(
        0,
        max(
            0,
            current_path.time_maxrecharge - slack,
        )
        - (data.B - data.q[current_node,next_node]),
    )
    @debug "  excess: $excess"
    push!(new_path.excesses, excess)

    new_path.time_mincharge = max(
        data.α[next_node],
        current_path.time_mincharge + data.t[current_node,next_node]
    )
    if current_path.recharged > 0
        new_path.time_mincharge += excess
    end
    @debug "  new_path.time_mincharge: $(new_path.time_mincharge)"

    if current_node in data.N_charging
        new_path.time_maxcharge = min(
            data.β[next_node],
            max(
                data.α[next_node],
                current_path.time_mincharge + current_path.time_maxrecharge 
                + data.t[current_node,next_node],
            )
        )
    else
        new_path.time_maxcharge = min(
            data.β[next_node],
            max(
                data.α[next_node],
                current_path.time_maxcharge 
                + data.t[current_node,next_node],
            )
        )
    end
    @debug "  new_path.time_maxcharge: $(new_path.time_maxcharge)"

    # Feasibility check
    if new_path.time_mincharge > data.β[next_node]
        @debug "  Infeasible due to time windows (after update):"
        @debug "    new_path.time_mincharge: $(new_path.time_mincharge)"
        @debug "    data.β[next_node]: $(data.β[next_node])"
        return (false, current_path)
    end
    if new_path.time_mincharge > new_path.time_maxcharge
        @debug "  Infeasible due to time windows (after update):"
        @debug "    new_path.time_mincharge: $(new_path.time_mincharge)"
        @debug "    new_path.time_maxcharge: $(new_path.time_maxcharge)"
        return (false, current_path)
    end

    if current_path.recharged == 0
        new_path.time_maxrecharge = current_path.time_maxrecharge + data.q[current_node,next_node]
        if new_path.time_maxrecharge > data.B
            @debug "  Infeasible due to battery limits:"
            @debug "    new_path.time_maxrecharge: $(new_path.time_maxrecharge)"
            return (false, current_path)
        end
    else
        new_path.time_maxrecharge = min(
            data.B,
            max(
                0,
                current_path.time_maxrecharge - slack,
            )
            + data.q[current_node, next_node]
        )
    end
    @debug "  new_path.time_maxrecharge: $(new_path.time_maxrecharge)"


    @debug "  modified_costs[current_node,next_node]: $(modified_costs[current_node,next_node])"
    new_path.cost += modified_costs[current_node,next_node]

    # Load
    if use_load
        new_path.load += data.d[next_node]
        @debug "  new_path.load: $(new_path.load)"
    end

    # Customer service
    if use_ngroute
        update_path_ngroute!(new_path, next_node, ng_neighborhoods)
        @debug "  new_path.ng_fset: $(new_path.ng_fset)"
    else
        # elementary
        if next_node in data.N_customers
            new_path.served[next_node] = true
        end
        @debug "  new_path.served: $(new_path.served)"
    end

    # Cuts
    if cuts == "SR3"
        update_path_cut_labels_SR3!(new_path, next_node, λvals, λcust)
        @debug "  new_path.cut_flabels: $(new_path.cut_flabels)"
    elseif cuts == "LmSR3"
        update_path_cut_labels_LmSR3!(new_path, next_node, λvals, λcust, λmemory)
        @debug "  new_path.cut_flabels: $(new_path.cut_flabels)"
    end

    return (true, new_path)
end

function bpath_dominates(
    p1::BPathLabel,
    p2::BPathLabel,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
)
    if cuts == "NoCuts"
        if p1.cost > p2.cost
            return false
        end
    else
        # cuts in ["SR3", "LmSR3"]
        if p1.cost - p2.cost > sum(λvals[p1.cut_flabels .& .~p2.cut_flabels])
            return false
        end
    end

    # Time windows
    if p1.time_mincharge > p2.time_mincharge
        return false
    end
    if (
        p1.time_mincharge + p1.time_maxrecharge
        > p2.time_mincharge + p2.time_maxrecharge
    )
        return false
    end
    if (
        p1.time_mincharge + p1.time_maxrecharge - p1.time_maxcharge
        > p2.time_mincharge + p2.time_maxrecharge - p2.time_maxcharge
    ) 
        return false
    end

    # Load
    if use_load
        if p1.load > p2.load
            return false
        end
    end

    # ng-route resources
    if use_ngroute
        # Remark: this only works since the ng-route comparison is the final comparison
        return bitvector_dominates(p1.ng_fset, p2.ng_fset)
    end
    
    return true

end

function add_bpath_to_collection!(
    collection::SortedDict{
        BPATH_VKEY_TYPE,
        BPathLabel,
        Base.Order.ForwardOrdering,
    },
    p1::BPathLabel,
    vkey1::BPATH_VKEY_TYPE,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
)
    added = true
    switched = false
    for (vkey2, p2) in collection
        if !switched && p1.cost > p2.cost
            # p1 cannot dominate p2
            # p2 may dominate p1
            if bpath_dominates(
                p2, p1,
                use_load,
                use_time_windows,
                use_ngroute,
                cuts,
                λvals,
            )
                added = false
                break
            end
            continue
        end
        switched = true
        # p1.cost <= p2.cost
        # p2 cannot dominate p1 
        # p1 may dominate p2
        if bpath_dominates(
            p1, p2,
            use_load,
            use_time_windows,
            use_ngroute,
            cuts,
            λvals,
        )
            pop!(collection, vkey2)
        end
    end

    if added
        insert!(collection, vkey1, p1)
    end

    return added

end

function generate_path_labels_from_node(
    data::EVRPData,
    graph::EVRPGraph,
    modified_costs::Matrix{Float64},
    starting_node::Int,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    ng_neighborhoods::BitMatrix,
    cuts::String,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
    ;
    exploration_method::String = "bestfirst",
    time_limit::Float64 = Inf,
)

    start_time = time()

    # Initialize data structures
    path_labels = Dict(
        current_node => SortedDict{
            BPATH_VKEY_TYPE,
            BPathLabel,
            Base.Order.ForwardOrdering,
        }(Base.Order.ForwardOrdering())
        for current_node in data.N_nodes
    )
    if exploration_method == "bestfirst"
        unexplored_states = SortedSet{BPATH_VKEY_TYPE}()
    elseif exploration_method == "breadthfirst"
        unexplored_states = BPATH_VKEY_TYPE[]
    else
        error("Exploration method $exploration_method not recognized.")
    end

    p = create_new_bpath_label(
        starting_node,
        data,
        ;
        n_cuts = length(λvals),
    )
    vkey = get_vkey(p)
    path_labels[starting_node][vkey] = p
    add_state_to_unexplored_states!(unexplored_states, vkey)

    while !isempty(unexplored_states)
        if time() - start_time > time_limit
            return (true, path_labels)
        end

        # Retrieve most promising unexplored state
        current_vkey = get_next_state!(unexplored_states)
        current_node = current_vkey[end][end]
        if !(current_vkey in keys(path_labels[current_node]))
            continue
        end

        current_path = path_labels[current_node][current_vkey]
        for next_node in setdiff(outneighbors(graph.G, current_node), current_node)
            (feasible, new_path) = compute_new_bpath(
                current_path,
                data,
                graph,
                modified_costs,
                current_node,
                next_node,
                use_load,
                use_time_windows,
                use_ngroute,
                ng_neighborhoods, 
                cuts,
                λvals,
                λcust,
                λmemory,
            )
            !feasible && continue

            new_vkey = get_vkey(new_path)
            added = add_bpath_to_collection!(
                path_labels[next_node],
                new_path,
                new_vkey,
                use_load,
                use_time_windows,
                use_ngroute,
                cuts,
                λvals,
            )
            if added && !(next_node in data.N_depots)
                add_state_to_unexplored_states!(unexplored_states, new_vkey)
            end
        end
    end

    return (false, path_labels)

end

function generate_path_labels_all(
    data::EVRPData,
    graph::EVRPGraph,
    modified_costs::Matrix{Float64},
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    ng_neighborhoods::BitMatrix,
    cuts::String,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
    ;
    exploration_method::String = "bestfirst",
    time_limit::Float64 = Inf,
)

    start_time = time()

    all_path_labels = Dict{
        Int,
        Dict{
            Int,
            SortedDict{
                BPATH_VKEY_TYPE,
                BPathLabel,
                Base.Order.ForwardOrdering,
            }
        }
    }()

    for starting_node in data.N_depots
        (timed_out, all_path_labels[starting_node]) = generate_path_labels_from_node(
            data,
            graph,
            modified_costs,
            starting_node,
            use_load,
            use_time_windows,
            use_ngroute,
            ng_neighborhoods,
            cuts,
            λvals,
            λcust,
            λmemory,
            ;
            exploration_method = exploration_method,
            time_limit = time_limit - (time() - start_time),
        )
        if timed_out
            return (true, time() - start_time, all_path_labels)
        end
    end

    # Cleanup
    ## Remove paths that do not end at the depot
    for starting_node in data.N_depots
        for end_node in setdiff(data.N_nodes, data.N_depots)
            pop!(all_path_labels[starting_node], end_node)
        end
    end

    ## Make sure that empty depot-to-depot paths have two nodes
    for depot in data.N_depots
        for (vkey, path) in all_path_labels[depot][depot]
            if length(path.nodes) == 1
                path.nodes = [depot, depot]
                path.excesses = [0]
                path.slacks = [0]
            end
        end
    end

    return (false, time() - start_time, all_path_labels)

end

function get_negative_path_labels_from_path_labels(
    path_labels::Dict{
        Int, Dict{
            Int, 
            SortedDict{
                BPATH_VKEY_TYPE,
                BPathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    },
)
    return Tuple{Float64, BPathLabel}[
        (path_label.cost, path_label)
        for (k, v) in pairs(path_labels)
        for (k_, v_) in pairs(v)
        for path_label in values(v_)
        if path_label.cost < -1e-3 * (length(path_label.nodes) - 1)
    ]
end

function convert_path_label_to_path(
    path_label::BPathLabel,
    data::EVRPData,
    graph::EVRPGraph,
    ;
    use_load::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    """
    Converts a BPathLabel into a Path.
    Checks that the path is feasible.
    """
    if use_nonlinear_charging
        error("Nonlinear charging not yet supported in benchmark method.")
    end
    p = Path(
        subpaths = Subpath[],
        charging_arcs = ChargingArc[],
        served = zeros(Int, data.n_customers),
        load = 0,
        arcs = NTuple{2, Int}[],
    )
    states = NTuple{3, Int}[]
    current_subpath = Subpath(
        n_customers = data.n_customers,
        starting_node = path_label.nodes[1],
        starting_time = 0, 
        starting_charge = data.B,
    )
    i = path_label.nodes[1]
    for (j, e, s) in zip(path_label.nodes[2:end], path_label.excesses, path_label.slacks)
        current_subpath.current_node = j
        push!(current_subpath.arcs, (i, j))
        current_subpath.starting_time += (e + s)
        current_subpath.starting_charge += (e + s) 
        current_subpath.current_time += (data.t[i,j] + e + s)
        current_subpath.current_charge += (- data.q[i,j] + e + s)
        if use_load
            current_subpath.load += data.d[j]
        end
        if j in data.N_charging
            @assert 0 <= current_subpath.current_charge <= data.B
            @assert 0 <= current_subpath.current_time <= data.T
            push!(
                states, 
                (current_subpath.starting_node, current_subpath.starting_time, current_subpath.starting_charge), 
                (current_subpath.current_node, current_subpath.current_time, current_subpath.current_charge), 
            )
            push!(
                p.subpaths,
                current_subpath,
            )
            current_subpath = Subpath(
                n_customers = data.n_customers,
                starting_node = j,
                starting_time = current_subpath.current_time, 
                starting_charge = current_subpath.current_charge,
            )
        elseif j in data.N_customers
            current_subpath.served[j] += 1
        end
        i = j
    end
    push!(
        states, 
        (current_subpath.starting_node, current_subpath.starting_time, current_subpath.starting_charge), 
        (current_subpath.current_node, current_subpath.current_time, current_subpath.current_charge), 
    )
    push!(
        p.subpaths,
        current_subpath,
    )
    for i in 1:(length(states)÷2)-1
        push!(
            p.charging_arcs, 
            ChargingArc(
                states[2*i][1],
                states[2*i][2],
                states[2*i+1][2],
                states[2*i+1][2] - states[2*i][2],
                states[2*i][3],
                states[2*i+1][3],
                states[2*i+1][3] - states[2*i][3],
            )
        )
    end
    p.served = sum(s.served for s in p.subpaths)
    p.load = sum(s.load for s in p.subpaths)
    p.arcs = vcat([s.arcs for s in p.subpaths]...)
    return p
end

function get_paths_from_negative_path_labels(
    data::EVRPData,
    graph::EVRPGraph,
    path_labels::Vector{Tuple{Float64, BPathLabel}},
    ;
    use_load::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    generated_paths = Dict{
        Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
        Vector{Tuple{Float64, Path}},
    }()
    for (c, path_label) in path_labels
        p = convert_path_label_to_path(
            path_label, 
            data, graph,
            ; 
            use_load = use_load,
            use_nonlinear_charging = use_nonlinear_charging,
            charging_function = charging_function
        )
        add_path_to_generated_paths!(generated_paths, c, p)
    end
    return generated_paths
end

function subproblem_iteration_benchmark(
    data::EVRPData,
    graph::EVRPGraph,
    modified_costs::Matrix{Float64},
    λ::Dict{<:Any, Float64},
    ;
    use_load::Bool = false,
    use_time_windows::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    exploration_method::String = "bestfirst",
    use_ngroute::Bool = false,
    ng_neighborhoods::BitMatrix = falses(data.n_nodes, data.n_nodes),
    time_limit::Float64 = Inf,
)

    if use_nonlinear_charging
        error("Nonlinear charging not yet supported in benchmark method.")
    end

    start_time = time()

    if use_ngroute
        if length(λ) == 0
            cuts = "NoCuts"
        elseif keytype(λ) == NTuple{3, Int}
            cuts = "SR3"
        elseif keytype(λ) == Tuple{NTuple{3, Int}, Tuple{Vararg{Int}}}
            cuts = "LmSR3"
        else
            error("Unrecognized key type for λ: $(keytype(λ))")
        end
    else
        ng_neighborhoods = falses(data.n_nodes, data.n_nodes)
        cuts = "NoCuts"
    end

    if cuts == "NoCuts"
        λvals = Float64[]
        λcust = falses(length(λ), data.n_nodes)
        λmemory = falses(length(λ), data.n_nodes)
    elseif cuts == "SR3"
        λvals, λcust = prepare_lambda(λ, data.n_nodes)
        λmemory = falses(length(λ), data.n_nodes)
    elseif cuts == "LmSR3"
        λvals, λcust, λmemory = prepare_lambda(λ, data.n_nodes)
    end

    (timed_out, path_labels_time, path_labels) = generate_path_labels_all(
        data, 
        graph,
        modified_costs,
        use_load,
        use_time_windows,
        use_ngroute,
        ng_neighborhoods,
        cuts,
        λvals,
        λcust,
        λmemory,
        ;
        exploration_method = exploration_method,
        time_limit = time_limit - (time() - start_time),
    )
    path_labels_time = round(path_labels_time, digits=3)
    if timed_out
        return (
            # timed_out
            true, 
            # generated_paths
            Dict{
                Tuple{NTuple{3, Int}, NTuple{3, Int}}, 
                Vector{Path},
            }(),
            # negative_path_labels_count
            0,
            # path_labels_time
            path_labels_time,
        )
    end
    
    negative_path_labels = get_negative_path_labels_from_path_labels(path_labels)
    negative_path_labels_count = length(negative_path_labels)
    
    generated_paths = get_paths_from_negative_path_labels(
        data, graph, 
        negative_path_labels,
        ;
        use_load = use_load,
    )

    return (
        false, # timed_out
        generated_paths, 
        negative_path_labels_count, 
        path_labels_time,
    )
end
