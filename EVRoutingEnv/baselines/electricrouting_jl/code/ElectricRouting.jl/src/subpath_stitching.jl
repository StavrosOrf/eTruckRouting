mutable struct SubpathLabel <: Label
    cost::Float64
    time_taken::Int
    time_end_earliest::Int
    time_start_latest::Int
    P::Int
    Q::Int
    charge_taken::Int
    load::Int
    served::Vector{Int}
    nodes::Vector{Int}
    ng_fset::BitVector
    ng_bset::BitVector
    ng_residue::BitVector
    cut_flabels::BitVector
    cut_blabels::BitVector
    cut_mlabels::BitVector
end

Base.copy(s::SubpathLabel) = SubpathLabel(
    s.cost,
    s.time_taken,
    s.time_end_earliest,
    s.time_start_latest,
    s.P,
    s.Q,
    s.charge_taken,
    s.load,
    copy(s.served),
    copy(s.nodes),
    copy(s.ng_fset),
    copy(s.ng_bset),
    copy(s.ng_residue),
    copy(s.cut_flabels),
    copy(s.cut_blabels),
    copy(s.cut_mlabels),
)

function Base.show(io::IO, s::SubpathLabel)
    @printf(io, """SubpathLabel(
        cost:       %e
        nodes:      %s
        time:       %d, %d, %d
        charge:     %d
        load:       %d
        served:     %s
        ng-route:   %s, %s, %s
        cuts:       %s, %s, %s
    )""",
        s.cost, 
        s.nodes, 
        s.time_taken, s.time_end_earliest, s.time_start_latest, 
        s.charge_taken, s.load, 
        [(i, s.served[i]) for i in 1:length(s.served) if s.served[i] > 0], 
        findall(x -> x == 1, s.ng_fset), 
        findall(x -> x == 1, s.ng_bset),
        findall(x -> x == 1, s.ng_residue),
        findall(x -> x == 1, s.cut_flabels),
        findall(x -> x == 1, s.cut_blabels),
        findall(x -> x == 1, s.cut_mlabels),
    )
end

# Note: get_vkey() is a key, not for comparison
get_vkey(s::SubpathLabel) = (
    # Necessary for sorting purposes
    s.cost, 
    # Helpful for sorting purposes
    s.time_end_earliest,
    - s.time_start_latest,
    s.Q,
    s.charge_taken,
    s.load,
    # Necessary to uniquely determine a subpath
    s.nodes, 
)

SUBPATH_VKEY_TYPE = Tuple{
    Float64,
    Int,
    Int,
    Int,
    Int,
    Int,
    Vector{Int},
}

mutable struct PPathLabel <: Label
    cost::Float64
    time_end_earliest::Int
    charge_end_max::Int
    load::Int
    subpath_label_vkeys::Vector{SUBPATH_VKEY_TYPE}
    charging_amounts_max::Vector{Int}
    charging_amounts::Vector{Int}
    served::Vector{Int}
    nodes::Vector{Int}
    ng_fset::BitVector
    cut_flabels::BitVector
end

Base.copy(p::PPathLabel) = PPathLabel(
    p.cost,
    p.time_end_earliest,
    p.charge_end_max,
    p.load,
    copy(p.subpath_label_vkeys),
    copy(p.charging_amounts_max),
    copy(p.charging_amounts),
    copy(p.served),
    copy(p.nodes),
    copy(p.ng_fset),
    copy(p.cut_flabels),
)

function Base.show(io::IO, p::PPathLabel)
    @printf(io, """PPathLabel(
        cost:               %e
        nodes:              %s
        time:               %d
        charge:             %d
        load:               %d
        subpath_labels:     %s
        served:             %s
        ng-route:           %s
        cuts:               %s
    )""",
        p.cost,
        p.nodes,
        p.time_end_earliest,
        p.charge_end_max,
        p.load,
        p.subpath_label_vkeys,
        [(i, p.served[i]) for i in 1:length(p.served) if p.served[i] > 0],
        findall(x -> x == 1, p.ng_fset),
        findall(x -> x == 1, p.cut_flabels)
    )
end

# Note: get_vkey() is a key, not for comparison
get_vkey(p::PPathLabel) = (
    # Necessary for sorting purposes
    p.cost,
    # Helpful for sorting purposes
    p.time_end_earliest,
    - p.charge_end_max,
    p.load,
    # Necessary to uniquely determine a subpath
    p.nodes,
)

PPATH_VKEY_TYPE = Tuple{
    Float64, 
    Int, 
    Int, 
    Int, 
    Vector{Int},
}

function create_new_subpath_label(
    starting_node::Int,
    data::EVRPData,
    ;
    n_cuts::Int = 0,
    ng_neighborhoods::BitMatrix = falses(data.n_nodes, data.n_nodes),
    λmemory::BitMatrix = falses(n_cuts, data.n_nodes),
)
    # ng-route resources
    # Π: Forward NG-set
    ng_fset = falses(data.n_nodes)
    ng_fset[starting_node] = true    
    # Φ: Backward NG-set
    ng_bset = falses(data.n_nodes)
    ng_bset[starting_node] = true
    # Ω: Nodes that if in the previous forward NG-set, stay in the next forward NG-set
    ng_residue = copy(ng_neighborhoods[:, starting_node])

    return SubpathLabel(
        0.0,                                # cost
        0,                                  # time_taken
        0,                                  # time_end_earliest
        data.T,                             # time_start_latest
        0,                                  # charge_taken
        0,                                  # load
        0,                                  # P
        0,                                  # Q
        zeros(Int, data.n_customers),       # served
        [starting_node,],                   # nodes
        ng_fset,                            # ng_fset
        ng_bset,                            # ng_bset
        ng_residue,                         # ng_residue
        falses(n_cuts),                     # cut_flabels
        falses(n_cuts),                     # cut_blabels
        copy(λmemory[:, starting_node]),    # cut_mlabels
    )
end

function update_subpath_ngroute!(
    new_subpath::SubpathLabel,
    next_node::Int,
    ng_neighborhoods::BitMatrix
)
    # update forward ng-set
    ngroute_update_fset!(new_subpath, next_node, ng_neighborhoods)
    ## IMPORTANT: update backward ng-set before residue, 
    ## since residue affects backward ng-set
    # update backward ng-set
    ngroute_update_bset!(new_subpath, next_node)
    # update ng-residue
    ngroute_update_residue!(new_subpath, next_node, ng_neighborhoods)
    return 
end

function update_subpath_cut_labels_SR3!(
    new_subpath::SubpathLabel,
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
)
    ## IMPORTANT: update cost before flabels
    ## since flabels affect cost
    SR3_update_cost!(new_subpath, next_node, λvals, λcust)
    SR3_update_cut_flabels!(new_subpath, next_node, λcust)
    return
end

function update_subpath_cut_labels_LmSR3!(
    new_subpath::SubpathLabel,
    next_node::Int,
    λvals::Vector{Float64},
    λcust::BitMatrix,
    λmemory::BitMatrix,
)
    ## IMPORTANT: update cost before flabels
    ## since flabels affect cost
    lmSR3_update_cost!(new_subpath, next_node, λvals, λcust, λmemory)
    # Update cut flabels
    lmSR3_update_cut_flabels!(new_subpath, next_node, λcust, λmemory)
    # Update cut blabels
    lmSR3_update_cut_blabels!(new_subpath, next_node, λcust)
    # Update cut mlabels
    lmSR3_update_cut_mlabels!(new_subpath, next_node, λmemory)
    return
end

function compute_new_subpath(
    current_subpath::SubpathLabel,
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
    # customer service
    if use_ngroute
        if current_subpath.ng_fset[next_node]
            return (false, current_subpath)
        end
    else
        # elementary
        if next_node in data.N_customers && current_subpath.served[next_node] > 0
            return (false, current_subpath)
        end
    end

    # Create new subpath
    new_subpath = copy(current_subpath)

    # Time windows feasibility
    if use_time_windows
        ## time_end_earliest
        new_subpath.time_end_earliest = max(
            new_subpath.time_end_earliest + data.t[current_node, next_node], 
            data.α[next_node],
        )
        if new_subpath.time_end_earliest > data.β[next_node]
            return (false, current_subpath)
        end
        if new_subpath.time_end_earliest + graph.min_t[next_node] > data.T
            return (false, current_subpath)
        end

        ## time_start_latest
        new_subpath.time_taken += data.t[current_node, next_node]
        new_subpath.time_start_latest = min(
            new_subpath.time_start_latest,
            data.β[next_node] - new_subpath.time_taken,
        )
        if new_subpath.time_start_latest < 0
            return (false, current_subpath)
        end
        # breakpoints
        new_subpath.P = min(
            new_subpath.time_start_latest,
            new_subpath.time_end_earliest - new_subpath.time_taken
        )
        new_subpath.Q = max(
            new_subpath.time_end_earliest - new_subpath.time_start_latest,
            new_subpath.time_taken,
        )
    else
        new_subpath.time_end_earliest += data.t[current_node, next_node]
        if new_subpath.time_end_earliest + graph.min_t[next_node] > data.T
            return (false, current_subpath)
        end
        new_subpath.time_taken += data.t[current_node, next_node]
        # breakpoints
        new_subpath.Q = new_subpath.time_end_earliest
    end
    
    # charge feasibility
    new_subpath.charge_taken += data.q[current_node, next_node]
    if new_subpath.charge_taken + graph.min_q[next_node] > data.B
        return (false, current_subpath)
    end

    # load feasibility
    if use_load
        new_subpath.load += data.d[next_node]
        if new_subpath.load > data.C
            return (false, current_subpath)
        end
    end

    # modified costs
    new_subpath.cost += modified_costs[current_node, next_node]

    # nodes
    push!(new_subpath.nodes, next_node)

    # Customer service
    if next_node in data.N_customers
        new_subpath.served[next_node] += 1
    end
    if use_ngroute
        update_subpath_ngroute!(new_subpath, next_node, ng_neighborhoods)
    end

    # Cuts
    if cuts == "SR3"
        update_subpath_cut_labels_SR3!(new_subpath, next_node, λvals, λcust)
    elseif cuts == "LmSR3" 
        update_subpath_cut_labels_LmSR3!(new_subpath, next_node, λvals, λcust, λmemory)
    end

    return (true, new_subpath)
end

function subpath_dominates(
    s1::SubpathLabel, 
    s2::SubpathLabel,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
)
    # Reduced cost comparison (depends on cuts)
    if cuts == "NoCuts"
        if s1.cost > s2.cost
            return false
        end
    elseif cuts == "SR3"
        if s1.cost - s2.cost > sum(λvals[s1.cut_flabels .& .~s2.cut_flabels])
            return false
        end
    elseif cuts == "LmSR3"
        if s1.cost - s2.cost > (
              sum(λvals[.~s1.cut_mlabels    .& .~s2.cut_mlabels     .& s1.cut_flabels   .& .~s2.cut_flabels])
            + sum(λvals[.~s1.cut_mlabels    .& .~s2.cut_mlabels     .& s1.cut_blabels   .& .~s2.cut_blabels])
            + sum(λvals[s1.cut_mlabels      .& .~s2.cut_mlabels     .& .~s2.cut_flabels .& .~s2.cut_blabels])
            + sum(λvals[s1.cut_mlabels      .& .~s2.cut_mlabels     .& s1.cut_flabels   .& (s2.cut_flabels .⊻ s2.cut_blabels)])
            + sum(λvals[.~s1.cut_mlabels    .& s2.cut_mlabels       .& s1.cut_flabels   .& s1.cut_blabels])
            + sum(λvals[.~s1.cut_mlabels    .& s2.cut_mlabels       .& .~s2.cut_flabels .& (s1.cut_flabels .⊻ s1.cut_blabels)])
            + sum(λvals[s1.cut_mlabels      .& s2.cut_mlabels       .& s1.cut_flabels   .& .~s2.cut_flabels])
        )
            return false
        end
    end
    # Time resources comparison
    if use_time_windows
        if s1.time_end_earliest > s2.time_end_earliest
            return false
        end
        if s1.time_start_latest < s2.time_start_latest
            return false
        end
        if s1.Q > s2.Q
            return false
        end
    else
        if s1.time_end_earliest > s2.time_end_earliest
            return false
        end
    end
    # Charge taken
    if s1.charge_taken > s2.charge_taken
        return false
    end
    # Load 
    if use_load
        if s1.load > s2.load
            return false
        end
    end
    # ng-route resources
    if use_ngroute
        # Remark: this only works since the ng-route comparison is the final comparison
        return (
            bitvector_dominates(s1.ng_fset, s2.ng_fset) 
            && bitvector_dominates(s1.ng_bset, s2.ng_bset) 
            && bitvector_dominates(s1.ng_residue, s2.ng_residue)
        )
    end

    return true
end


function add_subpath_to_collection!(
    collection::SortedDict{
        SUBPATH_VKEY_TYPE,
        SubpathLabel,
        Base.Order.ForwardOrdering,
    },
    s1::SubpathLabel,
    vkey1::SUBPATH_VKEY_TYPE,
    use_load::Bool,
    use_time_windows::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
)
    added = true
    switched = false
    for (vkey2, s2) in collection
        if !switched && s1.cost > s2.cost
            # s1 cannot dominate s2
            if subpath_dominates(
                s2, s1,
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
        # s1.cost ≤ s2.cost
        # s1 may dominate s2
        if subpath_dominates(
            s1, s2,
            use_load, 
            use_time_windows, 
            use_ngroute, 
            cuts, 
            λvals,
        )
            pop!(collection, vkey2)
        end
    end

    # Finally, add s1
    if added
        insert!(collection, vkey1, s1)
    end

    return added
end


function generate_subpath_labels_from_node(
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
    subpath_labels = Dict(
        current_node => SortedDict{
            SUBPATH_VKEY_TYPE,
            SubpathLabel,
            Base.Order.ForwardOrdering,
        }(Base.Order.ForwardOrdering())
        for current_node in data.N_nodes
    )
    if exploration_method == "bestfirst"
        unexplored_states = SortedSet{SUBPATH_VKEY_TYPE}()
    elseif exploration_method == "breadthfirst"
        unexplored_states = SUBPATH_VKEY_TYPE[]
    else
        error("Unknown exploration method: $exploration_method")
    end

    s = create_new_subpath_label(
        starting_node,
        data,
        ;
        n_cuts = length(λvals),
        ng_neighborhoods = ng_neighborhoods,
        λmemory = λmemory,
    )
    vkey = get_vkey(s)
    subpath_labels[starting_node][vkey] = s
    add_state_to_unexplored_states!(unexplored_states, vkey)

    # Iterate over unextended labels
    while !isempty(unexplored_states)
        if time() - start_time > time_limit
            return (true, subpath_labels)
        end
        # Retrieve most promising unexplored state
        current_vkey = get_next_state!(unexplored_states)
        current_node = current_vkey[end][end]
        if !(current_vkey in keys(subpath_labels[current_node]))
            continue
        end

        current_subpath = subpath_labels[current_node][current_vkey]
        for next_node in setdiff(outneighbors(graph.G, current_node), current_node)
            # Check feasibility and create new subpath
            (feasible, new_subpath) = compute_new_subpath(
                current_subpath,
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

            new_vkey = get_vkey(new_subpath)
            added = add_subpath_to_collection!(
                subpath_labels[next_node],
                new_subpath,
                new_vkey,
                use_load,
                use_time_windows,
                use_ngroute,
                cuts,
                λvals,
            )

            if added && next_node in data.N_customers
                add_state_to_unexplored_states!(unexplored_states, new_vkey)
            end
        end
    end

    # Cleanup
    ## Remove subpaths ending at customers
    for end_node in data.N_customers
        pop!(subpath_labels, end_node)
    end

    ## Remove empty initial subpath
    s = create_new_subpath_label(
        starting_node,
        data,
        ;
        n_cuts = length(λvals),
        ng_neighborhoods = ng_neighborhoods,
        λmemory = λmemory,
    )
    pop!(subpath_labels[starting_node], get_vkey(s))

    return (false, subpath_labels)

end

function generate_subpath_labels_all(
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

    all_subpath_labels = Dict{
        Int,
        Dict{
            Int,
            SortedDict{
                SUBPATH_VKEY_TYPE,
                SubpathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    }()
    # Threads.@threads for starting_node in data.N_depots_charging
    for starting_node in data.N_depots_charging
        (timed_out, all_subpath_labels[starting_node]) = generate_subpath_labels_from_node(
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
            return (true, time() - start_time, all_subpath_labels)
        end
    end

    return (false, time() - start_time, all_subpath_labels)
end


function create_new_path_label(
    starting_node::Int,
    data::EVRPData,
    ;
    n_cuts::Int = 0,
)
    # ng-route resources
    # Π: Forward NG-set
    ng_fset = falses(data.n_nodes)
    ng_fset[starting_node] = true

    return PPathLabel(
        0.0,                            # cost
        0,                              # time_end_earliest
        data.B,                         # charge_end_max
        0,                              # load
        SubpathLabel[],                 # subpath_labels
        Int[],                          # charging_amounts_max
        Int[],                          # charging_amounts
        zeros(Int, data.n_customers),   # served
        Int[starting_node,],            # nodes
        ng_fset,                        # ng_fset
        falses(n_cuts),                 # cut_flabels
    )
end

function check_path_ngroute(
    path_fset::BitVector,
    node::Int,
    subpath_fset::BitVector,
    subpath_residue::BitVector,
    subpath_bset::BitVector,
)
    new_path_inf = path_fset .& subpath_bset
    new_path_inf[node] = false
    if any(new_path_inf)
        return (false, path_fset)
    else
        return (true, subpath_fset .| (path_fset .& subpath_residue))
    end
end

function update_path_cut_labels_SR3!(
    new_path::PPathLabel,
    s::SubpathLabel,
    λvals::Vector{Float64},
)
    # Update reduced cost
    new_path.cost -= sum(λvals[new_path.cut_flabels .& s.cut_flabels])
    # Update cut labels
    new_path.cut_flabels .⊻= s.cut_flabels
    return
end

function update_path_cut_labels_lmSR3!(
    new_path::PPathLabel,
    s::SubpathLabel,
    λvals::Vector{Float64},
)
    # Update reduced cost
    new_path.cost -= sum(λvals[new_path.cut_flabels .& s.cut_blabels])
    # Update cut labels
    new_path.cut_flabels .&= s.cut_mlabels
    new_path.cut_flabels .⊻= s.cut_flabels
    return
end

function compute_new_path(
    current_path::PPathLabel,
    data::EVRPData,
    graph::EVRPGraph,
    current_node::Int,
    next_node::Int,
    s::SubpathLabel,
    use_load::Bool,
    use_time_windows::Bool,
    use_nonlinear_charging::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64}
    ;
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    # @debug "Computing new path with: path = $current_path, subpath = $s"

    # Time windows feasibility
    if use_time_windows
        if use_nonlinear_charging
            # time_end_earliest
            charging_duration_min = compute_charging_duration(
                charging_function,
                current_path.charge_end_max, # BendMax{p}
                s.charge_taken, # B(s)
            )
        else
            # time_end_earliest
            charging_duration_min = max(
                0, 
                s.charge_taken - current_path.charge_end_max, # B(s) - BendMax{p}
            )
        end
        # @debug "  charging_duration_min = $charging_duration_min"

        if current_path.time_end_earliest + charging_duration_min > s.time_start_latest
            # @debug "  Infeasible extension from node $current_node to node $next_node (time windows): $(current_path.time_end_earliest), $charging_duration_min, $(s.time_start_latest)"
            return (false, current_path)
        end
        new_time_end_earliest = max(
            s.time_end_earliest,
            s.time_taken + current_path.time_end_earliest + charging_duration_min,
        )
        # @debug "  new_time_end_earliest = $new_time_end_earliest"
        if new_time_end_earliest + graph.min_t[next_node] > data.T
            # @debug "  Infeasible extension from node $current_node to node $next_node (time boundary): $(new_time_end_earliest), $(graph.min_t[next_node]), $(data.T)"
            return (false, current_path)
        end

        if use_nonlinear_charging
            # charge_end_max 
            charge_end_max_before_subpath = min(
                data.B,
                max(
                    compute_end_charge(
                        charging_function,
                        current_path.charge_end_max, # BendMax{p}
                        max(
                            0, 
                            s.P - current_path.time_end_earliest,
                        ) # (P(s) - \TendMin{p})⁺
                    ),
                    s.charge_taken
                ),
            )
        else
            charge_end_max_before_subpath = min(
                data.B,
                    max(
                    current_path.charge_end_max,
                    current_path.charge_end_max
                    + s.P - current_path.time_end_earliest,
                    s.charge_taken
                ),
            )
        end
        # @debug "  charge_end_max_before_subpath = $charge_end_max_before_subpath"
        new_charge_end_max = charge_end_max_before_subpath - s.charge_taken
        # @debug "  new_charge_end_max = $new_charge_end_max"
    else
        if use_nonlinear_charging
            # time_end_earliest
            charging_duration_min = compute_charging_duration(
                charging_function,
                current_path.charge_end_max,
                s.charge_taken,
            )
        else
            # time_end_earliest
            charging_duration_min = max(
                0,
                s.charge_taken - current_path.charge_end_max,
            )
        end
        # @debug "  charging_duration_min = $charging_duration_min"
        new_time_end_earliest = s.time_taken + current_path.time_end_earliest + charging_duration_min

        # @debug "  new_time_end_earliest = $new_time_end_earliest"
        if new_time_end_earliest + graph.min_t[next_node] > data.T
            # @debug "  Infeasible extension from node $current_node to node $next_node (time boundary): $(new_time_end_earliest), $(graph.min_t[next_node]), $(data.T)"
            return (false, current_path)
        end
        # charge_end_max 
        new_charge_end_max = max(
            0, 
            current_path.charge_end_max - s.charge_taken,
        )
        # @debug "  charge_end_max = $charge_end_max"
        # charge_end_max 
        charge_end_max_before_subpath = min(
            max(
                current_path.charge_end_max,
                s.charge_taken
            ),
            data.B,
        )
        # @debug "  charge_end_max_before_subpath = $charge_end_max_before_subpath"
    end

    # load feasibility
    if use_load
        new_load = current_path.load + s.load
        # @debug "  current_path.load  = $(current_path.load), s.load = $(s.load), new_load = $new_load"
        if new_load > data.C
            # @debug "  Infeasible extension from node $current_node to node $next_node (load): $(new_load), $(data.C)"
            return (false, current_path)
        end
    end

    # elementarity
    if use_ngroute
        (feasible, new_path_ng_fset) = check_path_ngroute(
            current_path.ng_fset, 
            current_node,
            s.ng_fset,
            s.ng_residue,
            s.ng_bset,
        )
        if !feasible
            # @debug "  Infeasible extension from node $current_node to node $next_node (ng-route): $(current_path.ng_fset), $(s.ng_bset)"
            return (false, current_path)
        end
    else
        # elementary
        for i in eachindex(s.served)
            if s.served[i] && current_path.served[i]
                # @debug "  Infeasible extension from node $current_node to node $next_node (elementary): $(current_path.served), $(current_path.served)"
                return (false, current_path)
            end
        end
    end

    new_path = copy(current_path)

    new_path.time_end_earliest = new_time_end_earliest
    new_path.charge_end_max = new_charge_end_max
    if use_load
        new_path.load = new_load
    end

    # modified costs
    new_path.cost += s.cost


    if length(current_path.subpath_label_vkeys) > 0
        amount_charged = charge_end_max_before_subpath - current_path.charge_end_max
        push!(new_path.charging_amounts_max, amount_charged)
        push!(new_path.charging_amounts, amount_charged)
        # @debug "  Charging amounts so far: $(new_path.charging_amounts)"
        # @debug "  Amount to be charged: $(amount_charged)"
    end

    push!(new_path.subpath_label_vkeys, get_vkey(s))
    new_path.nodes = vcat(new_path.nodes, s.nodes[2:end])

    # Customer service
    new_path.served .+= s.served
    if use_ngroute
        new_path.ng_fset = new_path_ng_fset
    end

    # Cuts
    if cuts == "SR3"
        update_path_cut_labels_SR3!(new_path, s, λvals)
    elseif cuts == "LmSR3"
        update_path_cut_labels_lmSR3!(new_path, s, λvals)
    end

    return (true, new_path)    
end


function ppath_dominates(
    p1::PPathLabel, 
    p2::PPathLabel,
    use_load::Bool,
    use_time_windows::Bool,
    use_nonlinear_charging::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
    ;
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    # Reduced cost comparison (depends on cuts)
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
    
    # Time resources comparison (time and charge)

    if use_nonlinear_charging
        if (
            p1.time_end_earliest 
            + compute_charging_duration(
                charging_function,
                p1.charge_end_max,
                p2.charge_end_max,
            )
        ) > p2.time_end_earliest
            return false
        end
    else
        if (
            p1.time_end_earliest 
            + max(0, p2.charge_end_max - p1.charge_end_max)
        ) > p2.time_end_earliest
            return false
        end
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


function add_ppath_to_collection!(
    collection::SortedDict{
        PPATH_VKEY_TYPE,
        PPathLabel,
        Base.Order.ForwardOrdering,
    },
    p1::PPathLabel,
    vkey1::PPATH_VKEY_TYPE,
    use_load::Bool,
    use_time_windows::Bool,
    use_nonlinear_charging::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
    ;
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    added = true
    switched = false
    for (vkey2, p2) in collection
        if !switched && p1.cost > p2.cost
            # p1 cannot dominate p2
            if ppath_dominates(
                p2, p1,
                use_load,
                use_time_windows,
                use_nonlinear_charging,
                use_ngroute,
                cuts,
                λvals,
                ;
                charging_function = charging_function,
            )
                added = false
                break
            end
            continue
        end
        switched = true
        # p1.cost ≤ p2.cost
        # p1 may dominate p2
        if ppath_dominates(
            p1, p2,
            use_load, 
            use_time_windows, 
            use_nonlinear_charging,
            use_ngroute, 
            cuts, 
            λvals,
            ;
            charging_function = charging_function,
        )
            pop!(collection, vkey2)
        end
    end

    # Finally, add p1
    if added
        insert!(collection, vkey1, p1)
    end

    return added
end


function generate_path_labels_from_node(
    data::EVRPData, 
    graph::EVRPGraph,
    subpath_labels::Dict{
        Int, 
        Dict{
            Int,
            SortedDict{
                SUBPATH_VKEY_TYPE,
                SubpathLabel,
                Base.Order.ForwardOrdering,
            },
        }
    },
    starting_node::Int,
    use_load::Bool,
    use_time_windows::Bool,
    use_nonlinear_charging::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
    ;
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    exploration_method::String = "bestfirst",
    time_limit::Float64 = Inf,
)

    start_time = time()

    # Initialize data structures
    path_labels = Dict(
        current_node => SortedDict{
            PPATH_VKEY_TYPE,
            PPathLabel,
            Base.Order.ForwardOrdering,
        }(Base.Order.ForwardOrdering())
        for current_node in data.N_depots_charging
    )
    if exploration_method == "bestfirst"
        unexplored_states = SortedSet{PPATH_VKEY_TYPE}()
    elseif exploration_method == "breadthfirst"
        unexplored_states = PPATH_VKEY_TYPE[]
    else
        error("Unknown exploration method: $exploration_method")
    end

    p = create_new_path_label(
        starting_node,
        data,
        ;
        n_cuts = length(λvals),
    )
    vkey = get_vkey(p)
    path_labels[starting_node][vkey] = p
    add_state_to_unexplored_states!(unexplored_states, vkey)

    # Iterate over unextended labels
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
        for next_node in data.N_depots_charging
            for (_, s) in pairs(subpath_labels[current_node][next_node])
                (feasible, new_path) = compute_new_path(
                    current_path,
                    data,
                    graph,
                    current_node,
                    next_node,
                    s,
                    use_load,
                    use_time_windows,
                    use_nonlinear_charging,
                    use_ngroute,
                    cuts,
                    λvals
                    ;
                    charging_function = charging_function,
                )
                !feasible && continue
                
                new_vkey = get_vkey(new_path)

                added = add_ppath_to_collection!(
                    path_labels[next_node],
                    new_path,
                    new_vkey,
                    use_load,
                    use_time_windows,
                    use_nonlinear_charging,
                    use_ngroute,
                    cuts,
                    λvals,
                    ;
                    charging_function = charging_function, 
                )
                if added && next_node in data.N_charging
                    add_state_to_unexplored_states!(unexplored_states, new_vkey)
                end
            end
        end
    end

    return (false, path_labels)
end


function generate_path_labels_all(
    data::EVRPData, 
    graph::EVRPGraph,
    subpath_labels::Dict{
        Int, 
        Dict{
            Int,
            SortedDict{
                SUBPATH_VKEY_TYPE,
                SubpathLabel,
                Base.Order.ForwardOrdering,
            },
        }
    },
    use_load::Bool,
    use_time_windows::Bool,
    use_nonlinear_charging::Bool,
    use_ngroute::Bool,
    cuts::String,
    λvals::Vector{Float64},
    ;
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
    exploration_method::String = "bestfirst",
    time_limit::Float64 = Inf,
)

    start_time = time()

    all_path_labels = Dict{
        Int,
        Dict{
            Int,
            SortedDict{
                PPATH_VKEY_TYPE,
                PPathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    }()
    # Threads.@threads for starting_node in graph.N_depots
    for starting_node in graph.N_depots
        (timed_out, all_path_labels[starting_node]) = generate_path_labels_from_node(
            data,
            graph,
            subpath_labels,
            starting_node,
            use_load,
            use_time_windows,
            use_nonlinear_charging,
            use_ngroute,
            cuts,
            λvals,
            ;
            charging_function = charging_function,
            exploration_method = exploration_method,
            time_limit = time_limit - (time() - start_time),
        )
        if timed_out
            return (true, time() - start_time, all_path_labels)
        end
    end

    for starting_node in graph.N_depots, end_node in data.N_charging
        pop!(all_path_labels[starting_node], end_node)
    end

    return (false, time() - start_time, all_path_labels)
end

function get_negative_path_labels_from_path_labels(
    path_labels::Dict{
        Int, Dict{
            Int, 
            SortedDict{
                PPATH_VKEY_TYPE,
                PPathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    },
)
    return Tuple{Float64, PPathLabel}[
        (path_label.cost, path_label)
        for (k, v) in pairs(path_labels)
        for (k_, v_) in pairs(v)
        for path_label in values(v_)
        if path_label.cost < -1e-3 * (length(path_label.nodes) - 1)
    ]
end

function convert_path_label_to_path(
    path_label::PPathLabel,
    data::EVRPData,
    graph::EVRPGraph,
    all_subpath_labels::Dict{
        Int,
        Dict{
            Int,
            SortedDict{
                SUBPATH_VKEY_TYPE,
                SubpathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    },
    ;
    use_load::Bool = false,
    use_nonlinear_charging::Bool = false,
    charging_function::PiecewiseLinearIncreasingConcaveFunction = PiecewiseLinearIncreasingConcaveFunction(
        [data.B], [1],
    ),
)
    """
    Converts a PPathLabel into a Path.
    Checks that the path is feasible.
    """
    current_time, current_charge = (0, data.B)
    prev_time, prev_charge = current_time, current_charge
    subpath_label_vkeys = copy(path_label.subpath_label_vkeys)
    charging_amounts = copy(path_label.charging_amounts)
    @assert length(subpath_label_vkeys) == length(charging_amounts) + 1
    current_node = subpath_label_vkeys[1][end][1]

    p = Path(
        subpaths = Subpath[],
        charging_arcs = ChargingArc[],
        served = zeros(Int, graph.n_customers),
        load = 0,
        arcs = NTuple{2, Int}[],
    )
    while true
        subpath_label_vkey = popfirst!(subpath_label_vkeys)
        prev_node = subpath_label_vkey[end][1]
        @assert prev_node == current_node
        current_node = subpath_label_vkey[end][end]
        subpath_label = all_subpath_labels[prev_node][current_node][subpath_label_vkey]
        prev_time = current_time
        prev_charge = current_charge
        current_time = max(
            subpath_label.time_end_earliest,
            current_time + subpath_label.time_taken,
        )
        current_charge = current_charge - subpath_label.charge_taken
        @assert current_time <= data.T
        @assert current_charge >= 0
        s = Subpath(
            n_customers = graph.n_customers,
            starting_node = subpath_label.nodes[1],
            starting_time = prev_time,
            starting_charge = prev_charge,
            current_node = current_node,
            arcs = collect(zip(subpath_label.nodes[1:end-1], subpath_label.nodes[2:end])),
            current_time = current_time,
            current_charge = current_charge,
            load = (use_load ? subpath_label.load : 0),
            served = subpath_label.served,
        )
        push!(p.subpaths, s)
        if length(charging_amounts) == 0 
            break
        end
        charging_amount = popfirst!(charging_amounts)
        prev_time = current_time
        prev_charge = current_charge
        if use_nonlinear_charging
            time_diff = compute_charging_duration(
                charging_function,
                current_charge,
                current_charge + charging_amount,
            )
        else
            time_diff = charging_amount
        end
        current_time = current_time + time_diff
        current_charge = current_charge + charging_amount
        @assert current_time <= data.T
        @assert current_charge <= data.B
        a = ChargingArc(
            node = current_node,
            time_start = prev_time,
            time_end = current_time,
            time_diff = time_diff,
            charge_start = prev_charge,
            charge_end = current_charge,
            charge_diff = charging_amount,
        )
        push!(p.charging_arcs, a)
    end
    p.served = sum(s.served for s in p.subpaths)
    p.load = sum(s.load for s in p.subpaths)
    p.arcs = vcat([s.arcs for s in p.subpaths]...)
    return p
end

function get_paths_from_negative_path_labels(
    data::EVRPData,
    graph::EVRPGraph,
    path_labels::Vector{Tuple{Float64, PPathLabel}},
    subpath_labels::Dict{
        Int,
        Dict{
            Int,
            SortedDict{
                SUBPATH_VKEY_TYPE,
                SubpathLabel,
                Base.Order.ForwardOrdering,
            },
        },
    },
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
            subpath_labels,
            ; 
            use_load = use_load,
            use_nonlinear_charging = use_nonlinear_charging,
            charging_function = charging_function
        )
        add_path_to_generated_paths!(generated_paths, c, p)
    end
    return generated_paths
end

function subproblem_iteration_ours(
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

    (timed_out, subpath_labels_time, subpath_labels) = generate_subpath_labels_all(
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
    subpath_labels_time = round(subpath_labels_time, digits=3)
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
            # subpath_labels_time
            subpath_labels_time,
            # path_labels_time
            0.0,
        )
    end

    (timed_out, path_labels_time, path_labels) = generate_path_labels_all(
        data,
        graph,
        subpath_labels,
        use_load,
        use_time_windows,
        use_nonlinear_charging,
        use_ngroute,
        cuts,
        λvals,
        ;
        charging_function = charging_function,
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
            # subpath_labels_time
            subpath_labels_time,
            # path_labels_time
            path_labels_time,
        )
    end

    negative_path_labels = get_negative_path_labels_from_path_labels(path_labels)
    negative_path_labels_count = length(negative_path_labels)

    generated_paths = get_paths_from_negative_path_labels(
        data, graph, 
        negative_path_labels,
        subpath_labels,
        ;
        use_load = use_load,
        use_nonlinear_charging = use_nonlinear_charging,
        charging_function = charging_function,
    )
    return (
        false, # timed_out
        generated_paths,
        negative_path_labels_count,
        subpath_labels_time,
        path_labels_time,
    )
end
