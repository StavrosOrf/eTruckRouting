# Rounded capacity cuts over customer successions.
#
# For a customer set S, every integer solution enters S (a customer of S
# directly preceded, among customers, by the depot or a customer outside S)
# at least r(S) = ceil(d(S) / payload) times, since one route carries at most
# `payload`. Entries are counted on successions rather than road arcs, so a
# route that leaves S only to recharge does not count twice and the cut stays
# as strong as in a classic CVRP.
#
# The dual sigma_S >= 0 is earned when a route steps from outside S into S:
# the pricing term rho(l, w) = sum of sigma_S over cuts with w in S and l not
# in S depends on the last customer `l` of a label, which charger-node
# dominance accounts for exactly (see `cut_slack`).

struct CapacityCut
    members::BitVector
    rhs::Float64
end

entries(cut::CapacityCut, succ::Vector{Tuple{Int,Int}}, n::Int) =
    count(((a, b),) -> b <= n && cut.members[b] && !(a <= n && cut.members[a]), succ)

"""rho[l, w] for l in 1..n+1 (n+1 = depot) and customers w."""
function succession_duals!(rho::Matrix{Float64}, cuts::Vector{CapacityCut}, σ::Vector{Float64}, n::Int)
    fill!(rho, 0.0)
    for (cut, s) in zip(cuts, σ)
        s > 1e-12 || continue
        for w in 1:n
            cut.members[w] || continue
            for l in 1:n+1
                (l <= n && cut.members[l]) && continue
                rho[l, w] += s
            end
        end
    end
    rho
end

"""
rho as above plus the duals mu of edge-branching rows y_ij + y_ji >= 1,
earned on both successions i -> j and j -> i (both customers).
"""
function succession_duals!(rho::Matrix{Float64}, cuts::Vector{CapacityCut}, σ::Vector{Float64}, n::Int,
                           edges::Vector{Tuple{Int,Int}}, μ::Vector{Float64})
    succession_duals!(rho, cuts, σ, n)
    for ((i, j), m) in zip(edges, μ)
        m > 1e-12 || continue
        rho[i, j] += m
        rho[j, i] += m
    end
    rho
end

"""
slack[a, b] = min(0, min_w rho[a, w] - rho[b, w]): a label whose last
customer is `a` dominates one with last customer `b` only if its cost is lower
by at least -slack[a, b] (w ranges over customers; the depot earns nothing).
"""
function cut_slack!(slack::Matrix{Float64}, rho::Matrix{Float64}, n::Int)
    @inbounds for a in 1:n+1, b in 1:n+1
        m = 0.0
        if a != b
            for w in 1:n
                d = rho[a, w] - rho[b, w]
                d < m && (m = d)
            end
        end
        slack[a, b] = m
    end
    slack
end

"""
Separate violated rounded capacity cuts from LP succession flows: connected
components of the support graph, then greedy growth from every customer.
"""
function separate_capacity_cuts(inst::Instance, y::Dict{Tuple{Int,Int},Float64}, known::Set{BitVector};
                                max_cuts::Int = 50, min_violation::Float64 = 0.05)
    n = inst.n
    x = zeros(n + 1, n + 1)   # undirected flow, depot = n+1
    for ((a, b), v) in y
        x[a, b] += v
        x[b, a] += v
    end
    found = Tuple{Float64,BitVector}[]
    seen = Set{BitVector}()
    function consider!(S::BitVector)
        (S in seen || S in known) && return
        push!(seen, S)
        d = sum(inst.demand[i] for i in 1:n if S[i]; init = 0.0)
        r = ceil(d / (inst.payload + EPS) - 1e-9)
        crossing = 0.0
        for i in 1:n
            S[i] || continue
            for j in 1:n+1
                (j <= n && S[j]) && continue
                crossing += x[i, j]
            end
        end
        violation = r - crossing / 2
        violation >= min_violation && push!(found, (violation, copy(S)))
    end
    # Components of the customer support graph.
    label = zeros(Int, n)
    for start in 1:n
        label[start] == 0 || continue
        stack = [start]
        label[start] = start
        while !isempty(stack)
            i = pop!(stack)
            for j in 1:n
                if label[j] == 0 && x[i, j] > 1e-6
                    label[j] = start
                    push!(stack, j)
                end
            end
        end
    end
    for c in unique(label)
        consider!(BitVector(label .== c))
    end
    # Greedy growth: repeatedly add the customer most connected to S.
    for seed in 1:n
        S = falses(n)
        S[seed] = true
        link = [x[seed, j] for j in 1:n]
        for _ in 2:min(n - 1, 40)
            best, pick = 1e-6, 0
            for j in 1:n
                if !S[j] && link[j] > best
                    best, pick = link[j], j
                end
            end
            pick == 0 && break
            S[pick] = true
            for j in 1:n
                link[j] += x[pick, j]
            end
            consider!(copy(S))
        end
    end
    sort!(found; by = f -> -f[1])
    [CapacityCut(S, ceil(sum(inst.demand[i] for i in 1:n if S[i]) / (inst.payload + EPS) - 1e-9))
     for (_, S) in found[1:min(max_cuts, length(found))]]
end

# Limited-memory subset-row cuts on customer triples (rank-1, multipliers
# 1/2; upstream's lm-SR3, after Pecin et al. 2017). Scanning a route, the
# parity for triple T toggles at each visit to T and is forgotten at any
# customer outside the cut's memory M (T is inside M); a penalty is earned at
# every return to odd -> even. The coefficient is therefore the sum over
# memory segments of floor(visits to T / 2), never more than floor(|p cap T|/2),
# so
#   sum_p alpha_p z_p <= 1
# stays valid. The count splits only at forgetting customers, so it is the
# same read forwards or backwards. The dual tau_T <= 0 is a penalty |tau_T|
# per earned unit, tracked by one parity bit per cut (see pricing.jl).

const MAX_SUBSET_ROW = 256

"""Parity bits of up to 256 subset-row cuts (cut c is bit c-1 of lo, then hi)."""
struct SRBits
    lo::UInt128
    hi::UInt128
end

Base.zero(::Type{SRBits}) = SRBits(zero(UInt128), zero(UInt128))
@inline Base.:&(a::SRBits, b::SRBits) = SRBits(a.lo & b.lo, a.hi & b.hi)
@inline Base.:|(a::SRBits, b::SRBits) = SRBits(a.lo | b.lo, a.hi | b.hi)
@inline Base.xor(a::SRBits, b::SRBits) = SRBits(a.lo ⊻ b.lo, a.hi ⊻ b.hi)
@inline Base.:~(a::SRBits) = SRBits(~a.lo, ~a.hi)
@inline sr_bit(c::Int) = c <= 128 ? SRBits(one(UInt128) << (c - 1), zero(UInt128)) :
                                    SRBits(zero(UInt128), one(UInt128) << (c - 129))

struct SubsetRowCut
    members::NTuple{3,Int}
    memory::BitVector       # customers that keep the parity (members included)
end

function sr_coefficient(cut::SubsetRowCut, customers::Vector{Int})
    odd, total = false, 0
    for c in customers
        if c in cut.members
            odd && (total += 1)
            odd = !odd
        elseif !cut.memory[c]
            odd = false
        end
    end
    total
end

"""
Smallest memory keeping the cut's full coefficient on the given support
routes: every customer between two consecutive visits to the triple.
"""
function cut_memory(T::NTuple{3,Int}, routes, n::Int)
    memory = falses(n)
    for t in T
        memory[t] = true
    end
    for r in routes
        last = 0
        for (k, c) in enumerate(r.customers)
            c in T || continue
            if last > 0
                for m in last+1:k-1
                    memory[r.customers[m]] = true
                end
            end
            last = k
        end
    end
    memory
end

"""Most violated subset-row triples among customers sharing LP support columns."""
function separate_subset_row_cuts(inst::Instance, routes::Vector{Route}, z::Vector{Float64},
                                  known::Set{NTuple{3,Int}}; max_cuts::Int, min_violation::Float64)
    n = inst.n
    weight = Dict{Tuple{Int,Int},Float64}()
    support = [k for k in eachindex(z) if z[k] > 1e-6]
    for k in support
        cs = unique(routes[k].customers)
        for a in eachindex(cs), b in a+1:length(cs)
            i, j = minmax(cs[a], cs[b])
            weight[(i, j)] = get(weight, (i, j), 0.0) + z[k]
        end
    end
    neighbours = [Int[] for _ in 1:n]
    for ((i, j), w) in weight
        w > 1e-6 || continue
        push!(neighbours[i], j)
        push!(neighbours[j], i)
    end
    found = Tuple{Float64,NTuple{3,Int}}[]
    tried = Set{NTuple{3,Int}}()
    # A violated triple has positive weight on at least two of its pairs, so
    # one of its customers is a common neighbour of the other two.
    for c in 1:n
        nb = sort(neighbours[c])
        for a in eachindex(nb), b in a+1:length(nb)
            T = Tuple(sort!([c, nb[a], nb[b]]))
            (T in tried || T in known) && continue
            push!(tried, T)
            i, j, k = T
            w = get(weight, (i, j), 0.0) + get(weight, (i, k), 0.0) + get(weight, (j, k), 0.0)
            w > 1 + min_violation || continue   # the cut's left side never exceeds the pair weights
            lhs = sum(z[q] * (count(in(T), routes[q].customers) ÷ 2) for q in support)
            lhs - 1 >= min_violation && push!(found, (lhs - 1, T))
        end
    end
    sort!(found; by = f -> -f[1])
    map(found[1:min(max_cuts, length(found))]) do (_, T)
        hits = [routes[q] for q in support if count(in(T), routes[q].customers) >= 2]
        SubsetRowCut(T, cut_memory(T, hits, n))
    end
end
