# Pricing self-tests that need no Gurobi licence:
#   julia --project=. test/runtests.jl
# The end-to-end checks (brute force, environment replay) live in the Python
# test suite and scripts/analysis/validate_branch_and_price.py.

using EVRPBranchPrice
using Test

const B = EVRPBranchPrice
const FIXTURE = joinpath(@__DIR__, "fixtures", "k3_n10_s5.json")

function reduced_cost(inst, r, π, cuts, τ)
    r.travel - sum(π[c] for c in r.customers; init = 0.0) +
    sum(-τ[k] * B.sr_coefficient(cuts[k], r.customers) for k in eachindex(cuts); init = 0.0)
end

@testset "EVRPBranchPrice pricing" begin
    inst = B.read_instance(FIXTURE)
    W = B.words_for(inst.n + 1)
    ws = B.PricingWorkspace{W}(inst; ng_size = 8, unit = 0.05, near = 25)
    bs = B.BackwardSpace{W}(inst, ws.R)
    restr = B.Restrictions{W}(inst, Tuple{Int,Int}[])
    π = fill(6.0, inst.n)
    bidirectional(τ, cuts; share = 0.5) = B.price_bidirectional!(ws, bs, inst, π, 0.0, restr; exact = true,
        max_columns = 200, rc_tol = 1e-6, time_limit = 60.0, label_limit = typemax(Int), bucket_cap = 64,
        cost_scale = 1.0, cuts = B.CapacityCut[], σ = Float64[], sr_cuts = cuts, τ = τ, forward_share = share)
    monodirectional(τ, cuts) = B.price!(ws, inst, π, 0.0, restr; exact = true, timefree = true, untimed = true,
        sr_cuts = cuts, τ = τ)

    # Cut 200 lives in the second 128-bit word of the parity state.
    cuts = [B.SubsetRowCut((1, 2, 3), trues(inst.n)) for _ in 1:200]
    cuts[200] = B.SubsetRowCut((4, 5, 6), trues(inst.n))
    for τ in (zeros(200), (t = zeros(200); t[1] = -2.0; t[200] = -3.0; t))
        mono, bi = monodirectional(τ, cuts), bidirectional(τ, cuts)
        @test mono.exact && bi.exact
        @test isapprox(mono.min_rc, bi.min_rc; atol = 1e-8)
        for res in (mono, bi), (r, rc) in zip(res.routes, res.rcs)
            @test isapprox(rc, reduced_cost(inst, r, π, cuts, τ); atol = 1e-8)
        end
        # any split of the payload between the two directions is exact
        for share in (0.3, 0.65, 0.9)
            @test isapprox(bidirectional(τ, cuts; share = share).min_rc, mono.min_rc; atol = 1e-8)
        end
    end

    # Limited memory: forgetting outside the memory can only lower a coefficient.
    full = B.SubsetRowCut((1, 2, 3), trues(inst.n))
    narrow = B.SubsetRowCut((1, 2, 3), BitVector(i in (1, 2, 3) for i in 1:inst.n))
    @test B.sr_coefficient(full, [1, 4, 2, 5, 3]) == 1
    @test B.sr_coefficient(narrow, [1, 4, 2, 5, 3]) == 0
    @test B.sr_coefficient(narrow, [1, 2, 4, 3]) == 1

    # Backward-label completion bounds only prune: same minimum either way.
    for τ in (zeros(200), (t = zeros(200); t[1] = -2.0; t[200] = -3.0; t))
        plain = B.price_bidirectional!(ws, bs, inst, π, 0.0, restr; exact = true, max_columns = 200, rc_tol = 1e-6,
            time_limit = 60.0, label_limit = typemax(Int), bucket_cap = 64, cost_scale = 1.0,
            cuts = B.CapacityCut[], σ = Float64[], sr_cuts = cuts, τ = τ, suffix_bounds = false)
        @test isapprox(bidirectional(τ, cuts).min_rc, plain.min_rc; atol = 1e-8)
    end

    # Edge-branching rows: their dual is earned on both orientations.
    let edges = [(1, 2), (3, 7)], μ = [1.5, 0.75], τ = zeros(200)
        mono = B.price!(ws, inst, π, 0.0, restr; exact = true, timefree = true, untimed = true, sr_cuts = cuts,
                        τ = τ, edges = edges, μ = μ)
        bi = B.price_bidirectional!(ws, bs, inst, π, 0.0, restr; exact = true, max_columns = 200, rc_tol = 1e-6,
            time_limit = 60.0, label_limit = typemax(Int), bucket_cap = 64, cost_scale = 1.0,
            cuts = B.CapacityCut[], σ = Float64[], sr_cuts = cuts, τ = τ, edges = edges, μ = μ)
        @test isapprox(mono.min_rc, bi.min_rc; atol = 1e-8)
        for res in (mono, bi), (r, rc) in zip(res.routes, res.rcs)
            succ = B.successions(inst, r)
            earned = sum(m * B.edge_entries(e, succ) for (e, m) in zip(edges, μ))
            @test isapprox(rc, reduced_cost(inst, r, π, cuts, τ) - earned; atol = 1e-8)
        end
    end

    # Enumeration pruned by backward-label bounds lists the same routes.
    for θ in (0.5, 3.0), τ in (zeros(200), (t = zeros(200); t[1] = -2.0; t; ))
        kw = (threshold = θ, cuts = B.CapacityCut[], σ = Float64[], sr_cuts = cuts, τ = τ, time_limit = 120.0,
              label_limit = typemax(Int), route_cap = typemax(Int))
        plain = B.enumerate_routes!(ws, inst, π, 0.0, restr; timefree = true, untimed = true, kw...)
        listed(res) = sort([(round(rc; digits = 8), r.customers) for (r, rc) in zip(res.routes, res.rcs)])
        for share in (0.5, 1.0)
            bounded = B.enumerate_routes_bounded!(ws, bs, inst, π, 0.0, restr; share = share, kw...)
            @test plain.exact && bounded.exact
            @test unique(listed(bounded)) == unique(listed(plain))
        end
    end

    # Every priced route re-timed exactly is accepted by the nominal simulator.
    for r in bidirectional(zeros(200), cuts).routes
        scheduled = B.schedule_route(inst, r)
        scheduled === nothing || @test B.simulate(inst, scheduled)[1]
    end
end
