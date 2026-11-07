# ---------------------------------------------------------------------------
# Exact support function of a (convex) GeoObject and a containment check against a bound.
#
# h_K(u) = max u·x over K = ⋂ {qᵢ ≤ 0}: one SOCP in three variables (the rows of feasibility.jl). Statuses as for every term:
# :finite, :infinite (certificate ray when Clarabel gives one), :empty, :failed. `face_dim` is not determined here (0).
# A bound Minkowski sum B contains the closed convex object K iff h_K(u) ≤ h_B(u) for every u, so `bound_violation` samples
# directions: a positive result PROVES the bound does not contain the object; a non-positive one is evidence (not proof) of
# containment on the sampled directions — the recovery guarantee assumes the supplied bounds really are outer bounds.
# ---------------------------------------------------------------------------

function support(o::GeoObject, u::AbstractVector)
    length(u) == 3 || throw(DimensionMismatch("a GeoObject lives in ℝ³"))
    is_convex(o) || throw(ArgumentError("exact support needs convex constraints (A ⪰ 0)"))
    u = Vector{Float64}(u)
    blocks = Matrix{Float64}[]; rhs = Vector{Float64}[]; cones = Clarabel.SupportedCone[]
    for c in o.constraints
        Ac, bc, cone = _quadric_cone_rows(c.quadric, 3, 0; xcols = 1:3)
        push!(blocks, Ac); push!(rhs, bc); push!(cones, cone)
    end
    sol = _solve_conic(spzeros(3, 3), -u, sparse(reduce(vcat, blocks)), reduce(vcat, rhs), cones; tols = (1e-10, 1e-8))
    st = uppercase(string(sol.status))
    if _solved(sol)
        x = Vector{Float64}(sol.x); return _sup_finite(dot(u, x), x, 0; solver = st, resid = Float64(max(sol.r_prim, sol.r_dual)))
    elseif occursin("PRIMAL_INFEASIBLE", st)
        return _sup_empty(solver = st)
    elseif occursin("DUAL_INFEASIBLE", st)
        r = Vector{Float64}(sol.x); nr = norm(r)
        return _sup_infinite((isfinite(nr) && nr > 0 && dot(u, r) > 0) ? r / nr : nothing; solver = st)
    end
    _sup_failed(st)
end

"""
    bound_violation(exact, bound; directions = 300)

max over sampled directions u of h_exact(u) − h_bound(u). Positive ⇒ the bound does NOT contain the object (proof); ≤ ~1e-6 ⇒
no violation found on the samples. Returns `(violation, direction)`.
"""
function bound_violation(exact::GeoObject, bound::MinkowskiSum; directions::Integer = 300)
    ambient_dim(bound) == 3 || throw(DimensionMismatch("bound must live in ℝ³"))
    dirs = Vector{Vector{Float64}}(_fallback_seeds(3))
    for i in 0:directions-1                                              # Fibonacci sphere
        z = 1 - 2 * (i + 0.5) / directions; ρ = sqrt(1 - z^2); φ = i * π * (3 - sqrt(5)); push!(dirs, [ρ * cos(φ), ρ * sin(φ), z])
    end
    worst = -Inf; wd = dirs[1]
    for u in dirs
        he, hb = support(exact, u), support(bound, u)
        he.status === :empty && continue
        v = he.status === :infinite ? (hb.status === :infinite ? -Inf : Inf) : (hb.status === :infinite ? -Inf : he.value - hb.value)
        isnan(v) && continue
        v > worst && (worst = v; wd = u)
    end
    (violation = worst, direction = wd)
end
