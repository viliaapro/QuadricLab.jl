# ---------------------------------------------------------------------------
# Feasibility and boundedness of K = ⋂ᵢ {qᵢ ≤ 0}  (convex quadratic constraints).
#
# Feasibility:   minimise s   s.t.  q̃ᵢ(x) ≤ s  (all i),  s ≥ -1,
# where q̃ᵢ = qᵢ / scaleᵢ is each quadric normalised by the largest entry of its M.
# The bound s ≥ -1 only fixes the scale of the margin (the geometry itself is never
# bounded). Each q̃ᵢ(x) ≤ s with Aᵢ = FᵢᵀFᵢ is a rotated second-order-cone constraint
#     ‖Fᵢx‖² ≤ s - c - 2bᵀx   ⇔   ((u+1)/2, (u-1)/2, Fᵢx) ∈ SOC,   u = s - c - 2bᵀx.
# Solved with Clarabel (interior point). margin = -s*: > 0 interior, ≈ 0 boundary-only, < 0 empty.
#
# Boundedness (nonempty K): the recession cone of K is {d : Aᵢd = 0, bᵢᵀd ≤ 0 ∀i}; K is
# bounded iff that cone is {0}. Checked with small LPs over the nullspace of the stacked Aᵢ.
# ---------------------------------------------------------------------------

struct FeasibilityResult{T}
    status::Symbol                       # :interior | :boundary | :empty | :numerical_failure
    point::Union{Nothing,SVector{3,T}}   # strictly-best point found (nothing if :empty/failed)
    margin::T                            # -s*, in normalised units of q̃
    solver_status::String
    iterations::Int
    residual::T                          # max(primal, dual) residual reported by the solver
end

function Base.show(io::IO, r::FeasibilityResult)
    print(io, "FeasibilityResult(", r.status, ", margin=", round(r.margin; sigdigits = 4))
    r.point === nothing || print(io, ", point=", round.(r.point; sigdigits = 4))
    print(io, ", solver=", r.solver_status, ", iter=", r.iterations, ")")
end

_scale(q::Quadric) = max(maximum(abs, q.M), eps(eltype(q.M)))

# Rows (cone slack = b_c - A_c z) for  q̃(x) ≤ s  where x = z[xcols] and s = z[sidx]
# (sidx = 0: no slack variable, i.e. the constraint q̃(x) ≤ 0). Returns (A_c, b_c, cone).
# Used by feasibility (x = z[1:3], s = z[4]) and by the distance QCQP (x = z[1:3] or z[4:6]).
function _quadric_cone_rows(q::Quadric{T}, nvar::Int, sidx::Int; xcols = 1:3, relax::Real = 0.0) where {T}
    A, b, c = affine_part(q)
    sc = _scale(q)
    A = Matrix{Float64}(A) / sc; b = Vector{Float64}(b) / sc; c = Float64(c) / sc - relax   # relax: q̃(x) ≤ relax
    E = eigen(Symmetric(A))
    keep = findall(>(1e-12 * max(1.0, maximum(abs, E.values))), E.values)
    if isempty(keep)       # linear constraint: slack = s - c - 2bᵀx ≥ 0
        Ac = zeros(1, nvar); Ac[1, xcols] = 2b; sidx > 0 && (Ac[1, sidx] = -1)
        return Ac, [-c], Clarabel.NonnegativeConeT(1)
    end
    F = Diagonal(sqrt.(E.values[keep])) * E.vectors[:, keep]'   # A ≈ FᵀF, r×3
    r = length(keep)
    Ac = zeros(r + 2, nvar); bc = zeros(r + 2)
    Ac[1, xcols] = b; bc[1] = (1 - c) / 2                       # (u+1)/2
    Ac[2, xcols] = b; bc[2] = (-1 - c) / 2                      # (u-1)/2
    if sidx > 0
        Ac[1, sidx] = -0.5; Ac[2, sidx] = -0.5
    end
    Ac[3:end, xcols] = -F                                       # Fx
    return Ac, bc, Clarabel.SecondOrderConeT(r + 2)
end

_solved(sol) = uppercase(string(sol.status)) in ("SOLVED", "ALMOST_SOLVED")

# Solve with Clarabel. `tols` is tried in order (gap/feasibility tolerance). A fully SOLVED solution is returned as
# soon as one is found; ALMOST_SOLVED (only the loose "reduced" tolerances met — points can then be visibly off the
# surfaces) is kept as a fallback and returned only if no tolerance level reaches SOLVED. Otherwise the last attempt
# is returned so callers can inspect its status.
function _solve_conic(P, q, A, b, cones; tols = (1e-8,))
    sol = nothing; almost = nothing
    for t in tols
        settings = Clarabel.Settings(verbose = false, tol_gap_abs = t, tol_gap_rel = t, tol_feas = t)
        solver = Clarabel.Solver()
        Clarabel.setup!(solver, P, q, A, b, cones, settings)
        sol = Clarabel.solve!(solver)
        st = uppercase(string(sol.status))
        st == "SOLVED" && return sol
        st == "ALMOST_SOLVED" && almost === nothing && (almost = sol)
    end
    almost === nothing ? sol : almost
end

"""
    feasibility(obj; tol = 1e-7)

Is the intersection nonempty? Requires every constraint convex. Returns a
`FeasibilityResult`; `status` is `:interior` (nonempty interior, `point` is a margin-maximising
interior point), `:boundary` (nonempty but no interior, within `tol`), `:empty`, or
`:numerical_failure` (solver did not converge — not a mathematical statement).
"""
function feasibility(o::GeoObject{T}; tol::Real = 1e-7) where {T}
    is_convex(o) || throw(ArgumentError("feasibility requires every constraint to be convex (A ⪰ 0)"))
    _feasibility(o.constraints, T, tol)
end

# Same computation on a plain constraint list (ids need not be unique here).
function _feasibility(cs::AbstractVector{<:Constraint}, ::Type{T}, tol::Real) where {T}
    nvar = 4; sidx = 4
    blocks = Matrix{Float64}[]; rhs = Vector{Float64}[]; cones = Clarabel.SupportedCone[]
    for c in cs
        Ac, bc, cone = _quadric_cone_rows(c.quadric, nvar, sidx)
        push!(blocks, Ac); push!(rhs, bc); push!(cones, cone)
    end
    # s ≥ -1  ⇔  slack = s + 1 ≥ 0
    Ac = zeros(1, nvar); Ac[1, sidx] = -1
    push!(blocks, Ac); push!(rhs, [1.0]); push!(cones, Clarabel.NonnegativeConeT(1))
    A = sparse(reduce(vcat, blocks)); b = reduce(vcat, rhs)
    P = spzeros(nvar, nvar); qv = [0.0, 0.0, 0.0, 1.0]
    sol = _solve_conic(P, qv, A, b, cones)
    st = string(sol.status)
    resid = T(max(sol.r_prim, sol.r_dual))
    if !_solved(sol)
        return FeasibilityResult{T}(:numerical_failure, nothing, T(NaN), st, Int(sol.iterations), resid)
    end
    margin = T(-sol.x[4])
    pt = SVector{3,T}(sol.x[1], sol.x[2], sol.x[3])
    status = margin > tol ? :interior : (margin ≥ -tol ? :boundary : :empty)
    FeasibilityResult{T}(status, status === :empty ? nothing : pt, margin, st, Int(sol.iterations), resid)
end

feasibility(x::Union{Quadric,Constraint}; kwargs...) = feasibility(intersection(x); kwargs...)

"Is the intersection nonempty (interior or boundary)?"
is_feasible(o; kwargs...) = feasibility(o; kwargs...).status in (:interior, :boundary)

"""
    is_bounded(obj)

`true` iff the (nonempty) intersection is bounded, i.e. its recession cone
{d : Aᵢd = 0, bᵢ·d ≤ 0 ∀i} is trivial. An empty intersection is reported bounded.
Throws if feasibility could not be decided numerically.
"""
function is_bounded(o::GeoObject; tol::Real = 1e-9)
    f = feasibility(o)
    f.status === :numerical_failure && error("feasibility solve failed ($(f.solver_status)); boundedness undecided")
    f.status === :empty && return true
    As = reduce(vcat, Matrix{Float64}(affine_part(c)[1]) / _scale(c.quadric) for c in o.constraints)
    N = nullspace(As; atol = tol)                         # all d with Aᵢd = 0 ∀i, 3×k
    k = size(N, 2)
    k == 0 && return true
    B = reduce(vcat, (Vector{Float64}(affine_part(c)[2])' * N) / _scale(c.quadric) for c in o.constraints) # rows bᵢᵀN
    # cone {y : B y ≤ 0, ‖y‖∞ ≤ 1} nontrivial iff some ±y_j can be made positive
    for j in 1:k, sgn in (1.0, -1.0)
        qv = zeros(k); qv[j] = -sgn                        # minimise -sgn·y_j
        A = sparse(vcat(B, Matrix(1.0I, k, k), -Matrix(1.0I, k, k)))
        b = vcat(zeros(size(B, 1)), ones(k), ones(k))
        sol = _solve_conic(spzeros(k, k), qv, A, b, [Clarabel.NonnegativeConeT(size(A, 1))])
        _solved(sol) || error("boundedness LP failed ($(sol.status))")
        sol.x[j] * sgn > 1e-7 && return false
    end
    return true
end
is_bounded(x::Union{Quadric,Constraint}) = is_bounded(intersection(x))
