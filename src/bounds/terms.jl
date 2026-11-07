# ---------------------------------------------------------------------------
# Convex terms of a bounding Minkowski sum. Natural representations, no cross-encoding:
#
#   PolyhedronTerm  {x : Gx ≤ g, Ex = e}                    linear inequalities and affine equalities
#   QuadricTerm     {x : xᵀAx + 2bᵀx + c ≤ 0, Ex = e}       A ⪰ 0 (the affine-chart form of `Quadric`)
#
# QuadricTerm is deliberately general: balls, ellipsoids, flat disks (with equalities), cylinders, paraboloids,
# parabolic cylinders, and degenerate cases (A = 0: a halfspace or whole space; points). PolyhedronTerm is the natural
# form for linear constraints, but nothing forbids the degenerate quadric.
# Every term provides: ambient_dim, support, recession_cone, affine_hull, translate, ∈.
# ---------------------------------------------------------------------------

abstract type Term end

const _TOL = 1e-10

# ---- PolyhedronTerm -----------------------------------------------------------------------------

struct PolyhedronTerm <: Term
    n::Int
    G::Matrix{Float64}
    g::Vector{Float64}
    E::Matrix{Float64}
    e::Vector{Float64}
    function PolyhedronTerm(n::Integer, G::AbstractMatrix, g::AbstractVector, E::AbstractMatrix, e::AbstractVector)
        size(G, 1) == length(g) && size(E, 1) == length(e) || throw(DimensionMismatch("constraint rows and right-hand sides differ"))
        (size(G, 1) == 0 || size(G, 2) == n) && (size(E, 1) == 0 || size(E, 2) == n) || throw(DimensionMismatch("constraints must have $n columns"))
        all(isfinite, G) && all(isfinite, g) && all(isfinite, E) && all(isfinite, e) || throw(ArgumentError("constraint data must be finite"))
        new(Int(n), Matrix{Float64}(reshape(G, size(G, 1), n)), Vector{Float64}(g),
            Matrix{Float64}(reshape(E, size(E, 1), n)), Vector{Float64}(e))
    end
end

"""
    PolyhedronTerm(G, g; E, e)

`{x : Gx ≤ g, Ex = e}`. With no rows at all use `whole_space_term(n)`.
"""
function PolyhedronTerm(G::AbstractMatrix, g::AbstractVector; E::AbstractMatrix = zeros(0, size(G, 2)), e::AbstractVector = Float64[])
    PolyhedronTerm(size(G, 2), G, g, E, e)
end

"Axis-aligned box `lo ≤ x ≤ hi` (entries may be ±Inf; lo[i] == hi[i] becomes an affine equality)."
function box_term(lo::AbstractVector, hi::AbstractVector)
    n = length(lo); length(hi) == n || throw(DimensionMismatch("lo and hi differ in length"))
    G = Vector{Float64}[]; g = Float64[]; E = Vector{Float64}[]; e = Float64[]
    for i in 1:n
        lo[i] ≤ hi[i] || throw(ArgumentError("lo[$i] > hi[$i]: empty box"))
        ei = zeros(n); ei[i] = 1.0
        if lo[i] == hi[i]
            push!(E, ei); push!(e, lo[i])
        else
            isfinite(hi[i]) && (push!(G, ei); push!(g, hi[i]))
            isfinite(lo[i]) && (push!(G, -ei); push!(g, -lo[i]))
        end
    end
    PolyhedronTerm(n, isempty(G) ? zeros(0, n) : reduce(hcat, G)', g, isempty(E) ? zeros(0, n) : reduce(hcat, E)', e)
end
interval_term(lo::Real, hi::Real) = box_term([lo], [hi])

"Closed halfspace `normal·x ≤ offset` in ℝⁿ (natural linear form)."
halfspace_term(normal::AbstractVector, offset::Real) = PolyhedronTerm(Matrix(reshape(Float64.(normal), 1, :)), [Float64(offset)])

"All of ℝⁿ (a 'line' for n = 1, a plane for n = 2, …)."
whole_space_term(n::Integer) = PolyhedronTerm(n, zeros(0, n), Float64[], zeros(0, n), Float64[])

"Affine subspace `point + span(basis)` (basis columns; zero columns give a point)."
function affine_term(point::AbstractVector, basis::AbstractMatrix = zeros(length(point), 0))
    n = length(point)
    E = size(basis, 2) == 0 ? Matrix{Float64}(I, n, n) : Matrix(nullspace(Matrix{Float64}(basis)')')
    PolyhedronTerm(n, zeros(0, n), Float64[], E, E * Vector{Float64}(point))
end

ambient_dim(f::PolyhedronTerm) = f.n

# min c·x over the term. status ∈ :finite | :empty | :unbounded (ray: certificate with c·ray < 0) | :failed
function _lp(f::PolyhedronTerm, c::Vector{Float64})
    n = f.n; m, p = size(f.G, 1), size(f.E, 1)
    if m + p == 0
        return norm(c) ≤ 1e-14 ? (status = :finite, x = zeros(n), value = 0.0, ray = nothing, solver = "trivial", resid = 0.0) :
               (status = :unbounded, x = nothing, value = -Inf, ray = -c / norm(c), solver = "trivial", resid = 0.0)
    end
    cones = Clarabel.SupportedCone[]
    p > 0 && push!(cones, Clarabel.ZeroConeT(p)); m > 0 && push!(cones, Clarabel.NonnegativeConeT(m))
    sol = _solve_conic(spzeros(n, n), c, sparse(vcat(f.E, f.G)), vcat(f.e, f.g), cones; tols = (1e-10, 1e-8))
    st = uppercase(string(sol.status))
    resid = Float64(max(sol.r_prim, sol.r_dual))
    if _solved(sol)
        x = Vector{Float64}(sol.x); (status = :finite, x = x, value = dot(c, x), ray = nothing, solver = st, resid = resid)
    elseif occursin("PRIMAL_INFEASIBLE", st)
        (status = :empty, x = nothing, value = Inf, ray = nothing, solver = st, resid = resid)
    elseif occursin("DUAL_INFEASIBLE", st)
        r = Vector{Float64}(sol.x); nr = norm(r)
        (status = :unbounded, x = nothing, value = -Inf, ray = (isfinite(nr) && nr > 0) ? r / nr : nothing, solver = st, resid = resid)
    else
        (status = :failed, x = nothing, value = NaN, ray = nothing, solver = st, resid = resid)
    end
end

function support(f::PolyhedronTerm, u::AbstractVector)
    length(u) == f.n || throw(DimensionMismatch("direction has length $(length(u)), term lives in ℝ^$(f.n)"))
    u = Vector{Float64}(u)
    r = _lp(f, -u)
    r.status === :failed && return _sup_failed("LP $(r.solver)")
    r.status === :empty && return _sup_empty(solver = r.solver)
    if r.status === :unbounded
        ray = r.ray !== nothing && dot(u, r.ray) > 0 ? r.ray : nothing
        return _sup_infinite(ray; solver = r.solver)
    end
    x = r.x
    act = [i for i in 1:size(f.G, 1) if f.G[i, :]' * x ≥ f.g[i] - 1e-7 * (1 + abs(f.g[i]))]
    M = vcat(f.E, f.G[act, :])
    fd = isempty(M) ? f.n : f.n - rank(M; atol = 1e-7)
    SupportResult(:finite, dot(u, x), x, fd, nothing, r.solver, r.resid, act, SupportResult[])
end

recession_cone(f::PolyhedronTerm) = PolyhedronTerm(f.n, f.G, zeros(size(f.G, 1)), f.E, zeros(size(f.E, 1)))

function affine_hull(f::PolyhedronTerm)
    n = f.n; implicit = Int[]
    if size(f.G, 1) == 0 && size(f.E, 1) == 0
        return AffineHull(zeros(n), Matrix{Float64}(I, n, n))
    end
    feasible = false
    for i in 1:size(f.G, 1)                                  # row i is an implicit equality iff max slack g_i − G_i x is 0
        r = _lp(f, Vector{Float64}(f.G[i, :]))
        r.status === :empty && return nothing
        r.status === :failed && error("affine hull: LP failed ($(r.solver))")
        feasible = true
        r.status === :finite && f.g[i] - r.value ≤ 1e-8 * (1 + abs(f.g[i])) && push!(implicit, i)
    end
    if !feasible                                            # only equalities
        r = _lp(f, zeros(n)); r.status === :empty && return nothing
    end
    Eh = vcat(f.E, f.G[implicit, :]); eh = vcat(f.e, f.g[implicit])
    isempty(Eh) && return AffineHull(zeros(n), Matrix{Float64}(I, n, n))
    AffineHull(Vector{Float64}(Eh \ eh), Matrix{Float64}(nullspace(Eh; atol = 1e-9)))
end

translate(f::PolyhedronTerm, t::AbstractVector) = PolyhedronTerm(f.n, f.G, f.g + f.G * t, f.E, f.e + f.E * t)

"""
    transform(f, L, t = 0)

Image of a term under the invertible affine map x ↦ Lx + t (same ambient dimension).
"""
function transform(f::PolyhedronTerm, L::AbstractMatrix, t::AbstractVector = zeros(f.n))
    Li = _inv_checked(L, f.n)
    PolyhedronTerm(f.n, f.G * Li, f.g + f.G * Li * t, f.E * Li, f.e + f.E * Li * t)
end
function _inv_checked(L, n)
    size(L) == (n, n) || throw(DimensionMismatch("linear map must be $n×$n"))
    1 / cond(Matrix{Float64}(L)) > 1e-12 || throw(ArgumentError("term transforms need an invertible map (singular maps are not representable per term)"))
    inv(Matrix{Float64}(L))
end

Base.in(x::AbstractVector, f::PolyhedronTerm) =
    all(f.G * x .≤ f.g .+ 1e-9 .* (1 .+ abs.(f.g))) && all(abs.(f.E * x .- f.e) .≤ 1e-9 .* (1 .+ abs.(f.e)))

# ---- QuadricTerm ----------------------------------------------------------------------------------

struct QuadricTerm <: Term
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Float64
    E::Matrix{Float64}
    e::Vector{Float64}
    function QuadricTerm(A::AbstractMatrix, b::AbstractVector, c::Real; E::AbstractMatrix = zeros(0, length(b)), e::AbstractVector = Float64[])
        n = length(b)
        size(A) == (n, n) || throw(DimensionMismatch("A must be $n×$n"))
        size(E, 1) == length(e) && (size(E, 1) == 0 || size(E, 2) == n) || throw(DimensionMismatch("equalities must have $n columns"))
        all(isfinite, A) && all(isfinite, b) && isfinite(c) && all(isfinite, E) && all(isfinite, e) || throw(ArgumentError("data must be finite"))
        sc = max(maximum(abs, A), 1.0)
        maximum(abs, A - A') ≤ 1e-10 * sc || throw(ArgumentError("A must be symmetric"))
        As = Matrix{Float64}((A + A') / 2)
        eigmin(Symmetric(As)) ≥ -1e-10 * sc || throw(ArgumentError("A must be positive semidefinite (convex quadric)"))
        new(As, Vector{Float64}(b), Float64(c), Matrix{Float64}(reshape(E, size(E, 1), n)), Vector{Float64}(e))
    end
end

"Reuse an existing 3-D `Quadric` (affine chart A, b, c) as a bounding term."
function QuadricTerm(q::Quadric)
    A, b, c = affine_part(q)
    QuadricTerm(Matrix(A), Vector(b), c)
end

"Ball `‖x − center‖ ≤ radius` in ℝⁿ (radius 0 is a point)."
function ball_term(center::AbstractVector, radius::Real)
    radius ≥ 0 || throw(ArgumentError("radius must be nonnegative"))
    n = length(center); c = Vector{Float64}(center)
    QuadricTerm(Matrix{Float64}(I, n, n), -c, dot(c, c) - radius^2)
end

"""
    ellipsoid_term(center, shape)

`{center + Σ^{1/2}y : ‖y‖ ≤ 1}` for a PSD shape matrix Σ. A singular Σ gives a lower-dimensional ellipsoid
(a disk, a segment…), expressed with affine equalities on the null space of Σ.
"""
function ellipsoid_term(center::AbstractVector, shape::AbstractMatrix)
    n = length(center); c = Vector{Float64}(center)
    F = eigen(Symmetric(Matrix{Float64}(shape))); λ = F.values; sc = max(1.0, maximum(abs, λ))
    λmin = minimum(λ); λmin ≥ -1e-10 * sc || throw(ArgumentError("shape matrix must be positive semidefinite"))
    P = findall(>(1e-10 * sc), λ); Z = setdiff(1:n, P)
    if isempty(P)                                                  # zero shape: the point {center}
        return QuadricTerm(zeros(n, n), zeros(n), 0.0; E = Matrix{Float64}(I, n, n), e = c)
    end
    Vp = F.vectors[:, P]; Sp = Vp * Diagonal(1 ./ λ[P]) * Vp'
    E = isempty(Z) ? zeros(0, n) : Matrix(F.vectors[:, Z]')
    QuadricTerm(Sp, -Sp * c, dot(c, Sp * c) - 1; E = E, e = E * c)
end

ambient_dim(f::QuadricTerm) = length(f.b)

"""
    embed_term(f, indices, N)

A convenience: the term `f ⊂ ℝⁿ` placed on coordinates `indices` of ℝᴺ (the other coordinates are fixed to 0 by affine
equalities), e.g. a disk and a line on separate coordinates giving a cylinder as their sum. Products do not need it or track it.
"""
function embed_term(f::PolyhedronTerm, idx::AbstractVector{<:Integer}, N::Integer)
    _check_embed(f.n, idx, N)
    G = zeros(size(f.G, 1), N); G[:, idx] = f.G; E = zeros(size(f.E, 1), N); E[:, idx] = f.E
    out = setdiff(1:N, idx); Z = zeros(length(out), N); for (k, j) in enumerate(out); Z[k, j] = 1.0; end
    PolyhedronTerm(N, G, f.g, vcat(E, Z), vcat(f.e, zeros(length(out))))
end
function embed_term(f::QuadricTerm, idx::AbstractVector{<:Integer}, N::Integer)
    _check_embed(ambient_dim(f), idx, N)
    A = zeros(N, N); A[idx, idx] = f.A; b = zeros(N); b[idx] = f.b
    E = zeros(size(f.E, 1), N); E[:, idx] = f.E
    out = setdiff(1:N, idx); Z = zeros(length(out), N); for (k, j) in enumerate(out); Z[k, j] = 1.0; end
    QuadricTerm(A, b, f.c; E = vcat(E, Z), e = vcat(f.e, zeros(length(out))))
end
function _check_embed(n, idx, N)
    length(idx) == n || throw(DimensionMismatch("term has dimension $n but $(length(idx)) indices were given"))
    (allunique(idx) && all(i -> 1 ≤ i ≤ N, idx)) || throw(ArgumentError("indices must be distinct and lie in 1:$N"))
end

# Restrict to the equality subspace: x = x0 + N y. Returns nothing if the equalities are inconsistent,
# else (x0, N, A', b', c') describing the quadric in y.
function _restrict(f::QuadricTerm)
    n = ambient_dim(f)
    if size(f.E, 1) == 0
        return (zeros(n), Matrix{Float64}(I, n, n), f.A, f.b, f.c)
    end
    x0 = Vector{Float64}(f.E \ f.e)
    norm(f.E * x0 - f.e) ≤ 1e-8 * (1 + norm(f.e)) || return nothing
    N = Matrix{Float64}(nullspace(f.E; atol = 1e-10))
    (x0, N, N' * f.A * N, N' * (f.A * x0 + f.b), dot(x0, f.A * x0) + 2dot(f.b, x0) + f.c)
end

# Analytic support of {y : yᵀAy + 2bᵀy + c ≤ 0}, A ⪰ 0, in direction u (any dimension, A may be singular or 0).
# Returns (:finite, h, y, face_dim) | (:infinite, ray_or_nothing) | (:empty,).
function _quad_support(A, b, c, u)
    d = length(b); nu = max(norm(u), 1e-300)
    F = eigen(Symmetric(A)); λ = F.values; V = F.vectors
    sc = max(1.0, maximum(abs, λ)); P = findall(>(1e-10 * sc), λ); Z = setdiff(1:d, P)
    β = V' * b; ũ = V' * u; bs = max(1.0, norm(b), abs(c))
    if isempty(P)                                                  # restricted set is linear: {2bᵀy + c ≤ 0}
        nb = norm(b)
        if nb ≤ 1e-12 * bs
            c > 1e-10 * bs && return (:empty,)
            return nu ≤ 1e-14 ? (:finite, 0.0, zeros(d), d) : (:infinite, u / nu)
        end
        β̂ = b / nb; τ = dot(u, β̂); perp = u - τ * β̂
        norm(perp) > 1e-10 * nu && return (:infinite, perp / norm(perp))
        τ < -1e-10 * nu && return (:infinite, -β̂)
        y = (-c / (2nb)) * β̂
        return (:finite, max(τ, 0.0) * (-c / (2nb)), y, τ > 1e-10 * nu ? d - 1 : d)
    end
    βZ = β[Z]; βP = β[P]; λP = λ[P]; ũP = ũ[P]; ũZ = ũ[Z]
    if isempty(Z) || norm(βZ) ≤ 1e-10 * bs                         # ellipsoid ⊕ free null directions (cylinder type)
        ρ = sum(βP .^ 2 ./ λP) - c
        ρ < -1e-9 * max(bs, sum(βP .^ 2 ./ λP)) && return (:empty,)
        ρ = max(ρ, 0.0)
        norm(ũZ) > 1e-10 * nu && return (:infinite, V[:, Z] * ũZ / norm(ũZ))
        cP = -βP ./ λP; q = sum(ũP .^ 2 ./ λP)
        q ≤ 1e-28 && return (:finite, 0.0, V[:, P] * cP, d)
        yP = cP + sqrt(ρ / q) * (ũP ./ λP)
        return (:finite, dot(ũP, cP) + sqrt(ρ * q), V[:, P] * yP, length(Z))
    end
    nbz = norm(βZ); β̂ = βZ / nbz                                    # paraboloid type: one null direction is bounded above
    τ = dot(ũZ, β̂); perp = ũZ - τ * β̂
    norm(perp) > 1e-10 * nu && return (:infinite, V[:, Z] * perp / norm(perp))
    τ < -1e-10 * nu && return (:infinite, -V[:, Z] * β̂)
    if τ ≤ 1e-10 * nu                                              # τ = 0: sup is +∞ unless u = 0 (the barrier cone is not closed)
        norm(ũP) ≤ 1e-12 * nu && return (:finite, 0.0, V[:, Z] * (min(0.0, -c / (2nbz)) * β̂), d)   # u = 0: any feasible point
        return (:infinite, nothing)
    end
    κ = nbz / τ
    yP = (κ * ũP - βP) ./ λP
    s = -(sum(λP .* yP .^ 2) + 2dot(βP, yP) + c) / (2nbz)
    (:finite, dot(ũP, yP) + τ * s, V[:, P] * yP + V[:, Z] * (s * β̂), length(Z) - 1)
end

function support(f::QuadricTerm, u::AbstractVector)
    length(u) == ambient_dim(f) || throw(DimensionMismatch("direction has length $(length(u)), term lives in ℝ^$(ambient_dim(f))"))
    u = Vector{Float64}(u)
    r = _restrict(f)
    r === nothing && return _sup_empty()
    x0, N, Ap, bp, cp = r
    if size(N, 2) == 0                                             # a single point x0
        return cp ≤ 1e-9 * max(1.0, abs(cp), norm(bp)) ? _sup_finite(dot(u, x0), x0, 0) : _sup_empty()
    end
    res = _quad_support(Ap, bp, cp, N' * u)
    res[1] === :empty && return _sup_empty()
    if res[1] === :infinite
        ray = res[2] === nothing ? nothing : (v = N * res[2]; dot(u, v) > 0 ? v / norm(v) : nothing)
        return _sup_infinite(ray)
    end
    _, h, y, fd = res
    _sup_finite(dot(u, x0) + h, x0 + N * y, fd)
end

function recession_cone(f::QuadricTerm)                            # {d : Ed = 0, Ad = 0, bᵀd ≤ 0}
    n = ambient_dim(f)
    rowsE = vcat(f.E, f.A); keep = [i for i in 1:size(rowsE, 1) if norm(rowsE[i, :]) > 1e-12]
    G = norm(f.b) > 1e-12 ? reshape(f.b, 1, :) : zeros(0, n)
    PolyhedronTerm(n, G, zeros(size(G, 1)), rowsE[keep, :], zeros(length(keep)))
end

function affine_hull(f::QuadricTerm)
    r = _restrict(f); r === nothing && return nothing
    x0, N, A, b, c = r; d = size(N, 2); n = ambient_dim(f)
    if d == 0
        return c ≤ 1e-9 * max(1.0, abs(c), norm(b)) ? AffineHull(x0, zeros(n, 0)) : nothing
    end
    F = eigen(Symmetric(A)); λ = F.values; V = F.vectors
    sc = max(1.0, maximum(abs, λ)); P = findall(>(1e-10 * sc), λ); Z = setdiff(1:d, P)
    β = V' * b; bs = max(1.0, norm(b), abs(c))
    full = AffineHull(x0, N)
    isempty(P) && return (norm(b) ≤ 1e-12 * bs && c > 1e-10 * bs) ? nothing : full
    (!isempty(Z) && norm(β[Z]) > 1e-10 * bs) && return full         # unbounded below: has interior
    qmin = c - sum(β[P] .^ 2 ./ λ[P])
    tol = 1e-9 * max(bs, sum(β[P] .^ 2 ./ λ[P]))
    qmin > tol && return nothing
    qmin < -tol && return full
    AffineHull(x0 + N * (V[:, P] * (-β[P] ./ λ[P])), N * V[:, Z])  # q ≤ 0 only on its minimiser set
end

function translate(f::QuadricTerm, t::AbstractVector)             # q(x − t) ≤ 0
    A = f.A; b = f.b - A * t; c = dot(t, A * t) - 2dot(f.b, t) + f.c
    QuadricTerm(A, b, c; E = f.E, e = f.e + f.E * t)
end

function transform(f::QuadricTerm, L::AbstractMatrix, t::AbstractVector = zeros(ambient_dim(f)))
    Li = _inv_checked(L, ambient_dim(f))
    A2 = Li' * f.A * Li; A2 = (A2 + A2') / 2
    QuadricTerm(A2, Li' * f.b - A2 * t, dot(t, A2 * t) - 2dot(f.b, Li * t) + f.c; E = f.E * Li, e = f.e + f.E * Li * t)
end

function Base.in(x::AbstractVector, f::QuadricTerm)
    q = dot(x, f.A * x) + 2dot(f.b, x) + f.c
    q ≤ 1e-9 * max(1.0, abs(f.c), norm(f.b) * norm(x), norm(x)^2) && all(abs.(f.E * x .- f.e) .≤ 1e-9 .* (1 .+ abs.(f.e)))
end

Base.show(io::IO, f::PolyhedronTerm) = print(io, "PolyhedronTerm(ℝ^", f.n, ", ", size(f.G, 1), " inequalities, ", size(f.E, 1), " equalities)")
Base.show(io::IO, f::QuadricTerm) = print(io, "QuadricTerm(ℝ^", ambient_dim(f), ", rank(A)=", rank(f.A; atol = 1e-9), ", ", size(f.E, 1), " equalities)")
