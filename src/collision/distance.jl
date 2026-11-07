# ---------------------------------------------------------------------------
# Distance between two convex quadratic objects A, B (each a GeoObject).
#
#   minimise ½‖x − y‖²   s.t.  qᵢ(x) ≤ 0 (i ∈ A),  qⱼ(y) ≤ 0 (j ∈ B)
#
# a convex QCQP in the six variables (x, y), solved as a second-order-cone program with
# Clarabel (P = [I −I; −I I]; each quadric is the rotated SOC of feasibility.jl).
#
# Overlap is decided first (feasibility of A ∩ B): interiors overlapping ⇒ :intersecting,
# distance 0 (penetration is a different operation; it is NOT a negative distance); only
# boundaries meeting ⇒ :touching; empty ⇒ :separated and the QCQP is solved.
#
# Separation normal n = (y − x)/‖y − x‖ points from A to B. KKT: n ∈ N_A(x) and −n ∈ N_B(y),
# which is verified by non-negative least squares on the active constraint gradients
# (`cone_residual`). For touching objects the normal is any n with n ∈ N_A(p), −n ∈ N_B(p),
# found from an LP over the active gradients.
#
# Witness uniqueness: the optimal x form A ∩ (B + (x*−y*)); its extent is measured at two relaxation levels
# (see _witness_extent): shrinking like √ε ⇒ :unique (point contact), constant ⇒ :nonunique (finite face),
# unbounded ⇒ :unbounded (parallel planes, cylinder lying on a plane); :unknown if the auxiliary solves fail.
# ---------------------------------------------------------------------------

struct DistanceResult{T}
    status::Symbol                          # :separated | :touching | :intersecting | :empty_object | :numerical_failure
    distance::T                             # ≥ 0; NaN when status is :empty_object / :numerical_failure
    point_a::Union{Nothing,SVector{3,T}}    # closest point of A (intersecting: a most-interior point of A ∩ B)
    point_b::Union{Nothing,SVector{3,T}}
    normal::Union{Nothing,SVector{3,T}}     # unit, from A toward B; nothing if intersecting / undetermined
    active_a::Vector{Int}                   # ids of constraints of A active at point_a
    active_b::Vector{Int}
    cone_a::Union{Nothing,NormalCone{T}}
    cone_b::Union{Nothing,NormalCone{T}}
    cone_residual::T                        # max residual of ±normal against N_A, N_B (≈0: optimality certified)
    witness_set::Symbol                     # :unique | :nonunique | :unbounded | :not_checked | :not_applicable
    extent::T                               # size of the (slightly relaxed) optimal witness set; Inf if unbounded
    solver_status::String
    iterations::Int
    residual::T                             # solver primal/dual residual
end

# ---- helpers -----------------------------------------------------------------

_as_geo(x::GeoObject) = x
_as_geo(x::Union{Quadric,Constraint}) = intersection(x)

function _pair(A, B)
    a = _as_geo(A); b = _as_geo(B)
    T = promote_type(eltype(a.constraints[1].quadric.M), eltype(b.constraints[1].quadric.M))
    GeoObject{T}([_cast(T, c) for c in a.constraints]), GeoObject{T}([_cast(T, c) for c in b.constraints]), T
end

function _failed(::Type{T}, status::Symbol, solver_status::AbstractString, iters::Integer = 0, resid = NaN) where {T}
    DistanceResult{T}(status, T(NaN), nothing, nothing, nothing, Int[], Int[], nothing, nothing, T(NaN),
                      :not_applicable, T(NaN), String(solver_status), Int(iters), T(resid))
end

# conic constraint blocks for  x ∈ A (columns xc),  y ∈ B (columns yc)
function _object_blocks!(blocks, rhs, cones, o::GeoObject, nvar, xc)
    for c in o.constraints
        Ac, bc, cone = _quadric_cone_rows(c.quadric, nvar, 0; xcols = xc)
        push!(blocks, Ac); push!(rhs, bc); push!(cones, cone)
    end
end

# ---- the QCQP ----------------------------------------------------------------

function _solve_distance(A::GeoObject, B::GeoObject)
    blocks = Matrix{Float64}[]; rhs = Vector{Float64}[]; cones = Clarabel.SupportedCone[]
    _object_blocks!(blocks, rhs, cones, A, 6, 1:3)
    _object_blocks!(blocks, rhs, cones, B, 6, 4:6)
    Id = Matrix(1.0I, 3, 3)
    P = sparse(triu([Id -Id; -Id Id]))
    _solve_conic(P, zeros(6), sparse(reduce(vcat, blocks)), reduce(vcat, rhs), cones; tols = (1e-12, 1e-10, 1e-8))
end

# ---- witness-set exploration ---------------------------------------------------

# Optimal witnesses x form the set A ∩ (B + w) with w = x* − y* (w is unique because the objective is
# strictly convex in x − y), i.e. a *touching* problem in one point variable. Its size is measured by
# min/maximising each coordinate over the constraints of A and of the translated B, each relaxed by ε in
# distance-like units (qᵢ ≤ ε·max(‖∇qᵢ(p0)‖, 1), p0 = a known witness), so the set has a thin genuine
# interior and the conic solver is well posed.
#
# Scale-free decision: measure the extent at ε₁ and at ε₂ = 10ε₁. A point contact (curved tangency) has relaxed
# size ∝ √ε, so the extent shrinks by ≈ 0.32 when ε drops 10×; a contact face keeps its size (ratio ≈ 1).
# ratio < 0.6 ⇒ :unique, else :nonunique. Unbounded LP ⇒ :unbounded. Solver trouble ⇒ retry at larger ε, then :unknown.
function _extent_at(cs, p0, ε)
    blocks = Matrix{Float64}[]; rhs = Vector{Float64}[]; cones = Clarabel.SupportedCone[]
    for c in cs
        relax = ε * max(norm(gradient(c, p0)), 1.0) / _scale(c.quadric)
        Ac, bc, cone = _quadric_cone_rows(c.quadric, 3, 0; xcols = 1:3, relax)
        push!(blocks, Ac); push!(rhs, bc); push!(cones, cone)
    end
    A_ = sparse(reduce(vcat, blocks)); b_ = reduce(vcat, rhs)
    ext = zeros(3)
    for k in 1:3
        vals = Float64[]
        for sgn in (1.0, -1.0)                      # minimise x_k, then maximise x_k
            qv = zeros(3); qv[k] = sgn
            sol = _solve_conic(spzeros(3, 3), qv, A_, b_, cones; tols = (1e-8, 1e-6))
            occursin("DUAL_INFEASIBLE", uppercase(string(sol.status))) && return :unbounded, Inf
            _solved(sol) || return :failed, NaN
            push!(vals, sgn * sol.x[k])
        end
        ext[k] = -vals[1] - vals[2]                 # vals = [min x_k, −max x_k]
    end
    :ok, norm(ext)
end

function _witness_extent(A::GeoObject, B::GeoObject, w::AbstractVector, p0::AbstractVector)
    Bt = iszero(w) ? B : transform(B, translation(w))
    cs = vcat(A.constraints, Bt.constraints)
    for ε in (1e-5, 1e-4)
        s1, e1 = _extent_at(cs, p0, ε)
        s1 === :unbounded && return :unbounded, Inf
        s1 === :ok || continue
        s2, e2 = _extent_at(cs, p0, 10ε)
        s2 === :unbounded && return :unbounded, Inf
        s2 === :ok || continue
        return (e1 ≤ 1e-9 || e1 < 0.6 * e2 ? :unique : :nonunique), e1
    end
    :unknown, NaN
end

# ---- public API ----------------------------------------------------------------

"""
    distance(A, B; active_tol = 1e-6, check_witness = true)

Euclidean distance between two convex quadratic objects (Quadric / Constraint / GeoObject),
as a `DistanceResult`. See the file header of `collision/distance.jl` for the formulation and the
meaning of `status`. Wrap bare quadrics in `Constraint(q)` first if you want the reported
constraint ids to be ones you hold; otherwise use the index lists via `cone_a.indices`.

The problem is solved in a local frame (both objects translated so a point between them is the origin;
results are translated back). The conic solver's tolerances are relative to each quadric's largest matrix
entry, which far from the origin is dominated by ‖centre‖²; recentring keeps the accuracy independent of
where the scene sits.
"""
function distance(A_, B_; active_tol::Real = 1e-6, check_witness::Bool = true)
    A, B, T = _pair(A_, B_)
    (is_convex(A) && is_convex(B)) || throw(ArgumentError("distance requires convex constraints (A ⪰ 0)"))

    fa = feasibility(A); fb = feasibility(B)
    for f in (fa, fb)
        f.status === :numerical_failure && return _failed(T, :numerical_failure, f.solver_status, f.iterations, f.residual)
        f.status === :empty && return _failed(T, :empty_object, "object has empty intersection")
    end
    c0 = SVector{3,T}((fa.point + fb.point) / 2)                 # local origin between the two objects
    H0 = translation(-c0)
    A0, B0 = transform(A, H0), transform(B, H0)
    r = _distance_core(A0, B0, T, active_tol, check_witness)
    # Interior points of unbounded objects can sit far along an axis. If the witnesses ended up far from the
    # local origin, recentre on them and solve once more (keeping the first answer if that fails).
    if r.point_a !== nothing
        m = SVector{3,T}((r.point_a + r.point_b) / 2)
        if norm(m) > 1 + (isfinite(r.distance) ? r.distance : 0)
            H1 = translation(-m)
            r2 = _distance_core(transform(A0, H1), transform(B0, H1), T, active_tol, check_witness)
            if r2.status !== :numerical_failure
                return _unshift(r2, c0 + m)
            end
        end
    end
    _unshift(r, c0)
end

# Translate a result computed in the recentred frame back by c0 (gradients/directions are unchanged).
function _unshift(r::DistanceResult{T}, c0) where {T}
    sh(p) = p === nothing ? nothing : SVector{3,T}(p + c0)
    shc(nc) = nc === nothing ? nothing :
        NormalCone{T}(SVector{3,T}(nc.point + c0), nc.ids, nc.indices, nc.gradients, nc.rank, nc.degenerate)
    DistanceResult{T}(r.status, r.distance, sh(r.point_a), sh(r.point_b), r.normal, r.active_a, r.active_b,
                      shc(r.cone_a), shc(r.cone_b), r.cone_residual, r.witness_set, r.extent,
                      r.solver_status, r.iterations, r.residual)
end

function _distance_core(A::GeoObject{T}, B::GeoObject{T}, ::Type{T}, active_tol, check_witness) where {T}
    fab = _feasibility(vcat(A.constraints, B.constraints), T, 1e-7)
    fab.status === :numerical_failure && return _failed(T, :numerical_failure, fab.solver_status, fab.iterations, fab.residual)

    if fab.status === :interior                                  # interiors overlap
        p = fab.point
        return DistanceResult{T}(:intersecting, zero(T), p, p, nothing, Int[], Int[], nothing, nothing, zero(T),
                                 :not_applicable, T(NaN), fab.solver_status, fab.iterations, fab.residual)
    end

    if fab.status === :boundary                                  # touching
        p = fab.point
        atol_eff = max(active_tol, 10 * fab.residual)            # points are only as exact as the solve that produced them
        ca = normal_cone(A, p; tol = atol_eff); cb = normal_cone(B, p; tol = atol_eff)
        n = _touching_normal(ca, cb)
        res = zero(T)
        if n !== nothing
            _, ra = cone_membership(ca, n); _, rb = cone_membership(cb, -n)
            res = T(max(ra, rb))
        end
        ws, ext = check_witness ? _witness_extent(A, B, zeros(3), p) : (:not_checked, NaN)
        return DistanceResult{T}(:touching, zero(T), p, p, n, ca.ids, cb.ids, ca, cb, res, ws, T(ext),
                                 fab.solver_status, fab.iterations, fab.residual)
    end

    sol = _solve_distance(A, B)                                  # separated
    _solved(sol) || return _failed(T, :numerical_failure, string(sol.status), sol.iterations, max(sol.r_prim, sol.r_dual))
    x = SVector{3,T}(sol.x[1], sol.x[2], sol.x[3]); y = SVector{3,T}(sol.x[4], sol.x[5], sol.x[6])
    d = norm(x - y)
    n = d > 0 ? (y - x) / d : nothing
    atol_eff = max(active_tol, 10 * max(sol.r_prim, sol.r_dual))   # reduced-accuracy solves (ALMOST_SOLVED) need a wider test
    ca = normal_cone(A, x; tol = atol_eff); cb = normal_cone(B, y; tol = atol_eff)
    res = zero(T)
    if n !== nothing
        _, ra = cone_membership(ca, n); _, rb = cone_membership(cb, -n)
        res = T(max(ra, rb))
    end
    ws, ext = check_witness ? _witness_extent(A, B, x - y, x) : (:not_checked, NaN)
    DistanceResult{T}(:separated, d, x, y, n, ca.ids, cb.ids, ca, cb, res, ws, T(ext),
                      string(sol.status), Int(sol.iterations), T(max(sol.r_prim, sol.r_dual)))
end

# Normal for touching objects: n = Σ λᵢ ĝᵢᴬ = −Σ μⱼ ĝⱼᴮ with λ, μ ≥ 0, Σλ = 1 (an LP; the interior-point
# solution is a central point of the feasible set, hence a "central" normal when it is not unique).
function _touching_normal(ca::NormalCone{T}, cb::NormalCone{T}) where {T}
    ga = [normalize(g) for g in ca.gradients if norm(g) > 1e-8]
    gb = [normalize(g) for g in cb.gradients if norm(g) > 1e-8]
    (isempty(ga) || isempty(gb)) && return nothing
    ka, kb = length(ga), length(gb); k = ka + kb
    Ga = reduce(hcat, Vector{Float64}.(ga)); Gb = reduce(hcat, Vector{Float64}.(gb))
    Aeq = vcat(hcat(Ga, Gb), hcat(ones(1, ka), zeros(1, kb)))     # [Ga Gb; 1ᵀ 0] z = (0, 0, 0, 1)
    A_ = sparse(vcat(Aeq, -Matrix(1.0I, k, k)))
    b_ = vcat(zeros(3), 1.0, zeros(k))
    sol = _solve_conic(spzeros(k, k), zeros(k), A_, b_, [Clarabel.ZeroConeT(4), Clarabel.NonnegativeConeT(k)])
    _solved(sol) || return nothing
    n = Ga * sol.x[1:ka]
    norm(n) > 1e-6 ? SVector{3,T}(normalize(n)) : nothing
end

# ---- display -------------------------------------------------------------------

_fmt(v) = v === nothing ? "—" : string(round.(v; sigdigits = 5))
_fmtids(ids) = isempty(ids) ? "none" : join(("#$i" for i in ids), ", ")

function Base.show(io::IO, ::MIME"text/plain", r::DistanceResult)
    println(io, "DistanceResult: ", r.status)
    r.status in (:empty_object, :numerical_failure) ||
        println(io, "  distance      ", round(r.distance; sigdigits = 8))
    r.point_a === nothing || println(io, "  point on A    ", _fmt(r.point_a), r.status === :intersecting ? "   (interior point of A ∩ B)" : "")
    r.point_b === nothing || r.status === :intersecting || println(io, "  point on B    ", _fmt(r.point_b))
    r.status in (:separated, :touching) && println(io, "  normal (A→B)  ", _fmt(r.normal))
    r.status in (:separated, :touching) && println(io, "  active        A: ", _fmtids(r.active_a), "   B: ", _fmtids(r.active_b))
    r.status in (:separated, :touching) && println(io, "  cone residual ", round(r.cone_residual; sigdigits = 3),
                                                  "   witness set: ", r.witness_set,
                                                  isfinite(r.extent) && r.witness_set !== :unique ? " (extent ≈ $(round(r.extent; sigdigits = 3)))" : "")
    print(io, "  solver        ", r.solver_status, ", ", r.iterations, " iterations, residual ", round(r.residual; sigdigits = 3))
end
Base.show(io::IO, r::DistanceResult) = print(io, "DistanceResult(", r.status, ", d=", round(r.distance; sigdigits = 6), ")")
