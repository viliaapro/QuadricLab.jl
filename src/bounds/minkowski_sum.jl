# ---------------------------------------------------------------------------
# Minkowski sums of convex terms in a common ambient space ℝᴺ.
#
#   K = K₁ + K₂ + … + K_k        (each term Kᵢ lives in ℝᴺ)
#
# Only the support function is used, and it is additive:
#
#   h_K(u) = Σᵢ h_{Kᵢ}(u),        h_{A−B}(u) = h_A(u) + h_B(−u).
#
# A Cartesian product of terms on separate coordinates is the special case where each term is a lower-dimensional set embedded in
# ℝᴺ (a disk × a line = a flat disk + a line; see `embed_term`). Nothing tracks which coordinates a term uses, and terms may overlap
# freely: the maths is the same. Transformations act on the terms.
# ---------------------------------------------------------------------------

struct MinkowskiSum
    terms::Vector{Term}
    dim::Int
    function MinkowskiSum(terms::AbstractVector{<:Term})
        isempty(terms) && throw(ArgumentError("a Minkowski sum needs at least one term"))
        N = ambient_dim(terms[1])
        all(f -> ambient_dim(f) == N, terms) || throw(DimensionMismatch("all terms must live in the same ambient space (use embed_term for lower-dimensional terms)"))
        new(Vector{Term}(terms), N)
    end
end
MinkowskiSum(f::Term...) = MinkowskiSum(collect(Term, f))

ambient_dim(P::MinkowskiSum) = P.dim
Base.length(P::MinkowskiSum) = length(P.terms)

# ---- support -------------------------------------------------------------------------------------

function _combine(P::MinkowskiSum, rs::Vector{SupportResult})
    any(r -> r.status === :empty, rs) && return _sup_empty()
    i = findfirst(r -> r.status === :infinite, rs)
    i !== nothing && return _sup_infinite(rs[i].ray)                  # +∞ dominates a failed term; a term's ray is a ray of the sum
    any(r -> r.status === :failed, rs) && return _sup_failed(join((r.solver_status for r in rs if r.status === :failed), "; "))
    SupportResult(:finite, sum(r.value for r in rs), sum(r.point for r in rs), min(sum(r.face_dim for r in rs), P.dim), nothing,
                  join(unique(r.solver_status for r in rs), ","), maximum(r.residual for r in rs), Int[], rs)    # face of a sum = sum of faces (dim: upper bound)
end

function support(P::MinkowskiSum, u::AbstractVector)
    length(u) == P.dim || throw(DimensionMismatch("direction has length $(length(u)), sum lives in ℝ^$(P.dim)"))
    _combine(P, [support(f, u) for f in P.terms])
end

# h_A(u) + h_B(−u): an infinite value dominates (a failed side cannot make +∞ finite); the point is a − b.
function _diff(ra::SupportResult, rb::SupportResult, nmax::Integer)
    (ra.status === :empty || rb.status === :empty) && return _sup_empty()
    ra.status === :infinite && return _sup_infinite(ra.ray)
    rb.status === :infinite && return _sup_infinite(rb.ray === nothing ? nothing : -rb.ray)
    (ra.status === :failed || rb.status === :failed) && return _sup_failed("support failed: $(ra.solver_status) / $(rb.solver_status)")
    SupportResult(:finite, ra.value + rb.value, ra.point - rb.point, min(ra.face_dim + rb.face_dim, nmax), nothing,
                  ra.solver_status * "," * rb.solver_status, max(ra.residual, rb.residual), Int[], [ra, rb])
end

"""
    support_difference(A, B, u)

Support function of the Minkowski difference A − B = {a − b}: `h_A(u) + h_B(−u)`, for two terms or two sums of equal
dimension. `point` is a support point a − b; `face_dim` of a sum or difference is an upper bound.
"""
function support_difference(A::Term, B::Term, u::AbstractVector)
    ambient_dim(A) == ambient_dim(B) || throw(DimensionMismatch("terms live in different spaces"))
    _diff(support(A, u), support(B, -u), ambient_dim(A))
end
function support_difference(A::MinkowskiSum, B::MinkowskiSum, u::AbstractVector)
    ambient_dim(A) == ambient_dim(B) || throw(DimensionMismatch("sums live in different spaces"))
    _diff(support(A, u), support(B, -u), ambient_dim(A))
end

# ---- operations on the terms ---------------------------------------------------------------------------

recession_cone(P::MinkowskiSum) = MinkowskiSum([recession_cone(f) for f in P.terms])
translate(P::MinkowskiSum, t::AbstractVector) = MinkowskiSum([translate(P.terms[1], t); P.terms[2:end]])

"""
    transform(P, L, t)  /  transform(P, H)

Image of the sum under x ↦ Lx + t (or a 3-D homogeneous affine `H`, as for `GeoObject`): every term is mapped by `L`, the
translation is applied once. `L` must be invertible (term transforms need it); h_{LK+t}(u) = h_K(Lᵀu) + t·u is exact. Projective
`H` is rejected.
"""
function transform(P::MinkowskiSum, L::AbstractMatrix, t::AbstractVector)
    size(L) == (P.dim, P.dim) || throw(DimensionMismatch("L must be $(P.dim)×$(P.dim)"))
    length(t) == P.dim || throw(DimensionMismatch("t must have $(P.dim) entries"))
    Lm = Matrix{Float64}(L)
    MinkowskiSum([transform(f, Lm, i == 1 ? Vector{Float64}(t) : zeros(P.dim)) for (i, f) in enumerate(P.terms)])
end
function transform(P::MinkowskiSum, H::AbstractMatrix)
    (size(H) == (4, 4) && P.dim == 3) || throw(ArgumentError("a homogeneous transform needs a 4×4 matrix and a 3-D sum; use transform(P, L, t) in other dimensions"))
    is_affine(H) || throw(ArgumentError("transform requires an affine H (last row 0 0 0 1); a projective map does not preserve convexity or support functions"))
    transform(P, Matrix{Float64}(H[1:3, 1:3]), Vector{Float64}(H[1:3, 4]))
end

"Affine hull of the (nonempty) sum: the sum of the term hulls; nothing if a term is empty."
function affine_hull(P::MinkowskiSum)
    hs = [affine_hull(f) for f in P.terms]
    any(isnothing, hs) && return nothing
    origin = sum(h.origin for h in hs)
    cols = [h.basis for h in hs if size(h.basis, 2) > 0]
    isempty(cols) && return AffineHull(origin, zeros(P.dim, 0))
    F = svd(reduce(hcat, cols)); r = count(>(1e-9 * max(1.0, F.S[1])), F.S)
    AffineHull(origin, Matrix{Float64}(F.U[:, 1:r]))
end
is_full_dimensional(P::MinkowskiSum) = (h = affine_hull(P); h !== nothing && is_full_dimensional(h))

Base.show(io::IO, P::MinkowskiSum) = print(io, "MinkowskiSum(ℝ^", P.dim, ": ", join((string(typeof(f).name.name) for f in P.terms), " + "), ")")
