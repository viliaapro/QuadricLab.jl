# ---------------------------------------------------------------------------
# Support-function results for the bounding Minkowski-sum layer (Phase 4).
#
# h_K(u) = sup_{x∈K} u·x. Four outcomes are kept distinct:
#   :finite    value h, a support point, and the dimension of the optimal face (0 = unique point)
#   :infinite  h = +∞ (u is outside the barrier cone) — a legitimate answer, never a solver failure;
#              `ray` is a recession direction d with u·d > 0 when one exists (for non-polyhedral sets such as
#              paraboloids the supremum can be infinite without any such ray)
#   :empty     K = ∅ (h = −∞)
#   :failed    numerical failure (value NaN)
# The bounding layer works in Float64 (its LPs go through Clarabel); coordinates are plain Vectors.
# ---------------------------------------------------------------------------

struct SupportResult
    status::Symbol
    value::Float64
    point::Union{Nothing,Vector{Float64}}       # a support point, in the relative interior of the exposed face
    face_dim::Int                               # dim of the exposed face ∂h(u); 0 ⇔ h is differentiable at u (gradient = point)
    ray::Union{Nothing,Vector{Float64}}
    solver_status::String
    residual::Float64
    active::Vector{Int}                         # polyhedron: indices of the active inequality rows at the support point
    parts::Vector{SupportResult}                # sum: per-term results; difference: [A result, B result]
end
SupportResult(status, value, point, face_dim, ray, solver, resid) =
    SupportResult(status, value, point, face_dim, ray, solver, resid, Int[], SupportResult[])

"True when the support function is nonsmooth at this direction (the exposed face is not a single point)."
nonsmooth(r::SupportResult) = r.status === :finite && r.face_dim > 0

_sup_finite(v, p, fd; solver = "analytic", resid = 0.0) = SupportResult(:finite, Float64(v), Vector{Float64}(p), fd, nothing, solver, resid)
_sup_infinite(ray; solver = "analytic") = SupportResult(:infinite, Inf, nothing, 0, ray === nothing ? nothing : Vector{Float64}(ray), solver, 0.0)
_sup_empty(; solver = "analytic") = SupportResult(:empty, -Inf, nothing, 0, nothing, solver, 0.0)
_sup_failed(msg) = SupportResult(:failed, NaN, nothing, 0, nothing, String(msg), NaN)

"Affine hull origin + span(basis); `basis` has orthonormal columns (zero columns: a point)."
struct AffineHull
    origin::Vector{Float64}
    basis::Matrix{Float64}
end
ambient_dim(h::AffineHull) = length(h.origin)
Base.ndims(h::AffineHull) = size(h.basis, 2)
is_full_dimensional(h::AffineHull) = ndims(h) == ambient_dim(h)

function Base.show(io::IO, r::SupportResult)
    print(io, "SupportResult(", r.status)
    r.status === :finite && print(io, ", h=", round(r.value; sigdigits = 8), ", face_dim=", r.face_dim)
    r.status === :infinite && print(io, r.ray === nothing ? ", no recession ray" : ", ray=$(round.(r.ray; sigdigits = 4))")
    r.status === :failed && print(io, ", ", r.solver_status)
    print(io, ")")
end
