# ---------------------------------------------------------------------------
# Homogeneous transforms H (4×4, invertible), acting on points X ↦ H X.
#
# Pullback convention:  M' = H⁻ᵀ M H⁻¹   ⇒   Q'(H X) = Q(X).
# i.e. transform(Q, H) is the image of the region {Q ≤ 0} under H.
#
# AFFINE (last row [0 0 0 1]) preserves the chart w=1 and convexity
# (A' = L⁻ᵀ A L⁻¹ stays PSD).  GENERAL PROJECTIVE transforms do not preserve
# the chart, distances or convexity; they are only available through
# `projective_transform`.
# ---------------------------------------------------------------------------

"Does H have last row (0,0,0,1) up to tolerance?"
is_affine(H::AbstractMatrix; tol::Real = 1e-12) =
    size(H) == (4, 4) && all(abs.(H[4, 1:3]) .≤ tol) && abs(H[4, 4] - 1) ≤ tol

"Build affine H = [L t; 0 1]."
function affine(L::AbstractMatrix, t::AbstractVector = zeros(3))
    size(L) == (3, 3) || throw(DimensionMismatch("linear part must be 3×3"))
    length(t) == 3 || throw(DimensionMismatch("translation must have 3 components"))
    T = float(promote_type(eltype(L), eltype(t)))
    SMatrix{4,4,T,16}(
        L[1,1], L[2,1], L[3,1], 0,
        L[1,2], L[2,2], L[3,2], 0,
        L[1,3], L[2,3], L[3,3], 0,
        t[1],   t[2],   t[3],   1)
end

"Translation by vector v."
translation(v::AbstractVector) = affine(SMatrix{3,3,Float64}(I), v)

"Right-handed rotation by `angle` (radians) about `axis` (normalised internally; Rodrigues)."
function rotation(axis::AbstractVector, angle::Real)
    n = norm(axis)
    n > 0 || throw(ArgumentError("rotation axis must be nonzero"))
    u = SVector{3,Float64}(axis) / n
    # column-major; K*v = u × v
    K = SMatrix{3,3,Float64}(0, u[3], -u[2],  -u[3], 0, u[1],  u[2], -u[1], 0)
    R = SMatrix{3,3,Float64}(I) + sin(angle) * K + (1 - cos(angle)) * (K * K)
    affine(R)
end

"Scaling by per-axis factors (nonzero)."
function scaling(s::AbstractVector)
    all(!iszero, s) || throw(ArgumentError("scale factors must be nonzero"))
    affine(SMatrix{3,3,Float64}(Diagonal(SVector{3,Float64}(s))))
end
scaling(s::Real) = scaling(SVector(s, s, s))

"""
    transform(Q, H)

Affine pullback M' = H⁻ᵀ M H⁻¹. Throws if H is not affine or is singular;
use `projective_transform` for general projective maps.
"""
function transform(Q::Quadric, H::AbstractMatrix)
    is_affine(H) || throw(ArgumentError(
        "transform requires an affine H (last row 0 0 0 1); use projective_transform for general projective maps"))
    _pullback(Q, H)
end

"""
    projective_transform(Q, H)

General homogeneous pullback. The result is a valid homogeneous quadric, but its
affine-chart region need not be convex or Euclidean-congruent to the original.
"""
projective_transform(Q::Quadric, H::AbstractMatrix) = _pullback(Q, H)

function _pullback(Q::Quadric{T}, H::AbstractMatrix) where {T}
    size(H) == (4, 4) || throw(DimensionMismatch("H must be 4×4"))
    Hs = SMatrix{4,4,T,16}(H)
    abs(det(Hs)) > 4 * eps(T) || throw(ArgumentError("H is singular or nearly singular"))
    Hi = inv(Hs)
    Quadric{T}(Hi' * Q.M * Hi)
end
