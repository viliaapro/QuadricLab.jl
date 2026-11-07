# ---------------------------------------------------------------------------
# Quadric: canonical homogeneous representation.
#
#   Q(X) = Xᵀ M X,   X = (x, y, z, w)ᵀ,   M symmetric 4×4.
#
# Affine chart w = 1 (all Euclidean computation happens here):
#
#   q(x) = xᵀ A x + 2 bᵀ x + c,   A = M[1:3,1:3],  b = M[1:3,4],  c = M[4,4].
#
# The intended region is { x : q(x) ≤ 0 }.  Convex iff A is PSD (necessary
# and sufficient for q to be a convex function of x).
#
# Numerical assumptions: symmetry is validated to a relative tolerance and the
# stored matrix is then symmetrised exactly, (M+Mᵀ)/2.
# ---------------------------------------------------------------------------

struct Quadric{T<:AbstractFloat}
    M::SMatrix{4,4,T,16}

    function Quadric{T}(M::SMatrix{4,4,T,16}; symtol::Real = 1e-10) where {T<:AbstractFloat}
        all(isfinite, M) || throw(ArgumentError("quadric matrix has non-finite entries"))
        scale = max(maximum(abs, M), one(T))
        maximum(abs, M - M') ≤ symtol * scale ||
            throw(ArgumentError("quadric matrix is not symmetric (tolerance $(symtol)·scale)"))
        new{T}((M + M') / 2)
    end
end

function Quadric(M::AbstractMatrix{<:Real}; kwargs...)
    size(M) == (4, 4) || throw(DimensionMismatch("quadric matrix must be 4×4, got $(size(M))"))
    T = eltype(M) <: AbstractFloat ? eltype(M) : Float64
    Quadric{T}(SMatrix{4,4,T,16}(M); kwargs...)
end

Base.eltype(::Type{Quadric{T}}) where {T} = T

"Affine-chart data (A, b, c) with q(x) = xᵀAx + 2bᵀx + c."
function affine_part(Q::Quadric)
    M = Q.M
    A = M[SOneTo(3), SOneTo(3)]
    b = M[SOneTo(3), 4]
    c = M[4, 4]
    return A, b, c
end

"Homogeneous evaluation Q(X) = Xᵀ M X for X = (x,y,z,w)."
homogeneous_eval(Q::Quadric, X::AbstractVector) = dot(X, Q.M * X)

"Evaluate q at an affine point x (chart w = 1). Negative/zero = inside/boundary."
function evaluate(Q::Quadric, p::AbstractVector)
    length(p) == 3 || throw(DimensionMismatch("affine point must have 3 components"))
    A, b, c = affine_part(Q)
    x = SVector{3}(p)
    return dot(x, A * x) + 2 * dot(b, x) + c
end

(Q::Quadric)(p::AbstractVector) = evaluate(Q, p)

"Spatial gradient ∇q(x) = 2(Ax + b). Points outward from the region on its boundary."
function gradient(Q::Quadric, p::AbstractVector)
    A, b, _ = affine_part(Q)
    return 2 * (A * SVector{3}(p) + b)
end

"""
Unit outward normal ∇q/‖∇q‖, or `nothing` where the gradient vanishes
(‖∇q‖ ≤ tol, e.g. the minimiser of q inside a region).
"""
function normal(Q::Quadric, p::AbstractVector; tol::Real = 1e-12)
    g = gradient(Q, p)
    n = norm(g)
    n ≤ tol ? nothing : g / n
end

"Is { q ≤ 0 } convex, i.e. A positive semidefinite (eigenvalues ≥ -tol·scale)?"
function is_convex(Q::Quadric; tol::Real = 1e-10)
    A, _, _ = affine_part(Q)
    Am = Matrix(A)
    minimum(eigvals(Symmetric(Am))) ≥ -tol * max(1, opnorm(Am))
end

function Base.show(io::IO, Q::Quadric)
    A, _, _ = affine_part(Q)
    kind = iszero(A) ? "A=0 (halfspace-like)" : (is_convex(Q) ? "convex" : "NON-convex")
    print(io, "Quadric{", eltype(Q.M), "}(", kind, ", rank(A)=", rank(Matrix(A)), ")")
end
