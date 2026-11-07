# ---------------------------------------------------------------------------
# Primitive constructors. All return a `Quadric` (Style/Constraint wrapping
# arrives in Phase 2). Region convention is always q(x) ≤ 0.
# ---------------------------------------------------------------------------

const origin = SVector(0.0, 0.0, 0.0)
const xaxis  = SVector(1.0, 0.0, 0.0)
const yaxis  = SVector(0.0, 1.0, 0.0)
const zaxis  = SVector(0.0, 0.0, 1.0)
const up     = zaxis

function _unit(v, what::AbstractString)
    s = SVector{3,Float64}(v)
    n = norm(s)
    (isfinite(n) && n > 0) || throw(ArgumentError("$what must be a nonzero finite vector"))
    s / n
end

# M = [A b; bᵀ c]
function _homogeneous(A::SMatrix{3,3,Float64}, b::SVector{3,Float64}, c::Float64)
    SMatrix{4,4,Float64,16}(
        A[1,1], A[2,1], A[3,1], b[1],
        A[1,2], A[2,2], A[3,2], b[2],
        A[1,3], A[2,3], A[3,3], b[3],
        b[1],   b[2],   b[3],   c)
end

"""
    sphere(center, radius)

Solid ball ‖x − center‖² ≤ radius². A = I, b = −center, c = ‖center‖² − r².
"""
function sphere(center::AbstractVector, radius::Real)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    c0 = SVector{3,Float64}(center)
    Quadric(_homogeneous(SMatrix{3,3,Float64}(I), -c0, dot(c0, c0) - Float64(radius)^2))
end

"""
    cylinder(axis, radius; point = origin)

Solid infinite circular cylinder: distance from the line {point + t·axis} ≤ radius.
`axis` is normalised internally (only its direction matters; sign is irrelevant).
With P = I − uuᵀ: q(x) = ‖P(x − point)‖² − r², so A = P, b = −P·point,
c = pointᵀP·point − r².
"""
function cylinder(axis::AbstractVector, radius::Real; point::AbstractVector = origin)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    u = _unit(axis, "cylinder axis")
    p = SVector{3,Float64}(point)
    P = SMatrix{3,3,Float64}(I) - u * u'
    Quadric(_homogeneous(P, -(P * p), dot(p, P * p) - Float64(radius)^2))
end

"""
    halfspace(normal, offset)

Closed halfspace { x : n̂·x ≤ offset } with n̂ = normal/‖normal‖.
`n̂` is the OUTWARD normal; `offset` is the signed distance of the boundary
plane from the origin along n̂ (after normalisation). Degenerate quadric with
A = 0, b = n̂/2, c = −offset, so q(x) = n̂·x − offset is the signed distance.
Example: halfspace(up, 0.0) is the solid region z ≤ 0 (below the floor plane).
"""
function halfspace(normal::AbstractVector, offset::Real)
    n = _unit(normal, "halfspace normal")
    Quadric(_homogeneous(zeros(SMatrix{3,3,Float64}), n / 2, -Float64(offset)))
end

# ---- Styled variants: a trailing `Style` yields a Constraint instead of a bare Quadric.
sphere(center::AbstractVector, radius::Real, style::Style) = Constraint(sphere(center, radius), style)
cylinder(axis::AbstractVector, radius::Real, style::Style; kwargs...) =
    Constraint(cylinder(axis, radius; kwargs...), style)
halfspace(normal::AbstractVector, offset::Real, style::Style) = Constraint(halfspace(normal, offset), style)
