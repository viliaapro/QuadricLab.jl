# ---------------------------------------------------------------------------
# GeoObject: finite intersection K = ⋂ᵢ {x : qᵢ(x) ≤ 0}.
# Scene: a union of objects (compositional; convexity is per object only).
# ---------------------------------------------------------------------------

struct GeoObject{T<:AbstractFloat}
    constraints::Vector{Constraint{T}}
    function GeoObject{T}(cs::AbstractVector{<:Constraint}) where {T<:AbstractFloat}
        isempty(cs) && throw(ArgumentError("a GeoObject needs at least one constraint"))
        ids = [c.id for c in cs]
        allunique(ids) || throw(ArgumentError("constraint ids in one object must be unique (use reid)"))
        new{T}(Vector{Constraint{T}}(cs))
    end
end

_cast(::Type{T}, c::Constraint{T}) where {T} = c
_cast(::Type{T}, c::Constraint) where {T} =
    Constraint(c.id, Quadric{T}(SMatrix{4,4,T,16}(c.quadric.M)), c.style)
_as_constraint(q::Quadric) = Constraint(q)
_as_constraint(c::Constraint) = c

_parts(x::Union{Quadric,Constraint}) = Constraint[_as_constraint(x)]
_parts(o::GeoObject) = Constraint[c for c in o.constraints]

"""
    intersection(parts...)

Object `⋂ parts`. Each part is a `Quadric` (default Style), a `Constraint`, or a
`GeoObject` (flattened). Constraint order is the seam-priority order used by the renderer.
"""
function intersection(parts...)
    cs = reduce(vcat, (_parts(p) for p in parts))
    T = mapreduce(c -> eltype(c.quadric.M), promote_type, cs)
    GeoObject{T}([_cast(T, c) for c in cs])
end

GeoObject(parts...) = intersection(parts...)

Base.length(o::GeoObject) = length(o.constraints)
Base.iterate(o::GeoObject, s...) = iterate(o.constraints, s...)
Base.getindex(o::GeoObject, i) = o.constraints[i]
Base.eltype(::Type{GeoObject{T}}) where {T} = Constraint{T}
constraints(o::GeoObject) = o.constraints

"max_i qᵢ(x): ≤ 0 iff x lies in the object."
violation(o::GeoObject, p::AbstractVector) = maximum(evaluate(c, p) for c in o.constraints)

"Membership with absolute tolerance on each qᵢ (default exact)."
inside(o::GeoObject, p::AbstractVector; tol::Real = 0.0) = all(evaluate(c, p) ≤ tol for c in o.constraints)
Base.in(p::AbstractVector, o::GeoObject) = inside(o, p)

"""
Constraints active at `p`: those with |qᵢ(p)|/max(‖∇qᵢ(p)‖,1) ≤ tol (a distance-like
test, so it is comparable across spheres and halfspaces). Returns the constraints themselves,
so their ids and styles stay available.
"""
function active_constraints(o::GeoObject, p::AbstractVector; tol::Real = 1e-8)
    [c for c in o.constraints if abs(evaluate(c, p)) ≤ tol * max(norm(gradient(c, p)), 1)]
end

is_convex(o::GeoObject; kwargs...) = all(is_convex(c; kwargs...) for c in o.constraints)

transform(o::GeoObject{T}, H::AbstractMatrix) where {T} = GeoObject{T}([transform(c, H) for c in o.constraints])

function Base.show(io::IO, o::GeoObject)
    print(io, "GeoObject(", length(o), " constraint", length(o) == 1 ? "" : "s", ": ")
    join(io, ("#$(c.id)" for c in o.constraints), ", ")
    print(io, ")")
end

"""
    Scene(parts...)

Union of objects for rendering. Each part may be a Quadric, Constraint (single-constraint
object) or GeoObject. Each object is individually an intersection; the scene composes them
by union (nearest visible surface wins).
"""
struct Scene
    objects::Vector{GeoObject{Float64}}
    Scene(objects::Vector{GeoObject{Float64}}) = new(objects)   # explicit: no implicit converting constructor
end

_as_object(x::Union{Quadric,Constraint}) = intersection(x)
_as_object(o::GeoObject) = o
function Scene(parts...)
    objs = GeoObject{Float64}[]
    for p in parts
        o = _as_object(p)
        push!(objs, GeoObject{Float64}([_cast(Float64, c) for c in o.constraints]))
    end
    Scene(objs)
end
Scene(s::Scene) = s
Base.length(s::Scene) = length(s.objects)
Base.show(io::IO, s::Scene) = print(io, "Scene(", length(s), " object", length(s) == 1 ? "" : "s", ")")
