# ---------------------------------------------------------------------------
# Contact: where (and against which surfaces) two objects touch or overlap.
#
#  :separated    no contact. `distance` holds the gap; `point` is nothing.
#  :touching     boundaries meet, interiors disjoint. `point` is a central point of the contact
#                set A ∩ B; `normal` (A→B) satisfies n ∈ N_A(p), −n ∈ N_B(p); the active
#                constraint ids of both objects and the normal cones are reported.
#                `contact_set` says whether the contact is :unique, :nonunique (finite face) or
#                :unbounded (e.g. plane on plane, cylinder lying on a plane).
#  :intersecting interiors overlap. `point` is the most interior point of A ∩ B (it lies in neither
#                boundary), no normal is defined here — use the Phase 4 bounding-product penetration estimate for the
#                separating direction.
#
# Multiple active constraints at one point (edges, corners) are kept, with their identities, in
# `cone_a` / `cone_b`; the reported normal is a central direction of the intersection of the cones.
# ---------------------------------------------------------------------------

struct ContactResult{T}
    status::Symbol
    point::Union{Nothing,SVector{3,T}}
    normal::Union{Nothing,SVector{3,T}}
    distance::T
    active_a::Vector{Int}
    active_b::Vector{Int}
    cone_a::Union{Nothing,NormalCone{T}}
    cone_b::Union{Nothing,NormalCone{T}}
    contact_set::Symbol
    extent::T
    cone_residual::T
    solver_status::String
    iterations::Int
    residual::T
end

"""
    contact(A, B; kwargs...)

Contact analysis of two convex quadratic objects; keywords as for `distance`.
Returns a `ContactResult`.
"""
function contact(A, B; kwargs...)
    r = distance(A, B; kwargs...)
    T = typeof(r.distance)
    point = r.status in (:touching, :intersecting) ? r.point_a : nothing
    ContactResult{T}(r.status, point, r.status === :touching ? r.normal : nothing,
                     r.distance, r.active_a, r.active_b, r.cone_a, r.cone_b,
                     r.status === :touching ? r.witness_set : :not_applicable, r.extent,
                     r.cone_residual, r.solver_status, r.iterations, r.residual)
end

function Base.show(io::IO, ::MIME"text/plain", r::ContactResult)
    println(io, "ContactResult: ", r.status)
    if r.status === :separated
        println(io, "  no contact; gap ", round(r.distance; sigdigits = 6))
    elseif r.status in (:touching, :intersecting)
        println(io, "  point         ", _fmt(r.point), r.status === :intersecting ? "   (interior point of A ∩ B)" : "")
        if r.status === :touching
            println(io, "  normal (A→B)  ", _fmt(r.normal))
            println(io, "  active        A: ", _fmtids(r.active_a), "   B: ", _fmtids(r.active_b))
            r.cone_a === nothing || println(io, "  cone A        ", r.cone_a)
            r.cone_b === nothing || println(io, "  cone B        ", r.cone_b)
            println(io, "  contact set   ", r.contact_set,
                    isfinite(r.extent) && r.contact_set !== :unique ? " (extent ≈ $(round(r.extent; sigdigits = 3)))" : "",
                    "   cone residual ", round(r.cone_residual; sigdigits = 3))
        end
    end
    print(io, "  solver        ", r.solver_status, ", ", r.iterations, " iterations, residual ", round(r.residual; sigdigits = 3))
end
Base.show(io::IO, r::ContactResult) = print(io, "ContactResult(", r.status, ")")
