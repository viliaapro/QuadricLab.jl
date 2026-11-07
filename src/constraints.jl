# ---------------------------------------------------------------------------
# Constraint = one inequality q(x) ≤ 0 together with its appearance.
# The collision code reads only `.quadric`; the renderer reads both.
#
# Identity: every Constraint carries an `id` (unique within the session). The
# id is preserved by `transform` — it names "this surface" as it moves — and
# is what contacts/diagnostics will refer to. Use `reid` for an independent copy.
# ---------------------------------------------------------------------------

const _ID_COUNTER = Threads.Atomic{Int}(0)
next_id() = Threads.atomic_add!(_ID_COUNTER, 1) + 1

struct Constraint{T<:AbstractFloat}
    id::Int
    quadric::Quadric{T}
    style::Style
end

Constraint(q::Quadric, style::Style = Style(); id::Integer = next_id()) = Constraint(Int(id), q, style)
Constraint(c::Constraint, style::Style) = Constraint(c.id, c.quadric, style)

"Same constraint with a freshly minted id (an independent surface)."
reid(c::Constraint) = Constraint(next_id(), c.quadric, c.style)

# Forwarding of the quadric interface.
evaluate(c::Constraint, p::AbstractVector) = evaluate(c.quadric, p)
(c::Constraint)(p::AbstractVector) = evaluate(c.quadric, p)
gradient(c::Constraint, p::AbstractVector) = gradient(c.quadric, p)
normal(c::Constraint, p::AbstractVector; kwargs...) = normal(c.quadric, p; kwargs...)
affine_part(c::Constraint) = affine_part(c.quadric)
is_convex(c::Constraint; kwargs...) = is_convex(c.quadric; kwargs...)

"Affine transform of the quadric and of the style's texture mapping; id preserved."
function transform(c::Constraint, H::AbstractMatrix)
    Q = transform(c.quadric, H)
    Constraint(c.id, Q, Style(c.style; mapping = transform(c.style.mapping, H)))
end

Base.show(io::IO, c::Constraint) = print(io, "Constraint(#", c.id, ", ", c.quadric, ", ", c.style, ")")
