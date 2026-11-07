# ---------------------------------------------------------------------------
# Active constraints and normal cones.
#
# At a boundary point p of K = ⋂ᵢ{qᵢ ≤ 0} the active set is I(p) = {i : qᵢ(p) = 0}. When the
# gradients ∇qᵢ(p) ≠ 0 (Slater / constraint qualification) the normal cone is
#     N_K(p) = { Σ_{i∈I(p)} λᵢ ∇qᵢ(p) : λ ≥ 0 }.
# Active means |qᵢ(p)| ≤ tol·max(‖∇qᵢ(p)‖, 1) (a distance-like tolerance comparable across
# spheres, cylinders and halfspaces). A constraint with ∇qᵢ(p) ≈ 0 and qᵢ(p) ≈ 0 is
# *degenerate* (e.g. {x² ≤ 0}, a double plane): it contributes no generator, and the cone
# computed from the remaining gradients is not a certificate. `degenerate` flags this.
# ---------------------------------------------------------------------------

struct NormalCone{T}
    point::SVector{3,T}
    ids::Vector{Int}                       # ids of the active constraints (identity preserved)
    indices::Vector{Int}                   # their positions in the object
    gradients::Vector{SVector{3,T}}        # ∇qᵢ(p), outward, unnormalised
    rank::Int                              # dimension of the span of the gradients
    degenerate::Bool
end

"Indices of the constraints of `o` active at `p` (distance-like tolerance, see file header)."
function active_indices(o::GeoObject, p::AbstractVector; tol::Real = 1e-6)
    [i for (i, c) in enumerate(o.constraints) if abs(evaluate(c, p)) ≤ tol * max(norm(gradient(c, p)), 1)]
end

"""
    normal_cone(obj, p; tol = 1e-6)

Active constraints of `obj` at `p` with their gradients (the generators of the normal cone).
"""
function normal_cone(o::GeoObject{T}, p::AbstractVector; tol::Real = 1e-6, gtol::Real = 1e-8) where {T}
    idx = active_indices(o, p; tol)
    gs = SVector{3,T}[gradient(o.constraints[i], p) for i in idx]
    degenerate = any(g -> norm(g) ≤ gtol, gs)
    live = filter(g -> norm(g) > gtol, gs)
    r = isempty(live) ? 0 : rank(reduce(hcat, normalize.(live)); atol = 1e-6)
    NormalCone{T}(SVector{3,T}(p), [o.constraints[i].id for i in idx], idx, gs, r, degenerate)
end

"""
    nnls(G, n)

Non-negative least squares  min_{λ ≥ 0} ‖Gλ − n‖  (Lawson–Hanson); returns (λ, residual norm).
Columns of `G` are few (≤ number of active constraints), so the dense active-set method is ample.
"""
function nnls(G::AbstractMatrix, n::AbstractVector; tol::Real = 1e-12, maxiter::Integer = 100)
    k = size(G, 2)
    λ = zeros(k)
    k == 0 && return λ, norm(n)
    passive = falses(k)
    w = G' * (n - G * λ)
    it = 0
    while it < maxiter && any(j -> !passive[j] && w[j] > tol, 1:k)
        it += 1
        j = argmax([passive[i] ? -Inf : w[i] for i in 1:k])
        passive[j] = true
        while true
            s = zeros(k)
            s[passive] = G[:, passive] \ n
            if all(s[passive] .> tol)
                λ = s; break
            end
            α = minimum(λ[i] / (λ[i] - s[i]) for i in 1:k if passive[i] && s[i] ≤ tol; init = 1.0)
            λ += α * (s - λ)
            for i in 1:k
                passive[i] && λ[i] ≤ tol && (passive[i] = false; λ[i] = 0.0)
            end
            any(passive) || break
        end
        w = G' * (n - G * λ)
    end
    λ, norm(G * λ - n)
end

"""
    cone_membership(nc, v)

Is the unit direction `v` in the normal cone? Returns `(λ, residual)` with `v ≈ Σ λᵢ ĝᵢ`
over the unit active gradients ĝᵢ, λ ≥ 0; `residual ≈ 0` iff `v ∈ N`. Degenerate or
empty cones return residual = ‖v‖ (nothing to certify against).
"""
function cone_membership(nc::NormalCone, v::AbstractVector)
    gs = [normalize(g) for g in nc.gradients if norm(g) > 1e-8]
    isempty(gs) && return Float64[], norm(v)
    nnls(reduce(hcat, Vector{Float64}.(gs)), Vector{Float64}(v))
end

function Base.show(io::IO, nc::NormalCone)
    print(io, "NormalCone(rank ", nc.rank, ", constraints ", isempty(nc.ids) ? "none" : join(("#$i" for i in nc.ids), ", "),
          nc.degenerate ? ", DEGENERATE" : "", ")")
end
