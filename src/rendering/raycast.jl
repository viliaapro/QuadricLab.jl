# ---------------------------------------------------------------------------
# CPU reference ray caster. Mirrors the algorithm of shaders/quadric.frag
# (candidate roots per quadric → validity against all other inequalities →
# nearest with priority tie-break) in Float64, so renderer maths can be tested
# against plain quadric evaluation. Not used for drawing.
# ---------------------------------------------------------------------------

struct RayHit{T}
    t::T
    point::SVector{3,T}
    normal::SVector{3,T}      # outward unit gradient of the generating constraint
    constraint::Int           # index into obj.constraints
    id::Int                   # that constraint's id
end

"""
Real roots (ascending) of a t² + 2 b t + c = 0 with the same degeneracy handling as the
shader: relative test for a ≈ 0 (linear case), stable root formula otherwise.
"""
function ray_roots(a::T, b::T, c::T; scale::T = one(T)) where {T}
    if abs(a) ≤ 1e-6 * scale
        abs(b) ≤ 1e-12 * max(scale, abs(c)) && return ()          # parallel to the boundary
        return (-c / (2b),)
    end
    disc = b * b - a * c
    disc < 0 && return ()
    sq = sqrt(disc)
    q = -(b + (b ≥ 0 ? sq : -sq))
    q == 0 && return (zero(T),)
    t1, t2 = q / a, c / q
    t1 ≤ t2 ? (t1, t2) : (t2, t1)
end

"""
    ray_hit(obj, o, d; tmin = 0, tol = 1e-4)

Nearest boundary point of `obj` along the ray o + t d (t > tmin), or `nothing`.
A candidate on constraint i is valid iff qⱼ(x)/‖∇qⱼ(x)‖ ≤ tol·(1 + 0.02t) for all j ≠ i.
Ties within `tol` go to the lowest constraint index (deterministic priority).
"""
function ray_hit(o::GeoObject{T}, origin::AbstractVector, dir::AbstractVector;
                 tmin::Real = 0.0, tol::Real = 1e-4) where {T}
    O = SVector{3,T}(origin); D = SVector{3,T}(dir) / norm(dir)
    best_t = T(Inf); best_i = 0
    for (i, c) in enumerate(o.constraints)
        A, b, cc = affine_part(c)
        # q(O + tD) = (D·AD) t² + 2 (D·(AO + b)) t + q(O)
        a = dot(D, A * D); bb = dot(D, A * O + b); c0 = evaluate(c, O)
        sc = max(tr(A), Base.eps(T))
        for t in ray_roots(a, bb, c0; scale = sc)
            t > tmin || continue
            t < best_t - tol || continue
            X = O + t * D
            valid = true
            for (j, cj) in enumerate(o.constraints)
                j == i && continue
                g = norm(gradient(cj, X))
                if evaluate(cj, X) > tol * (1 + 0.02t) * g
                    valid = false; break
                end
            end
            if valid
                best_t = t; best_i = i
                break                        # roots ascending: first valid is nearest for this constraint
            end
        end
    end
    best_i == 0 && return nothing
    c = o.constraints[best_i]; X = O + best_t * D
    n = something(normal(c, X), SVector{3,T}(0, 0, 0))
    RayHit{T}(best_t, X, n, best_i, c.id)
end

"Nearest hit over all objects of a scene: (object index, RayHit) or nothing."
function ray_hit(s::Scene, origin, dir; kwargs...)
    best = nothing; bi = 0
    for (k, o) in enumerate(s.objects)
        h = ray_hit(o, origin, dir; kwargs...)
        h === nothing && continue
        if best === nothing || h.t < best.t
            best = h; bi = k
        end
    end
    best === nothing ? nothing : (bi, best)
end
