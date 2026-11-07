# ---------------------------------------------------------------------------
# Support-based conservative collision recovery (Phase 4).
#
# GUARANTEE. Let A′ ⊇ A, B′ ⊇ B be the supplied bounds (Minkowski sums of terms), D′ = A′ − B′ ⊇ D = A − B, and h = h_{D′},
# h(u) = h_{A′}(u) + h_{B′}(−u), evaluated through the terms of the sums (the Minkowski difference is never built).
#   • Translating A by t (B fixed) gives interior overlap iff −t ∈ int D.
#   • For any unit u with h(u) finite, int D′ lies in the open halfspace {x : u·x < h(u)}. So −t = s·u with s ≥ h(u) is not in
#     int D′ ⊇ int D: t_A = −s·u removes interior overlap — for EVERY direction u, not only a minimiser. (B alone: t_B = +s·u.
#     Split by weights: t_A = −w_A s u, t_B = +w_B s u; only t_B − t_A = s·u matters.) In terms of A's own translation direction
#     v = −u the magnitude is h(−v) = h_{A′}(−v) + h_{B′}(v).
#   • Invertible affine maps of a term are exact: h_{LK+c}(u) = h_K(Lᵀu) + c·u, so transformed terms need nothing special.
#   • h(u) < 0 for some u certifies that the bounds' interiors are disjoint (a separating direction).
# The optimiser therefore only reduces the magnitude s; it can never invalidate sufficiency. The one caveat is numerical: a
# support value can be low by the solver tolerance, which is what `margin` is for.
#
# This is NOT an exact penetration-depth solver. A stationary point of the descent is a local minimiser of h on the sphere;
# nothing here claims it is the global one.
#
# OPTIMISER (one for every convex term type). Minimise h over the unit sphere.
#   differentiable (exposed face a single point, face_dim = 0): gradient = support point, Riemannian step g_T = g − (g·u)u.
#   nonsmooth (face_dim > 0, e.g. polyhedral faces/edges): ∂h(u) is the exposed face; gradient sampling — support points at u
#     and at u ± ε·(tangent basis) form a bundle; its tangentially projected convex hull's minimum-norm element is the
#     descent direction (zero ⇒ stationary, e.g. at a facet normal).
#   Armijo line search on normalised updates; user budget (`max_iter`, `max_evals`) and accuracy (`tol`). At nonsmooth points the
#   sampling radius is refined from 1e-4 to a floor of 1e-6 (LP noise); there "stationary" means ε-stationary at that resolution,
#   so the reported value can exceed the local minimum by about 1e-6 — conservative, never below it.
# SEEDS. Reference-locus direction first; for meeting loci, the complement of the loci directions; then deterministic seeds
#   (only if needed, or `extra_starts` more are requested).
# UNBOUNDED/DEGENERATE. Finite support requires u ∈ polar(rec D′), and rec D′ ⊇ rec(A′) − rec(B′) (the recession cone of a sum contains the sum of the cones), so
#   K = { u : u ∈ polar(rec Kᵢᴬ) ∀i,  −u ∈ polar(rec Kⱼᴮ) ∀j }  contains every finite-support direction. With rec Kᵢ = {Gd ≤ 0,
#   Ed = 0}, polar(rec Kᵢ) = {Gᵀλ + Eᵀμ : λ ≥ 0}, so projecting a seed onto K is one small QP (exact for slabs, rays, lineality —
#   no sampling of arbitrary sphere seeds). K = {0} (checked by projecting ±eᵢ) certifies :impossible: every direction has infinite
#   support. On K's boundary a non-closed barrier cone (paraboloids) can still give +∞; seeds are then blended toward an interior
#   point of K. No finite direction found otherwise: :no_direction_found (not a proof). Empty sets: :empty. Empty interior
#   (lower-dimensional D′): :degenerate, magnitude 0 (no interior to eliminate).
# ---------------------------------------------------------------------------

struct Displacement
    status::Symbol                 # :sufficient :separated :degenerate :empty :impossible :no_direction_found :failed
    magnitude::Float64             # conservative displacement s = h(u)·(1+rel) + margin ≥ 0 (Inf if :impossible)
    raw::Float64                   # h(u) at the returned direction (no margin); negative ⇒ certified separation
    direction::Union{Nothing,Vector{Float64}}   # unit u: A moves along −u, B along +u
    optimality::Symbol             # :stationary | :budget | :stalled | :unoptimized
    locally_optimized::Bool
    seed::Symbol                   # :user :locus :complement :fallback
    iterations::Int
    evaluations::Int
    nonsmooth_steps::Int
    gradient_norm::Float64         # stationarity residual (tangential gradient / min-norm bundle element)
    margin::Float64
    locus_distance::Float64        # distance between the reference loci (NaN if not used)
    dim::Int
    note::String
end

function _disp(status, N; magnitude = NaN, raw = NaN, dir = nothing, opt = :unoptimized, seed = :none, iters = 0, evals = 0, ns = 0, gn = NaN,
               margin = 0.0, ld = NaN, note = "")
    Displacement(status, Float64(magnitude), Float64(raw), dir === nothing ? nothing : Vector{Float64}(dir), opt, opt === :stationary, seed,
                 iters, evals, ns, Float64(gn), Float64(margin), Float64(ld), N, note)
end

# deterministic fallback directions
function _fallback_seeds(n::Int)
    S = Vector{Vector{Float64}}()
    if n ≤ 4
        for c in Iterators.product(ntuple(_ -> (-1.0, 0.0, 1.0), n)...)
            v = collect(c); any(!iszero, v) && push!(S, normalize(v))
        end
    else
        for i in 1:n, s in (1.0, -1.0); e = zeros(n); e[i] = s; push!(S, e); end
    end
    sort!(S; by = v -> (count(!iszero, v), v))                    # axes first, then edges, then corners
    S
end

# minimum-norm element of conv{columns of Q} (a tiny QP)
function _min_norm_hull(Q::Matrix{Float64})
    k = size(Q, 2); k == 1 && return Q[:, 1]
    G = Q' * Q
    A = vcat(ones(1, k), -Matrix{Float64}(I, k, k)); b = vcat(1.0, zeros(k))
    sol = _solve_conic(sparse(triu(2 * G)), zeros(k), sparse(A), b, [Clarabel.ZeroConeT(1), Clarabel.NonnegativeConeT(k)]; tols = (1e-12, 1e-10, 1e-8))
    if !_solved(sol)
        return Q[:, argmin([norm(Q[:, j]) for j in 1:k])]
    end
    Q * Vector{Float64}(sol.x)
end

"""
    conservative_displacement(A, B; locus_a, locus_b, direction, margin, rel_margin, tol, max_iter, max_evals, extra_starts)

Conservative collision-recovery displacement for two bounding `MinkowskiSum`s (see the file header for the guarantee).
`A` is displaced by `−s·u`, `B` by `+s·u` (or split by weights, see `separation_translations`).
- `locus_a`, `locus_b`: optional `ReferenceLocus`es; their relative geometry gives the initial direction. `direction`: optional user seed.
- `margin` (absolute, default 1e-8) and `rel_margin` (default 0): safety added to the evaluated support value.
- `tol`: stationarity tolerance on the tangential (sub)gradient; `max_iter` per start and `max_evals` in total bound the effort;
  `extra_starts`: additional seeds beyond the first successful one (more effort, possibly smaller displacement).
"""
function conservative_displacement(A::MinkowskiSum, B::MinkowskiSum; locus_a::Union{Nothing,ReferenceLocus} = nothing,
                                   locus_b::Union{Nothing,ReferenceLocus} = nothing, direction::Union{Nothing,AbstractVector} = nothing,
                                   margin::Real = 1e-8, rel_margin::Real = 0.0, tol::Real = 1e-8, max_iter::Integer = 100,
                                   max_evals::Integer = 2000, extra_starts::Integer = 0)
    N = ambient_dim(A); N == ambient_dim(B) || throw(DimensionMismatch("sums live in different spaces"))
    (margin ≥ 0 && rel_margin ≥ 0) || throw(ArgumentError("margins must be nonnegative"))
    ha, hb = affine_hull(A), affine_hull(B)
    (ha === nothing || hb === nothing) && return _disp(:empty, N; note = "a bound is empty")
    rank(hcat(ha.basis, hb.basis); atol = 1e-9) < N &&
        return _disp(:degenerate, N; magnitude = margin, raw = 0.0, note = "A′ − B′ is lower-dimensional: empty interior, no interior overlap to eliminate")

    evals = Ref(0)
    sep = Ref{Any}(nothing)                                              # a separating direction (h < 0), if one is ever seen
    ev(u) = (evals[] += 1; r = support_difference(A, B, u); r.status === :finite && r.value < -1e-9 * max(1.0, norm(r.point)) && (sep[] = (copy(u), r.value)); r)

    # ---- finite-support domain K (built lazily: bounded problems never pay for it) ----------------------------
    domain = Ref{Any}(nothing)
    function build_domain()
        nv = N; Id = Matrix{Float64}(I, N, N)                            # variables: u, then (λ, μ) of every term of A and B
        spec = Tuple{MinkowskiSum,Float64}[(A, 1.0), (B, -1.0)]
        rows = Tuple{Int,Matrix{Float64},Matrix{Float64},Matrix{Float64},Int,Int}[]
        total_eq = 0; offs = Int[]
        for (P, sg) in spec, f in P.terms
            R = recession_cone(f); m, pe = size(R.G, 1), size(R.E, 1)
            push!(rows, (N, sg .* Id, R.G, R.E, m, pe)); total_eq += N; nv += m + pe
        end
        Aeq = zeros(total_eq, nv); Ale = Matrix{Float64}(undef, 0, nv); pos = N; row = 0; nonneg = Int[]
        for (n_i, Mt, G, E, m, pe) in rows
            Aeq[row+1:row+n_i, 1:N] = Mt                                  # (±Mᵀu)ᵢ − Gᵀλ − Eᵀμ = 0
            m > 0 && (Aeq[row+1:row+n_i, pos+1:pos+m] = -G'; append!(nonneg, pos+1:pos+m))
            pe > 0 && (Aeq[row+1:row+n_i, pos+m+1:pos+m+pe] = -E')
            row += n_i; pos += m + pe
        end
        NL = length(nonneg)
        Ac = vcat(Aeq, NL > 0 ? hcat([(j == nonneg[k] ? -1.0 : 0.0) for k in 1:NL, j in 1:nv]) : zeros(0, nv))
        bc = zeros(size(Ac, 1))
        cones = Clarabel.SupportedCone[Clarabel.ZeroConeT(total_eq)]; NL > 0 && push!(cones, Clarabel.NonnegativeConeT(NL))
        Pm = spzeros(nv, nv); for i in 1:N; Pm[i, i] = 1.0; end
        domain[] = (P = Pm, A = sparse(Ac), b = bc, cones = cones, nv = nv)
    end
    function project(u)                                                  # Euclidean projection onto K; nothing if it is 0
        domain[] === nothing && build_domain()
        d = domain[]; q = zeros(d.nv); q[1:N] = -u
        sol = _solve_conic(d.P, q, d.A, d.b, d.cones; tols = (1e-12, 1e-10, 1e-8))
        _solved(sol) || return nothing
        v = Vector{Float64}(sol.x[1:N]); norm(v) ≤ 1e-9 * max(norm(u), 1.0) && return nothing
        normalize(v)
    end
    kbar = Ref{Any}(:unset)                                              # an interior point of K (sum of projections of ±eᵢ); nothing if K = {0}
    function interior_point()
        if kbar[] === :unset
            acc = zeros(N); ps = Vector{Float64}[]
            for i in 1:N, sg in (1.0, -1.0)
                e = zeros(N); e[i] = sg; p = project(e); p !== nothing && (acc .+= p; push!(ps, p))
            end
            # K = {0} iff every ±eᵢ projects to 0. The sum lies in K but cancels for cones with lineality (a plane, a line): then any
            # single projection (an element of K) serves.
            kbar[] = isempty(ps) ? nothing : (norm(acc) > 1e-9 ? normalize(acc) : ps[1])
        end
        kbar[]
    end
    # bring a seed into the finite-support domain: (:ok, u, r) | (:outside,) | (:impossible,) | (:failed,) | (:empty,)
    function find_finite(u)
        u = normalize(u)
        evals[] ≥ max_evals && return (:outside,)
        r = ev(u)
        r.status === :finite && return (:ok, u, r)
        r.status === :empty && return (:empty,)
        r.status === :failed && return (:failed,)
        kb = interior_point(); kb === nothing && return (:impossible,)
        p = project(u)
        p === nothing && (p = kb)                                        # the seed lies in −K°: use the domain's own interior point
        for t in (0.0, 0.05, 0.2, 0.5, 1.0)
            evals[] ≥ max_evals && break
            w = normalize((1 - t) * p + t * kb); r = ev(w)
            r.status === :finite && return (:ok, w, r)
            r.status === :failed && return (:failed,)
        end
        (:outside,)
    end
    function descend(u, r)
        h = r.value; iters = 0; ns = 0; gn = Inf; opt = :budget; ε = 1e-4; probes = 0; ε_floor = 1e-6   # sampling radius: below ~1e-6 LP noise swamps the bundle
        while iters < max_iter && evals[] < max_evals
            sep[] !== nothing && break
            iters += 1
            nsm = nonsmooth(r)
            if nsm
                ns += 1
                T = Matrix{Float64}(nullspace(reshape(u, 1, :); atol = 1e-12)); pts = [r.point]
                for j in 1:size(T, 2), s in (1.0, -1.0)
                    rr = ev(normalize(u + s * ε * T[:, j])); rr.status === :finite && push!(pts, rr.point)
                end
                gT = _min_norm_hull(reduce(hcat, [p - dot(p, u) * u for p in pts]))
            else
                g = r.point; gT = g - dot(g, u) * u
            end
            gn = norm(gT)
            if gn ≤ tol * max(1.0, abs(h))
                if nsm && ε > ε_floor; ε = max(ε / 10, ε_floor); continue; end       # refine the sampling radius before trusting stationarity
                # stationarity can also sit at a maximum or saddle (e.g. a locus pointing exactly away): probe the neighbourhood once
                moved = false
                if probes < 3
                    probes += 1
                    T = Matrix{Float64}(nullspace(reshape(u, 1, :); atol = 1e-12))
                    for j in 1:size(T, 2), s in (1.0, -1.0)
                        evals[] ≥ max_evals && break
                        u2 = normalize(u + s * 1e-3 * T[:, j]); r2 = ev(u2)
                        if r2.status === :finite && r2.value < h - 1e-10 * max(1.0, abs(h))
                            u, r, h, moved = u2, r2, r2.value, true; break
                        end
                    end
                end
                moved && continue
                opt = :stationary; break
            end
            α = 1.0; ok = false
            for _ in 1:25
                evals[] ≥ max_evals && break
                u2 = normalize(u - α * gT); r2 = ev(u2)
                if r2.status === :infinite                               # stepped out of the finite-support domain: project back onto K
                    u2p = project(u2)
                    if u2p !== nothing; u2 = u2p; r2 = ev(u2); end
                end
                if r2.status === :finite && r2.value ≤ h - 1e-4 * α * gn^2
                    u, r, h, ok = u2, r2, r2.value, true; break
                end
                α /= 2
            end
            if !ok
                nsm && ε > ε_floor && (ε = max(ε / 10, ε_floor); continue)          # a coarse bundle can mislead: retry with a finer one
                opt = nsm ? :stationary : :stalled; break                           # nonsmooth: no descent at sampling resolution ε_floor (Goldstein ε-stationary)
            end
        end
        (u = u, h = h, opt = opt, iters = iters, ns = ns, gn = gn)
    end

    # ---- seeds -----------------------------------------------------------------------------------------------
    seeds = Tuple{Symbol,Vector{Float64}}[]; ld = NaN; note = ""
    direction !== nothing && norm(direction) > 0 && push!(seeds, (:user, Vector{Float64}(direction)))
    if locus_a !== nothing && locus_b !== nothing
        lv = locus_vector(locus_a, locus_b); ld = lv.distance
        if lv.status === :separate
            push!(seeds, (:locus, lv.vector))                            # from A's locus towards B's: A moves along −u
        elseif lv.status === :coincident
            for c in _locus_complement(locus_a, locus_b); push!(seeds, (:complement, c)); end
            note = "reference loci meet: no locus direction (zero vector not normalised); complement directions used"
        else
            note = "locus closest-point solve failed; fallback seeds used"
        end
    end
    fallback = [(:fallback, s) for s in _fallback_seeds(N)]

    best = nothing; successes = 0; failed = false; empty = false; outside = false; impossible = false
    function attempt(label, s)
        res = find_finite(s)
        res[1] === :impossible && (impossible = true; return)
        res[1] === :failed && (failed = true; return)
        res[1] === :empty && (empty = true; return)
        res[1] === :outside && (outside = true; return)
        _, u, r = res
        d = descend(u, r); successes += 1
        (best === nothing || d.h < best.h) && (best = (; d..., label = label))
    end
    domain_seed = Tuple{Symbol,Vector{Float64}}[]                        # last resort: the domain's own interior point
    for (label, s) in vcat(seeds, fallback)
        impossible && break
        (successes ≥ 1 + extra_starts || sep[] !== nothing) && break
        label === :fallback && successes ≥ 1 + extra_starts && break
        evals[] ≥ max_evals && break
        attempt(label, s)
    end

    if best === nothing && !impossible && !failed && !empty && sep[] === nothing && domain[] !== nothing
        kb = interior_point()
        kb === nothing ? (impossible = true) : attempt(:domain, kb)
    end
    if sep[] !== nothing
        u, h = sep[]
        return _disp(:separated, N; magnitude = 0.0, raw = h, dir = u, seed = best === nothing ? :fallback : best.label, evals = evals[],
                     margin = margin, ld = ld, note = "h_D′(u) < 0: the bounds' interiors are disjoint (certified separating direction)")
    end
    impossible && return _disp(:impossible, N; magnitude = Inf, evals = evals[], margin = margin, ld = ld,
                               note = "the finite-support domain polar(rec A′ − rec B′) is {0}: every direction has infinite support, no translation separates the bounds")
    empty && return _disp(:empty, N; evals = evals[], note = "a bound is empty")
    if best === nothing
        return failed ? _disp(:failed, N; evals = evals[], ld = ld, note = "support evaluations failed") :
               _disp(:no_direction_found, N; evals = evals[], ld = ld,
                     note = outside ? "no finite-support direction found: seeds projected onto the recession-cone domain still had infinite support (e.g. a non-closed barrier cone) or the budget ended; NOT a proof of impossibility" :
                                      "no finite-support direction found within the budget")
    end
    mag = max(best.h, 0.0) * (1 + rel_margin) + margin
    _disp(:sufficient, N; magnitude = mag, raw = best.h, dir = best.u, opt = best.opt, seed = best.label, iters = best.iters, evals = evals[], ns = best.ns,
          gn = best.gn, margin = margin, ld = ld, note = note)
end
conservative_displacement(A::Term, B::Term; kwargs...) = conservative_displacement(MinkowskiSum(A), MinkowskiSum(B); kwargs...)

# ---- distribution between the two objects -------------------------------------------------------------

struct Translations
    t_a::Vector{Float64}
    t_b::Vector{Float64}
    magnitude::Float64
    direction::Union{Nothing,Vector{Float64}}
    w_a::Float64
    w_b::Float64
end

"""
    separation_translations(d; mode = :a, inv_mass_a, inv_mass_b)

Distribute a `Displacement` between the objects. Default: A moves, B stays — `t_A = −s u`; `mode = :b` gives `t_B = +s u`;
`mode = :equal` gives `∓s u/2`. With inverse masses (static body = 0): `w_A = ia/(ia+ib)`, `t_A = −w_A s u`, `t_B = +w_B s u`,
so `t_B − t_A = s u` always (only the relative translation enters the guarantee). Zero vectors unless `d.status === :sufficient`.
"""
function separation_translations(d::Displacement; mode::Symbol = :a, inv_mass_a::Union{Nothing,Real} = nothing, inv_mass_b::Union{Nothing,Real} = nothing)
    if inv_mass_a !== nothing || inv_mass_b !== nothing
        ia, ib = Float64(something(inv_mass_a, 1.0)), Float64(something(inv_mass_b, 1.0))
        (ia ≥ 0 && ib ≥ 0) || throw(ArgumentError("inverse masses must be nonnegative"))
        ia + ib > 0 || throw(ArgumentError("both bodies are static: nothing can move"))
        wa = ia / (ia + ib)
    else
        mode in (:a, :b, :equal) || throw(ArgumentError("mode must be :a, :b or :equal"))
        wa = mode === :a ? 1.0 : mode === :b ? 0.0 : 0.5
    end
    z = zeros(d.dim)
    (d.status === :sufficient && d.direction !== nothing) || return Translations(z, copy(z), d.status === :separated || d.status === :degenerate ? 0.0 : d.magnitude, d.direction, wa, 1 - wa)
    Translations(-wa * d.magnitude * d.direction, (1 - wa) * d.magnitude * d.direction, d.magnitude, d.direction, wa, 1 - wa)
end

function Base.show(io::IO, ::MIME"text/plain", d::Displacement)
    println(io, "Displacement: ", d.status)
    if d.status === :sufficient
        println(io, "  magnitude  ", round(d.magnitude; sigdigits = 8), "   (h = ", round(d.raw; sigdigits = 8), " + margin ", d.margin, ")")
        println(io, "  u          ", round.(d.direction; sigdigits = 5), "   (A moves along −u, B along +u)")
        println(io, "  search     ", d.optimality, d.locally_optimized ? " (local minimiser, not claimed global)" : " (limited)", ", seed ", d.seed,
                ", ", d.iterations, " iterations, ", d.evaluations, " evaluations, ", d.nonsmooth_steps, " nonsmooth steps, residual ", round(d.gradient_norm; sigdigits = 3))
    end
    isempty(d.note) || print(io, "  note       ", d.note)
end
Base.show(io::IO, d::Displacement) = print(io, "Displacement(", d.status, ", magnitude=", round(d.magnitude; sigdigits = 6), ")")
