# QuadricLab.jl (Phases 1–4)

Interactive geometry lab for intersections of convex quadratic regions `{x : q(x) ≤ 0}`,
rendered by analytic ray–quadric intersection in GLSL.

```julia
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'      # CPU tests
QUADRICLAB_GL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'   # + OpenGL tests (needs a GL 3.3 context)
julia --project examples/demo.jl                                   # interactive demo window
```

```julia
using QuadricLab, Colors

ball  = sphere(origin, 2.0)                       # bare Quadric
tube  = cylinder(zaxis, 1.0, Style(albedo = colorant"seagreen"))   # trailing Style → Constraint
floor_ = halfspace(up, 0.0)                       # solid region z ≤ 0
obj = intersection(ball, tube, floor_)            # GeoObject; order = seam priority
[0, 0, -1.0] in obj                               # membership
feasibility(obj)                                  # :interior / :boundary / :empty (+ interior point, margin)
is_bounded(obj)                                   # recession-cone test; unbounded geometry is never clipped
ray_hit(obj, [0, 0, 5.0], [0, 0, -1.0])           # CPU reference of what the shader computes
render(Scene(obj, sphere([4, 0, 0.0], 1.0)))      # union of objects; mouse: orbit / pan / zoom, Esc closes
render_image(obj; width = 800, height = 600)      # offscreen → Matrix{RGB}; write_ppm("out.ppm", img)
```

## Conventions (Phase 1)
- **Quadric**: `Q(X)=XᵀMX`, `M=[A b; bᵀ c]`, chart `w=1`: `q(x)=xᵀAx+2bᵀx+c`. Region `q ≤ 0`; convex iff `A ⪰ 0`.
- **Normals**: `∇q = 2(Ax+b)`, outward on the boundary.
- **Sphere / cylinder / halfspace**: cylinder axis normalised (optional `point=`); halfspace `n̂·x ≤ offset`, outward unit `n̂`.
- **Transforms**: `M' = H⁻ᵀ M H⁻¹`, so `Q'(HX) = Q(X)`. `transform` is affine-only; `projective_transform` is explicit and carries no Euclidean or convexity guarantees.

## Phase 2 additions
- **Style** `(albedo, roughness, metallic, opacity, texture, mapping)`; `Texture` + `checker`; mappings `PlanarMapping`, `CylindricalMapping`, `SphericalMapping` (frame + scale, transformed together with their constraint).
- **Constraint** = quadric + Style + `id`. Ids are stable under `transform` (they name "this surface"); `reid` mints a fresh one. Collision code reads only `.quadric`.
- **GeoObject** = conjunction; **Scene** = union of objects (nearest visible surface wins).
- **Feasibility**: minimise `s` subject to `q̃ᵢ(x) ≤ s` (each quadric scaled by its largest entry) as a second-order-cone program in Clarabel; `s ≥ -1` only normalises the margin and never bounds the geometry. `margin = -s*`.
  Status `:interior` (margin > tol), `:boundary` (|margin| ≤ tol), `:empty`, or `:numerical_failure` (solver did not converge — distinct from a mathematical answer).
- **Boundedness**: recession cone `{d : Aᵢd = 0, bᵢ·d ≤ 0 ∀i}` is trivial ⇔ bounded (small LPs). Empty ⇒ reported bounded.
- **Renderer**: fullscreen-triangle fragment shader (`shaders/quadric.{vert,frag}`). Per pixel and object: roots of each constraint → validity against all other inequalities (`qⱼ ≤ tol·‖∇qⱼ‖`, a distance-like tolerance) → nearest candidate; ties within tolerance go to the lowest constraint index. GGX shading, up to 6 translucent layers, no bounding volume.
- `src/rendering/raycast.jl` mirrors the shader in Float64 and is what the tests compare against.

## Phase 3: distance and contact
```julia
a = Constraint(sphere(origin, 1.0)); b = Constraint(sphere([5, 0, 0.0], 1.5))   # Constraints keep their ids
distance(a, b)          # DistanceResult: distance, witness points, normal (A→B), active ids, cones, solver info
contact(a, b)           # ContactResult: :separated | :touching | :intersecting
normal_cone(obj, p)     # active constraints at p with their gradients
```
- **Distance** is the convex QCQP `min ½‖x−y‖²` over both objects' constraints (six variables, Clarabel SOCP). Overlap is decided first
  (feasibility of A∩B): interiors overlap ⇒ `:intersecting`, distance **0** (never negative; penetration is estimated separately, see Phase 4);
  boundaries only ⇒ `:touching`; otherwise `:separated` and the QCQP is solved.
  Other statuses: `:empty_object` (an object has no points) and `:numerical_failure` (solver did not converge — not a mathematical answer).
- **Normal** `n = (y−x)/d` points from A to B and satisfies `n ∈ N_A(x)`, `−n ∈ N_B(y)`; `cone_residual` is the non-negative-least-squares
  residual of that statement over the active gradients (≈ 0 certifies optimality). For touching objects `n` comes from an LP over the active gradients.
- **Active constraints** at a point: `|qᵢ|/max(‖∇qᵢ‖,1) ≤ 1e-6`. All are kept with their ids (edges, corners, coincident surfaces).
  A constraint with zero gradient on its boundary (e.g. `{x² ≤ 0}`) is flagged `degenerate`: it gives no generator, so no certificate.
- **Witness set**: `:unique`, `:nonunique` (finite face), `:unbounded` (plane on plane, cylinder lying on a plane) or `:unknown`
  (auxiliary solves failed — not a mathematical answer). The optimal witnesses are exactly `A ∩ (B + w)` with `w = x*−y*` unique, so
  the question is the size of a touching set. It is measured at two relaxation levels ε and 10ε: a point contact shrinks like √ε (ratio ≈ 0.32),
  a contact face keeps its size (ratio ≈ 1); the threshold is 0.6. This is scale-free (no tolerance to tune), but it is a numerical
  classification: a face much smaller than √(ε·R) is indistinguishable from a point.
- **Accuracy independent of position**: `distance` solves in a local frame (both objects translated so a point between them, then the
  witnesses, is the origin) and translates the results back, because the solver's tolerances are relative to each quadric's largest
  entry, which far from the origin is dominated by ‖centre‖². Solves fall back through tolerances 1e-12 → 1e-10 → 1e-8.
- `contact` of intersecting objects returns the most interior point of A∩B and no normal; the separating direction comes from the Phase 4 bounding-product penetration estimate.
- Constraint ids survive `transform`, so `distance(A, transform(A, H))` works (shared ids are handled).

## Phase 4: support-based conservative collision recovery (`src/bounds/`)
Two independent things: the exact **narrow phase** (`distance` / `contact` on the `GeoObject`, unchanged) and a **recovery displacement** computed from
separately supplied outer **bounds** (`MinkowskiSum`s; nothing derives bounds from quadrics). Recovery needs no narrow-phase validation: the geometry is the guarantee.

**Guarantee.** With bounds A′ ⊇ A, B′ ⊇ B and D′ = A′ − B′ ⊇ D = A − B, `h(u) = h_{A′}(u) + h_{B′}(−u)`. Translating A by `t` (B fixed) has interior overlap iff `−t ∈ int D`.
For any unit `u` with finite `h(u)`, `int D′ ⊂ {x : u·x < h(u)}`, so `t_A = −s·u` with `s ≥ h(u)` removes interior overlap — **for every direction, not only a minimiser**
(`t_B = +s·u` moves B instead; weighted: `t_A = −w_A s u`, `t_B = +w_B s u`, only `t_B − t_A = s·u` matters). In terms of A's own direction `v = −u`: `s = h_{A′}(−v) + h_{B′}(v)`.
Invertible affine maps of a term are exact (`h_{LK+c}(u) = h_K(Lᵀu) + c·u`). Any `u` with `h(u) < 0` certifies that the bounds' interiors are disjoint. The optimiser only *reduces* `s`; the only caveat is solver tolerance, which `margin` covers.
This is not an exact penetration-depth solver, and a stationary point is never claimed to be globally minimal.

**Geometry**
- `PolyhedronTerm {Gx ≤ g, Ex = e}` and `QuadricTerm {xᵀAx + 2bᵀx + c ≤ 0, Ex = e}`, `A ⪰ 0`, degenerate allowed (`A = 0`, points, flat disks, cylinders, paraboloids, parabolic slabs). Constructors: `box_term`, `interval_term`, `halfspace_term`, `affine_term`, `whole_space_term`, `ball_term`, `ellipsoid_term`, `QuadricTerm(::Quadric)`.
- `MinkowskiSum(f₁, f₂, …)`: terms in one common space ℝᴺ; the set is their Minkowski sum and only the support function is used: `h_K(u) = Σ hᵢ(u)`.
  A Cartesian product of terms on separate coordinates is the special case of lower-dimensional terms embedded in ℝᴺ (a disk × a line = a flat disk + a line; the optional
  `embed_term(f, indices, N)` places a term on coordinates `indices`). Nothing tracks which coordinates a term uses and terms may overlap — the maths is the same.
  `transform(P, H)` / `transform(P, L, t)` maps every term by an invertible `L` (exact, general rotations included); projective maps and singular `L` are rejected.
- `support(f, u)` → `SupportResult`: `:finite` / `:infinite` / `:empty` / `:failed` (+∞ is never a failure), value, support point, **exposed-face dimension** (`nonsmooth(r)`), active polyhedron rows, a recession ray when one exists, and the per-term results of a product/difference (`r.parts`).
  Quadrics are analytic; polyhedra use an LP. `face_dim` of a sum/difference is an upper bound.
- `ReferenceLocus` (`point_locus`, `segment_locus`, `halfline_locus`, `line_locus`, `affine_locus`; transformable): a positional reference, independent of the support function. `locus_vector` gives the closest-point vector (unique even for parallel loci). If it is zero (meeting/coincident loci) it is never normalised; the orthogonal complement of the loci directions supplies seeds instead.

**Optimiser** (`conservative_displacement(A, B; locus_a, locus_b, direction, margin, rel_margin, tol, max_iter, max_evals, extra_starts)`) — one procedure for every term type:
- minimise `h` over the unit sphere; seeds: user direction, locus direction, locus complement, then deterministic fallbacks (only if needed or `extra_starts` more are requested); the loci are guides, not constraints (an escape probe leaves a stationary maximum).
- differentiable directions (`face_dim = 0`): Riemannian step `g_T = g − (g·u)u`, `g` = support point. **Nonsmooth** directions (`face_dim > 0`): gradient sampling over the exposed face — support points at `u` and `u ± ε·tangents`; the min-norm element of their projected hull is the descent direction (zero ⇒ stationary, e.g. at a facet normal). Sampling radius 1e-4 → 1e-6, so "stationary" there is ε-stationary (excess ≲ 1e-6, never below the minimum).
- unbounded sets: finite support needs `u ∈ polar(rec D′)`; `K = {u : u ∈ polar(rec Kᵢᴬ) ∀i, −u ∈ polar(rec Kⱼᴮ) ∀j}` is a polyhedral cone, `polar{Gd ≤ 0, Ed = 0} = {Gᵀλ + Eᵀμ, λ ≥ 0}`, so projecting a seed onto K is one small QP (exact for slabs, rays, lineality). `K = {0}` ⇒ `:impossible` (certified). Non-closed barrier cones (paraboloids) can still give +∞ on K's boundary; seeds are blended toward an interior point.
- result `Displacement`: `status` ∈ `:sufficient` (finite conservative `magnitude = h·(1+rel) + margin`), `:separated` (certified), `:degenerate` (empty interior: nothing to eliminate, magnitude = margin), `:empty`, `:impossible` (certified), `:no_direction_found` (not a proof), `:failed` (numerical); plus `direction u*`, `optimality` (`:stationary | :budget | :stalled`), `locally_optimized`, seed, iterations/evaluations/nonsmooth steps, residual.
- `separation_translations(d; mode = :a | :b | :equal, inv_mass_a, inv_mass_b)`: A moves along `−u`, B along `+u`; static bodies have inverse mass 0.

## Known limitations
- Textures are sampled without mip-mapping (aliasing at grazing angles / distance).
- Shader arithmetic is Float32; very large or very small scenes may need rescaling.
- `render(...; block=false)` runs the loop as an `@async` task (default in the REPL); GLFW is only exercised from that one task.
- Penetration depth, diagnostics overlay, robustness/performance work: Phases 4–5.
