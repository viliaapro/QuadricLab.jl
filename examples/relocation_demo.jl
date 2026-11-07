# Collision / relocation demo.
#
#   julia --project examples/relocation_demo.jl [outdir]      (needs an OpenGL 3.3 context; `xvfb-run` works)
#
# Every piece has
#   flesh   translucent body, an intersection of convex quadrics      (exact geometry: the narrow phase sees only this)
#   marrow  its reference locus (a segment), drawn as the bright core
#   space   its "personal space": a Minkowski sum built FROM the marrow (capsule = marrow ⊕ ball, puck = marrow ⊕ flat disk).
#           It is an outer bound of the flesh (checked below) and the only thing the recovery step looks at.
# Relocation: the exact narrow phase (`distance`) finds overlapping flesh; `conservative_displacement` on the two personal spaces,
# seeded by the marrows, gives a displacement that is sufficient for the bounds (hence for the flesh); it is split by inverse
# mass and applied. The floor is an infinite halfspace (unbounded geometry) with infinite mass.
using QuadricLab, Colors, LinearAlgebra, Random, Printf

const UP = [0.0, 0.0, 1.0]
const OUT = isempty(ARGS) ? "relocation_out" : ARGS[1]
mkpath(OUT)

# ---- building blocks -------------------------------------------------------------------------------------
marrow_style = Style(albedo = colorant"orangered", roughness = 0.25, metallic = 0.3)
bubble_style = Style(albedo = colorant"lightskyblue", roughness = 0.2, opacity = 0.10)
flesh_style(c) = Style(albedo = c, roughness = 0.3, opacity = 0.34)

"A solid rod from a to b of radius r: cylinder ∩ two halfspaces."
function tube(a, b, r, st)
    d = normalize(b - a)
    intersection(cylinder(d, r, st; point = a), halfspace(d, dot(d, b), st), halfspace(-d, -dot(d, a), st))
end
"Capsule = tube + two end balls (drawn as three translucent bodies)."
capsule(a, b, r, st) = [tube(a, b, r, st), intersection(sphere(a, r, st)), intersection(sphere(b, r, st))]
ellipsoid(a, b, st) = transform(Constraint(sphere(zeros(3), 1.0), st), scaling([a, b, b]))

struct Piece
    name::String
    flesh::GeoObject{Float64}
    marrow::ReferenceLocus
    marrow_vis::GeoObject{Float64}
    space::MinkowskiSum
    bubble::Vector{GeoObject{Float64}}
    mass::Float64
end

function bean(c)
    fs = flesh_style(c); a, b = [-0.8, 0, 0], [0.8, 0, 0]; m = segment_locus(a, b)
    Piece("bean", intersection(ellipsoid(1.4, 0.6, fs)), m, tube(a, b, 0.045, marrow_style),
          MinkowskiSum(m.set, ball_term(zeros(3), 0.65)), capsule(a, b, 0.65, bubble_style), 2.1)
end
function disc(c)
    fs = flesh_style(c); a, b = [0, 0, -0.45], [0, 0, 0.45]; m = segment_locus(a, b)
    Piece("disc", intersection(sphere(zeros(3), 0.95, fs), halfspace(UP, 0.45, fs), halfspace(-UP, 0.45, fs)), m, tube(a, b, 0.045, marrow_style),
          MinkowskiSum(m.set, ellipsoid_term(zeros(3), Matrix(Diagonal([0.95^2, 0.95^2, 0.0])))), [tube(a, b, 0.95, bubble_style)], 2.4)
end
function rod(c)
    fs = flesh_style(c); a, b = [0, -0.85, 0], [0, 0.85, 0]; m = segment_locus(a, b)
    Piece("rod", intersection(cylinder([0.0, 1, 0], 0.35, fs), sphere(zeros(3), 1.25, fs)), m, tube(a, b, 0.045, marrow_style),
          MinkowskiSum(m.set, ball_term(zeros(3), 0.5)), capsule(a, b, 0.5, bubble_style), 1.4)
end
function bell(c)                                                # paraboloid x² + y² + 0.8 z ≤ 0 cut at z ≥ −1: a bell, apex up
    fs = flesh_style(c); a, b = [0, 0, -1.0], [0, 0, 0.0]; m = segment_locus(a, b)
    par = Quadric([1.0 0 0 0; 0 1 0 0; 0 0 0 0.4; 0 0 0.4 0])
    Piece("bell", intersection(Constraint(par, fs), halfspace(-UP, 1.0, fs)), m, tube(a, b, 0.045, marrow_style),
          MinkowskiSum(m.set, ellipsoid_term(zeros(3), Matrix(Diagonal([0.9^2, 0.9^2, 0.0])))), [tube(a, b, 0.9, bubble_style)], 1.0)
end

# ---- bodies: a piece at a pose -------------------------------------------------------------------------------
mutable struct Body
    p::Piece
    H::Matrix{Float64}
end
flesh(b) = transform(b.p.flesh, b.H)
space(b) = transform(b.p.space, b.H)
locus(b) = transform(b.p.marrow, b.H)
shift!(b, t) = (b.H = Matrix{Float64}(translation(t)) * b.H; b)

const FLOOR_STYLE = Style(albedo = colorant"white", roughness = 0.9, texture = checker(colorant"gray92", colorant"gray55"; n = 2),
                          mapping = PlanarMapping(u = (2, 0, 0), v = (0, 2, 0)))
const FLOOR = intersection(halfspace(UP, 0.0, FLOOR_STYLE))
const FLOOR_SPACE = MinkowskiSum(halfspace_term(UP, 0.0))                       # unbounded bound
const FLOOR_LOCUS = affine_locus(zeros(3), [1.0 0; 0 1; 0 0])                    # the floor plane
collides(a, b) = distance(a, b).status === :intersecting

# ---- the relocation loop -----------------------------------------------------------------------------------------
function relocate!(bodies; sweeps = 20, margin = 0.02)
    history = [[copy(b.H) for b in bodies]]; resolved = 0; events = 0
    for sweep in 1:sweeps
        t0 = time(); ncoll = 0; maxd = 0.0
        for b in bodies                                                          # against the floor (static)
            collides(flesh(b), FLOOR) || continue
            ncoll += 1; events += 1
            d = conservative_displacement(space(b), FLOOR_SPACE; locus_a = locus(b), locus_b = FLOOR_LOCUS, margin = margin)
            d.status === :sufficient || (@warn "floor recovery" b.p.name d.status; continue)
            tr = separation_translations(d; inv_mass_a = 1.0, inv_mass_b = 0.0)
            shift!(b, tr.t_a); maxd = max(maxd, norm(tr.t_a)); resolved += !collides(flesh(b), FLOOR)
        end
        for i in eachindex(bodies), j in i+1:length(bodies)                    # pairs
            bi, bj = bodies[i], bodies[j]
            collides(flesh(bi), flesh(bj)) || continue
            ncoll += 1; events += 1
            d = conservative_displacement(space(bi), space(bj); locus_a = locus(bi), locus_b = locus(bj), margin = margin)
            d.status === :sufficient || (@warn "pair recovery" bi.p.name bj.p.name d.status; continue)
            tr = separation_translations(d; inv_mass_a = 1 / bi.p.mass, inv_mass_b = 1 / bj.p.mass)
            shift!(bi, tr.t_a); shift!(bj, tr.t_b); maxd = max(maxd, norm(tr.t_a), norm(tr.t_b))
            resolved += !collides(flesh(bi), flesh(bj))                          # the guarantee: this very pair must now be free
        end
        push!(history, [copy(b.H) for b in bodies])
        @printf("sweep %2d: %2d overlapping pairs found, largest shift %.3f  (%.1fs)\n", sweep, ncoll, maxd, time() - t0)
        ncoll == 0 && break
    end
    (history = history, resolved = resolved, events = events)
end

# ---- scene / rendering -----------------------------------------------------------------------------------------------
function scene(bodies; bubbles = false)
    objs = GeoObject{Float64}[FLOOR]
    for b in bodies
        append!(objs, [transform(b.p.flesh, b.H), transform(b.p.marrow_vis, b.H)])
        bubbles && append!(objs, [transform(o, b.H) for o in b.p.bubble])
    end
    Scene(objs...)
end
const CAM = Camera(target = (0, 0, 0.6), distance = 8.5, yaw = -0.85, pitch = 0.5, fov = deg2rad(40))
snap(bodies, H) = [Body(b.p, h) for (b, h) in zip(bodies, H)]

# ---- main -----------------------------------------------------------------------------------------------------------------
function main()
    rng = MersenneTwister(11)
    pieces = [bean(colorant"lightcoral"), bean(colorant"mediumpurple"), disc(colorant"mediumaquamarine"), disc(colorant"goldenrod"),
              rod(colorant"lightskyblue"), rod(colorant"yellowgreen"), bell(colorant"hotpink")]
    println("containment of flesh by personal space (support-function check):")
    for p in unique(q -> q.name, pieces)
        v = bound_violation(p.flesh, p.space)
        @printf("  %-5s max h_flesh − h_space = %9.2e  %s\n", p.name, v.violation, v.violation ≤ 1e-6 ? "contained" : "NOT CONTAINED")
        v.violation ≤ 1e-6 || error("the personal space of $(p.name) does not contain its flesh")
    end
    bodies = Body[]
    for (k, p) in enumerate(pieces)
        θ = 2π * (k - 1) / length(pieces); pos = [1.1cos(θ), 1.1sin(θ), 0.35 + 0.8rand(rng)] .+ 0.15 .* randn(rng, 3)
        R = Matrix{Float64}(rotation(randn(rng, 3), 2π * rand(rng)))
        push!(bodies, Body(p, Matrix{Float64}(translation(pos)) * R))
    end
    n0 = count(i -> any(j -> j > i && collides(flesh(bodies[i]), flesh(bodies[j])), eachindex(bodies)), eachindex(bodies)) +
         count(b -> collides(flesh(b), FLOOR), bodies)
    println("initially: $n0 bodies in collision (pairs or floor)")
    res = relocate!(bodies)
    println("every handled overlap was resolved by its own displacement: $(res.resolved)/$(res.events)")
    final_free = !any(collides(flesh(bodies[i]), flesh(bodies[j])) for i in eachindex(bodies) for j in i+1:length(bodies)) && !any(b -> collides(flesh(b), FLOOR), bodies)
    println("narrow phase after relocation: ", final_free ? "no overlaps" : "OVERLAPS REMAIN")
    gaps = [distance(flesh(bodies[i]), flesh(bodies[j])).distance for i in eachindex(bodies) for j in i+1:length(bodies)]
    @printf("smallest flesh-to-flesh gap %.3f (personal space leaves slack)\n", minimum(gaps))
    println("rendering ...")
    frames = unique(vcat(1, 2, min(3, length(res.history)), length(res.history) ÷ 2 + 1, length(res.history)))
    for (k, f) in enumerate(frames)
        write_ppm(joinpath(OUT, @sprintf("frame_%02d.ppm", k)), render_image(scene(snap(bodies, res.history[f])); width = 720, height = 440, camera = CAM))
        println("  frame $k (state $f of $(length(res.history)))")
    end
    write_ppm(joinpath(OUT, "final_space.ppm"), render_image(scene(bodies; bubbles = true); width = 720, height = 440, camera = CAM))
    write_ppm(joinpath(OUT, "initial_space.ppm"), render_image(scene(snap(bodies, res.history[1]); bubbles = true); width = 720, height = 440, camera = CAM))
    println("done: ", OUT)
end
main()
