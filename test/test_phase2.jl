using Colors

@testset "QuadricLab Phase 2" begin
    ball = sphere(origin, 2.0)
    tube = cylinder(zaxis, 1.0)
    floor_ = halfspace(up, 0.0)

    @testset "Style" begin
        s = Style(albedo = colorant"tomato", roughness = 0.3, metallic = 1, opacity = 0.5)
        @test s.albedo isa RGB{Float32} && s.opacity == 0.5f0
        @test_throws ArgumentError Style(roughness = 1.5)
        @test_throws ArgumentError Style(texture = checker())          # texture needs a mapping
        s2 = Style(s; opacity = 1.0)
        @test s2.albedo == s.albedo && s2.opacity == 1f0
        @test Style(texture = checker(), mapping = PlanarMapping()).texture isa Texture
        @test_throws ArgumentError CylindricalMapping(axis = [0, 0, 1], ref = [0, 0, 2])
    end

    @testset "Constraint / GeoObject" begin
        c = Constraint(ball, Style(albedo = colorant"red"))
        @test c(origin) ≈ -4
        @test sphere(origin, 2.0, Style()) isa Constraint
        @test cylinder(zaxis, 1.0, Style(); point = [1, 0, 0]) isa Constraint
        c2 = Constraint(ball); c3 = Constraint(ball)
        @test c2.id != c3.id
        @test reid(c2).id != c2.id

        obj = intersection(ball, tube, floor_)
        @test length(obj) == 3 && obj isa GeoObject{Float64}
        @test [0.0, 0.0, -1.0] in obj
        @test !([0.0, 0.0, 1.0] in obj)                 # above the floor
        @test !([1.5, 0.0, -0.5] in obj)                # outside the tube
        @test violation(obj, [0.0, 0.0, -1.0]) ≈ -1.0 atol = 1e-12
        @test is_convex(obj)
        @test length(intersection(obj, sphere(origin, 5.0))) == 4          # flattening
        @test_throws ArgumentError intersection(c, c)                       # duplicate ids
        @test_throws ArgumentError GeoObject{Float64}(Constraint{Float64}[])

        # ids survive transformation, styles follow
        moved = transform(obj, translation([1, 0, 0]))
        @test [c.id for c in moved] == [c.id for c in obj]
        @test [1.0, 0.0, -1.0] in moved

        # active constraints keep identity
        o2 = intersection(sphere(origin, 2.0), halfspace(up, 0.0))
        act = active_constraints(o2, [0.0, 0.0, -2.0])                      # on the sphere only
        @test length(act) == 1 && act[1].id == o2[1].id
        act2 = active_constraints(o2, [1.0, 0.0, 0.0])                      # on the plane only
        @test length(act2) == 1 && act2[1].id == o2[2].id

        # mixed Float32 / Float64 promotes
        q32 = Quadric{Float32}(SMatrix{4,4,Float32,16}(sphere(origin, 1.0).M))
        @test intersection(q32, ball) isa GeoObject{Float64}
    end

    @testset "mapping transforms with its constraint" begin
        m = PlanarMapping(origin = (0, 0, 0), u = (1, 0, 0), v = (0, 1, 0))
        m2 = transform(m, translation([1, 2, 3]) * rotation(zaxis, π / 2))
        @test m2.origin ≈ Float32[1, 2, 3]
        @test m2.u ≈ Float32[0, 1, 0] atol = 1e-6
        @test transform(NoMapping(), translation([1, 0, 0])) isa NoMapping
    end

    @testset "feasibility" begin
        st(o) = feasibility(o).status
        @test st(intersection(ball, tube, floor_)) === :interior
        @test st(intersection(ball)) === :interior
        @test st(intersection(floor_)) === :interior                         # unbounded, no bounding volume
        # parallel halfspaces: slab, touching, disjoint
        @test st(intersection(halfspace(up, 1.0), halfspace(-up, 1.0))) === :interior   # -1 ≤ z ≤ 1
        @test st(intersection(halfspace(up, 0.0), halfspace(-up, 0.0))) === :boundary   # z = 0 plane only
        @test st(intersection(halfspace(up, -1.0), halfspace(-up, -1.0))) === :empty    # z ≤ -1 and z ≥ 1
        # spheres
        @test st(intersection(sphere(origin, 1.0), sphere([1.5, 0, 0.0], 1.0))) === :interior
        @test st(intersection(sphere(origin, 1.0), sphere([2.0, 0, 0.0], 1.0))) === :boundary   # tangent
        @test st(intersection(sphere(origin, 1.0), sphere([3.0, 0, 0.0], 1.0))) === :empty
        # sphere vs halfspace
        @test st(intersection(sphere(origin, 1.0), halfspace(up, -0.5))) === :interior
        @test st(intersection(sphere(origin, 1.0), halfspace(up, -1.0))) === :boundary
        @test st(intersection(sphere(origin, 1.0), halfspace(up, -2.0))) === :empty
        # cylinders: crossing, tangent, disjoint
        @test st(intersection(cylinder(zaxis, 1.0), cylinder(xaxis, 1.0))) === :interior
        @test st(intersection(cylinder(zaxis, 1.0), cylinder(zaxis, 1.0; point = [2, 0, 0]))) === :boundary
        @test st(intersection(cylinder(zaxis, 1.0), cylinder(zaxis, 1.0; point = [3, 0, 0]))) === :empty
        # coincident boundaries (duplicate surface) is still interior
        @test st(intersection(sphere(origin, 1.0), sphere(origin, 1.0) |> q -> Constraint(q))) === :interior
        # a returned interior point satisfies every inequality
        r = feasibility(intersection(ball, tube, floor_))
        @test r.margin > 0 && r.point in intersection(ball, tube, floor_)
        @test feasibility(intersection(halfspace(up, -2.0), sphere(origin, 1.0))).point === nothing
        # nonconvex rejected
        bad = Quadric(Matrix(Diagonal([1.0, -1.0, 1.0, -1.0])))
        @test_throws ArgumentError feasibility(intersection(bad))
        @test is_feasible(intersection(ball, floor_))
        @test !is_feasible(intersection(halfspace(up, -1.0), halfspace(-up, -1.0)))
    end

    @testset "boundedness" begin
        @test is_bounded(intersection(ball))
        @test is_bounded(intersection(ball, floor_))
        @test !is_bounded(intersection(floor_))
        @test !is_bounded(intersection(tube))
        @test !is_bounded(intersection(tube, floor_))                         # half-infinite tube
        @test is_bounded(intersection(tube, halfspace(up, 1.0), halfspace(-up, 1.0)))
        @test !is_bounded(intersection(halfspace(up, 1.0), halfspace(-up, 1.0)))   # slab
        @test is_bounded(intersection(halfspace(up, -1.0), halfspace(-up, -1.0))) # empty ⇒ bounded
        @test is_bounded(intersection(cylinder(zaxis, 1.0), cylinder(xaxis, 1.0)))  # Steinmetz solid
        @test !is_bounded(intersection(cylinder(zaxis, 1.0), cylinder(zaxis, 2.0)))
        # triangular prism: bounded in plane, unbounded along y
        prism = intersection(halfspace(up, 0.0), halfspace([1, 0, -1.0], 0.0), halfspace([-1, 0, -1.0], 0.0))
        @test !is_bounded(prism)
    end

    @testset "CPU ray caster vs quadric evaluation" begin
        obj = intersection(ball, tube, floor_)
        # camera above: ray straight down the z axis hits the sphere?  z ≤ 0 half: from above, hits floor plane at z=0
        h = ray_hit(obj, [0.0, 0.0, 5.0], [0.0, 0.0, -1.0])
        @test h !== nothing
        @test h.t ≈ 5
        @test h.point ≈ [0, 0, 0] atol = 1e-9
        @test h.id == obj[3].id && h.normal ≈ [0, 0, 1]
        # horizontal ray at height -1 hits the tube (radius 1) before the sphere (r=2)
        h = ray_hit(obj, [-5.0, 0.0, -1.0], [1.0, 0.0, 0.0])
        @test h.t ≈ 4 && h.id == obj[2].id && h.normal ≈ [-1, 0, 0]
        # ray through the tube far below: sphere cap closes it
        h = ray_hit(obj, [0.0, 0.0, -10.0], [0.0, 0.0, 1.0])
        @test h.id == obj[1].id && h.point ≈ [0, 0, -2]
        # miss
        @test ray_hit(obj, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0]) === nothing
        # unbounded: half-infinite tube is hit from the side without any bounding volume
        h = ray_hit(intersection(tube, floor_), [-1e4, 0.0, -1e3], [1.0, 0.0, 0.0])
        @test h.t ≈ 1e4 - 1 rtol = 1e-9
        # inside looking out → exit surface
        h = ray_hit(intersection(ball), [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        @test h.t ≈ 2 && h.normal ≈ [0, 1, 0]
        # seam priority: two coincident halfspaces → lowest index wins
        a = Constraint(halfspace(up, 0.0)); b = Constraint(halfspace(up, 0.0))
        h = ray_hit(intersection(a, b), [0.0, 0.0, 3.0], [0.0, 0.0, -1.0])
        @test h.id == a.id

        # randomised: every hit lies on its own surface and inside all the others
        rng = MersenneTwister(7)
        objs = [intersection(ball, tube, floor_),
                intersection(sphere([1, 0, 0.0], 1.5), sphere([-1, 0, 0.0], 1.5), halfspace([1, 1, 1.0], 1.0)),
                intersection(cylinder(zaxis, 1.0), cylinder(xaxis, 1.2), cylinder(yaxis, 1.4)),
                transform(intersection(ball, halfspace(up, 0.5)), scaling([1.0, 2.0, 0.7]) * rotation([1, 1, 0.0], 0.7))]
        nhits = 0
        for o in objs, _ in 1:300
            org = 6 * normalize(randn(rng, 3)); tgt = randn(rng, 3)
            h = ray_hit(o, org, tgt - org)
            h === nothing && continue
            nhits += 1
            c = o[h.constraint]
            g = norm(gradient(c, h.point))
            @test abs(c(h.point)) ≤ 1e-8 * max(g, 1)                   # on its own surface
            for cj in o.constraints                                   # inside the rest
                @test cj(h.point) ≤ 1e-3 * max(norm(gradient(cj, h.point)), 1)
            end
            @test h.normal ≈ normalize(gradient(c, h.point))
            # nothing nearer: sample points along the ray before the hit must be outside the object
            d = normalize(tgt - org)
            for s in 0.05:0.1:0.95
                @test !(org + s * h.t * d in o)
            end
        end
        @test nhits > 150
        # scene: nearest object wins
        sc = Scene(sphere([0, 0, 0.0], 1.0), sphere([5, 0, 0.0], 1.0))
        k, hit = ray_hit(sc, [10.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        @test k == 2 && hit.t ≈ 4
    end

    @testset "roots (shared with shader)" begin
        @test QuadricLab.ray_roots(1.0, 0.0, -4.0) == (-2.0, 2.0)
        @test QuadricLab.ray_roots(0.0, 2.0, -4.0) == (1.0,)                # linear
        @test QuadricLab.ray_roots(1.0, 0.0, 1.0) == ()                      # no real roots
        @test QuadricLab.ray_roots(0.0, 0.0, 1.0) == ()                      # parallel, outside
        r = QuadricLab.ray_roots(1.0, 1e8, 1.0)                              # cancellation-prone
        @test r[1] * r[2] ≈ 1.0 rtol = 1e-9
    end

    @testset "camera" begin
        cam = Camera(target = (0, 0, 0), distance = 10, yaw = 0, pitch = 0)
        @test eye(cam) ≈ [10, 0, 0]
        r, u, f = basis(cam)
        @test f ≈ [-1, 0, 0]
        @test u ≈ [0, 0, 1]
        @test r ≈ [0, 1, 0]                       # right = forward × up
        o, d = camera_ray(cam, 0, 0, 1.5)
        @test d ≈ f
        _, d2 = camera_ray(cam, 0, 1, 1.0)                                    # top edge: half-fov above forward
        @test acos(dot(d2, f)) ≈ cam.fov / 2
        orbit!(cam, 0, 10.0); @test cam.pitch < π / 2                          # clamped
        zoom!(cam, 0.5); @test cam.distance ≈ 5
        t0 = cam.target; pan!(cam, 1, 0); @test norm(cam.target - t0) ≈ 1
    end

    @testset "GPU packing" begin
        t = checker()
        sc = Scene(sphere(origin, 1.0, Style(albedo = colorant"red", texture = t, mapping = SphericalMapping())),
                   intersection(Constraint(cylinder(zaxis, 1.0), Style(texture = t, mapping = PlanarMapping())),
                                halfspace(up, 0.0)))
        p = pack(sc)
        @test size(p.constraints) == (4, QuadricLab.TEXELS_PER_CONSTRAINT, 3)
        @test p.objects[:, 1] == Float32[0, 1, 0, 0] && p.objects[:, 2] == Float32[1, 2, 0, 0]
        @test p.ntextures == 1 && size(p.textures, 4) == 1                    # shared texture, one layer
        @test p.constraints[:, 1:4, 1] ≈ Float32.(sphere(origin, 1.0).M)       # M columns
        @test p.constraints[1:3, 5, 1] ≈ Float32[1, 0, 0] atol = 0.01
        @test p.constraints[3, 6, 1] == 0f0 && p.constraints[4, 6, 1] == 3f0   # layer 0, spherical
        @test p.constraints[3, 6, 3] == -1f0 && p.constraints[4, 6, 3] == 0f0  # no texture
        @test Int.(p.constraints[4, 9, :]) == p.ids
        @test size(pack(Scene(ball)).textures, 4) == 1                        # dummy layer
    end
end
