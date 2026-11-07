@testset "QuadricLab Phase 3" begin
    S(c, r) = Constraint(sphere(c, r))
    near(a, b; tol = 1e-5) = norm(a - b) ≤ tol

    @testset "nnls / cone membership" begin
        G = [1.0 0 0; 0 1 0; 0 0 1]
        λ, r = nnls(G, [1.0, 2.0, 3.0]); @test r < 1e-12 && λ ≈ [1, 2, 3]
        λ, r = nnls(G, [-1.0, 2.0, 0.0]); @test r ≈ 1.0 atol = 1e-12            # negative part cannot be matched
        λ, r = nnls([1.0 1.0; 0.0 0.0; 0.0 0.0], [2.0, 0, 0]); @test r < 1e-12  # duplicate columns
        o = intersection(halfspace(up, 0.0), halfspace(xaxis, 0.0))             # wedge z ≤ 0, x ≤ 0
        nc = normal_cone(o, [0.0, 3.0, 0.0])
        @test length(nc.ids) == 2 && nc.rank == 2 && !nc.degenerate
        @test cone_membership(nc, normalize([1.0, 0, 1]))[2] < 1e-9              # inside the wedge
        @test cone_membership(nc, [0.0, 1.0, 0.0])[2] ≈ 1 atol = 1e-9            # orthogonal to it
        @test isempty(normal_cone(o, [-1.0, 0, -1.0]).ids)                       # interior: nothing active
    end

    @testset "sphere–sphere" begin
        a, b = S(origin, 1.0), S([5.0, 0, 0], 1.5)
        r = distance(a, b)
        @test r.status === :separated && isapprox(r.distance, 2.5; atol = 1e-6)
        @test near(r.point_a, [1, 0, 0]) && near(r.point_b, [3.5, 0, 0]) && near(r.normal, [1, 0, 0])
        @test r.active_a == [a.id] && r.active_b == [b.id]            # identities preserved
        @test r.cone_residual < 1e-6 && r.witness_set === :unique
        @test distance(b, a).normal ≈ -r.normal atol = 1e-5           # argument order flips the normal
        rng = MersenneTwister(3)
        for _ in 1:10
            c1, c2 = randn(rng, 3), 6 .+ randn(rng, 3); r1, r2 = 0.5 + rand(rng), 0.5 + rand(rng)
            d = norm(c1 - c2) - r1 - r2
            d > 0.1 || continue
            q = distance(sphere(c1, r1), sphere(c2, r2))
            @test q.distance ≈ d atol = 1e-5
            @test near(q.normal, normalize(c2 - c1); tol = 1e-4)
        end
        # tangent and overlapping
        t = distance(S(origin, 1.0), S([2.0, 0, 0], 1.0))
        @test t.status === :touching && t.distance == 0 && near(t.point_a, [1, 0, 0]; tol = 1e-4)
        @test near(t.normal, [1, 0, 0]; tol = 1e-4) && t.witness_set === :unique
        o = distance(S(origin, 1.0), S([1.0, 0, 0], 1.0))
        @test o.status === :intersecting && o.distance == 0 && o.normal === nothing   # not a negative distance
        @test o.point_a in intersection(sphere(origin, 1.0), sphere([1.0, 0, 0], 1.0))
    end

    @testset "sphere–halfspace" begin
        ball = S([0.0, 0, 5], 1.0); fl = Constraint(halfspace(up, 0.0))        # solid z ≤ 0
        r = distance(ball, fl)
        @test isapprox(r.distance, 4; atol = 1e-6) && near(r.point_a, [0, 0, 4]) && near(r.point_b, [0, 0, 0])
        @test near(r.normal, [0, 0, -1]) && r.active_b == [fl.id]
        @test r.witness_set === :unique                                       # ball touches the plane at a point
        t = distance(S([0.0, 0, 1], 1.0), fl)
        @test t.status === :touching && near(t.point_a, [0, 0, 0]; tol = 1e-4) && near(t.normal, [0, 0, -1]; tol = 1e-4)
        @test length(t.active_a) == 1 && length(t.active_b) == 1
        @test distance(S([0.0, 0, -0.5], 1.0), fl).status === :intersecting
    end

    @testset "sphere–cylinder, cylinder–cylinder, cylinder–halfspace" begin
        cyl = Constraint(cylinder(zaxis, 2.0))
        r = distance(S([5.0, 0, 7.0], 1.0), cyl)
        @test isapprox(r.distance, 2; atol = 1e-6) && near(r.normal, [-1, 0, 0]; tol = 1e-5)
        # parallel axes: contact along a whole line
        r = distance(Constraint(cylinder(zaxis, 1.0)), Constraint(cylinder(zaxis, 1.5; point = [5, 0, 0])))
        @test isapprox(r.distance, 2.5; atol = 1e-6) && near(r.normal, [1, 0, 0]; tol = 1e-5)
        @test r.witness_set === :unbounded                                    # whole line of closest pairs
        # skew axes: unique closest pair (x on the z-axis at z = 3, y on the other axis at x = 0)
        r = distance(Constraint(cylinder(zaxis, 1.0)), Constraint(cylinder(xaxis, 1.5; point = [0, 4, 3])))
        @test isapprox(r.distance, 4 - 1 - 1.5; atol = 1e-6) && near(r.normal, [0, 1, 0]; tol = 1e-5)
        @test near(r.point_a, [0, 1, 3]; tol = 1e-4) && r.witness_set === :unique
        # cylinder vs halfspace x ≥ 3
        r = distance(Constraint(cylinder(zaxis, 1.0)), Constraint(halfspace(-xaxis, -3.0)))
        @test isapprox(r.distance, 2; atol = 1e-6) && r.witness_set === :unbounded
        # cylinder lying on a plane: touching along a line
        t = distance(Constraint(cylinder(xaxis, 1.0; point = [0, 0, 1.0])), Constraint(halfspace(up, 0.0)))
        @test t.status === :touching && near(t.normal, [0, 0, -1]; tol = 1e-4) && t.witness_set === :unbounded
    end

    @testset "parallel halfspaces and slabs" begin
        r = distance(Constraint(halfspace(up, 0.0)), Constraint(halfspace(-up, -2.0)))     # z ≤ 0  vs  z ≥ 2
        @test isapprox(r.distance, 2; atol = 1e-6) && near(r.normal, [0, 0, 1]) && r.witness_set === :unbounded
        t = distance(Constraint(halfspace(up, 0.0)), Constraint(halfspace(-up, 0.0)))      # share the plane z = 0
        @test t.status === :touching && near(t.normal, [0, 0, 1]; tol = 1e-4) && t.witness_set === :unbounded
        slab = intersection(halfspace(up, 1.0), halfspace(-up, 1.0))                        # |z| ≤ 1, unbounded
        @test distance(slab, S([0.0, 0, 4], 1.0)).distance ≈ 2 atol = 1e-6
    end

    @testset "multiple constraints and multiple active constraints" begin
        # A = ball(2) ∩ {x ≤ 1};   B = ball(0.5) at (3,3,0): closest A point is on the rim x = 1, y²+z² = 3
        A = intersection(Constraint(sphere(origin, 2.0)), Constraint(halfspace(xaxis, 1.0)))
        B = Constraint(sphere([3.0, 3, 0], 0.5))
        r = distance(A, B)
        c = [3.0, 3, 0]; rim = [1.0, sqrt(3.0), 0]
        @test r.distance ≈ norm(c - rim) - 0.5 atol = 1e-5
        @test near(r.point_a, rim; tol = 1e-4)
        @test sort(r.active_a) == sort([A[1].id, A[2].id])                  # two active constraints on A
        @test r.cone_a.rank == 2 && r.cone_residual < 1e-4                    # normal lies in the 2-D cone
        # touching at an edge: wedge {z ≤ 0, x ≤ 0} and a ball touching the edge line at the origin
        W = intersection(Constraint(halfspace(up, 0.0)), Constraint(halfspace(xaxis, 0.0)))
        t = distance(W, Constraint(sphere([1.0, 0, 1.0], sqrt(2.0))))
        @test t.status === :touching && near(t.point_a, origin; tol = 1e-3)
        @test length(t.active_a) == 2 && length(t.active_b) == 1
        @test near(t.normal, normalize([1.0, 0, 1.0]); tol = 1e-4) && t.cone_residual < 1e-4
        @test t.witness_set === :unique
        # coincident boundaries (the same surface twice) are both reported active
        dup = intersection(Constraint(halfspace(up, 0.0)), Constraint(halfspace(up, 0.0)))
        r = distance(dup, S([0.0, 0, 3], 1.0))
        @test isapprox(r.distance, 2; atol = 1e-6) && length(r.active_a) == 2
    end

    @testset "degenerate, empty, unbounded, non-convex" begin
        e = intersection(Constraint(halfspace(up, -1.0)), Constraint(halfspace(-up, -1.0)))   # z ≤ −1 and z ≥ 1
        r = distance(e, S(origin, 1.0))
        @test r.status === :empty_object && isnan(r.distance)
        @test distance(S(origin, 1.0), e).status === :empty_object
        # unbounded half-infinite tube vs a ball above its axis
        tube = intersection(Constraint(cylinder(zaxis, 1.0)), Constraint(halfspace(up, 0.0)))
        @test distance(tube, S([0.0, 0, 6], 1.0)).distance ≈ 5 atol = 1e-6
        @test distance(tube, S([4.0, 0, 3], 1.0)).distance ≈ sqrt(18.0) - 1 atol = 1e-6   # nearest tube point (1,0,0)
        # degenerate quadric x² ≤ 0 (the plane x = 0, no interior)
        plane = Quadric(Matrix(Diagonal([1.0, 0, 0, 0])))
        r = distance(plane, S([3.0, 0, 0], 1.0))
        @test r.status in (:separated, :numerical_failure)
        r.status === :separated && @test isapprox(r.distance, 2; atol = 1e-4)
        # non-convex rejected
        bad = Quadric(Matrix(Diagonal([1.0, -1.0, 1.0, -1.0])))
        @test_throws ArgumentError distance(bad, S(origin, 1.0))
    end

    @testset "rigid-motion invariance and shared ids" begin
        rng = MersenneTwister(11)
        A = intersection(Constraint(sphere(origin, 1.5)), Constraint(halfspace(xaxis, 0.5)))
        B = intersection(Constraint(cylinder(zaxis, 0.7; point = [4, 1, 0])), Constraint(halfspace(up, 2.0)))
        d0 = distance(A, B)
        @test d0.status === :separated
        for _ in 1:4
            H = translation(3 * randn(rng, 3)) * rotation(randn(rng, 3), 2π * rand(rng))
            d1 = distance(transform(A, H), transform(B, H))
            @test d1.distance ≈ d0.distance atol = 1e-5
            @test d1.active_a == d0.active_a                                 # ids survive transform
        end
        # an object and its own translated copy share every id; must not error
        r = distance(A, transform(A, translation([10.0, 0, 0])))
        @test r.status === :separated && r.distance > 0
    end

    @testset "contact" begin
        a, b = S(origin, 1.0), S([2.0, 0, 0], 1.0)
        c = contact(a, b)
        @test c.status === :touching && isapprox(c.point, [1, 0, 0]; atol = 1e-4) && isapprox(c.normal, [1, 0, 0]; atol = 1e-4)
        @test c.active_a == [a.id] && c.active_b == [b.id] && c.contact_set === :unique
        c = contact(a, S([5.0, 0, 0], 1.0))
        @test c.status === :separated && c.point === nothing && isapprox(c.distance, 3; atol = 1e-6)
        c = contact(a, S([1.0, 0, 0], 1.0))
        @test c.status === :intersecting && c.point !== nothing && c.normal === nothing
        # plane on plane: whole-plane contact is reported explicitly as unbounded
        c = contact(Constraint(halfspace(up, 0.0)), Constraint(halfspace(-up, 0.0)))
        @test c.status === :touching && c.contact_set === :unbounded
    end

    @testset "robustness: perturbed inputs, large radii, far from the origin" begin
        rng = MersenneTwister(21)
        # regression: the witness-set probe used to return :unknown for most slightly perturbed sphere pairs
        for _ in 1:40
            c2 = [5.0, 0, 0] .+ 1e-3 * randn(rng, 3); r2 = 1.5 + 1e-3 * randn(rng)
            r = distance(S(origin, 1.0), S(c2, r2))
            @test r.status === :separated && r.witness_set === :unique && r.cone_residual < 1e-6
            @test r.distance ≈ norm(c2) - 1.0 - r2 atol = 1e-6
        end
        # large radii (curved contact must still be a point, not a face)
        r = distance(S(origin, 20.0), S([50.0, 0, 0], 12.0))
        @test isapprox(r.distance, 18; atol = 1e-6) && r.witness_set === :unique
        # the same configurations far from the origin: distance and classification unchanged
        shift = translation([1000.0, -2000.0, 500.0])
        for (A, B, d, ws) in ((S(origin, 1.0), S([5.0, 0, 0], 1.5), 2.5, :unique),
                              (Constraint(halfspace(up, 0.0)), Constraint(halfspace(-up, -2.0)), 2.0, :unbounded),
                              (Constraint(cylinder(zaxis, 1.0)), Constraint(cylinder(xaxis, 1.5; point = [0, 4, 3])), 1.5, :unique))
            r = distance(transform(intersection(A), shift), transform(intersection(B), shift))
            @test r.status === :separated && isapprox(r.distance, d; atol = 1e-6) && r.witness_set === ws
        end
        t = distance(transform(intersection(S(origin, 1.0)), shift), transform(intersection(S([2.0, 0, 0], 1.0)), shift))
        @test t.status === :touching && t.witness_set === :unique
        @test isapprox(t.point_a, [1001.0, -2000.0, 500.0]; atol = 1e-4)            # points come back in the original frame
    end

    @testset "show" begin
        r = distance(S(origin, 1.0), S([5.0, 0, 0], 1.0))
        s = sprint(show, MIME("text/plain"), r)
        @test occursin("separated", s) && occursin("normal", s) && occursin("distance", s)
        @test occursin("touching", sprint(show, MIME("text/plain"), contact(S(origin, 1.0), S([2.0, 0, 0], 1.0))))
    end
end
