@testset "Phase 4 / Increment 1: bounding terms and Minkowski sums" begin
    rng = MersenneTwister(12)
    D3(v...) = Matrix(Diagonal(Float64[v...]))

    @testset "ball and ellipsoid (QuadricTerm)" begin
        B = ball_term([1.0, 2, 3], 2.0)
        for u in ([0, 0, 1.0], [1, 2, 3.0], [-3, 0.5, 2.0], [0.0, 0, 0])      # not necessarily unit
            r = support(B, u)
            @test r.status === :finite && r.face_dim == (norm(u) == 0 ? 3 : 0)
            @test r.value ≈ dot([1, 2, 3], u) + 2norm(u) atol = 1e-12
            @test dot(u, r.point) ≈ r.value atol = 1e-9
            @test r.point in B
        end
        R = rotation([1, 1, 0.0], 0.7)[1:3, 1:3]
        Σ = R * D3(4, 1, 0.25) * R'; c = [0.5, -1, 2.0]
        El = ellipsoid_term(c, Σ)
        for _ in 1:5
            u = randn(rng, 3); r = support(El, u)
            @test r.status === :finite && isapprox(r.value, dot(c, u) + sqrt(dot(u, Σ * u)); atol = 1e-9)
            @test r.point in El && isapprox(dot(u, r.point), r.value; atol = 1e-9)
        end
        @test ndims(affine_hull(El)) == 3 && is_full_dimensional(affine_hull(El))
        # reuse of an existing 3-D Quadric
        Fq = QuadricTerm(sphere([1.0, 2, 3], 2.0))
        @test support(Fq, [0, 0, 1.0]).value ≈ 5 atol = 1e-12
        # translation
        t = [0.3, -0.2, 1.0]; u = [1, 1, 1.0]
        @test support(translate(B, t), u).value ≈ support(B, u).value + dot(u, t) atol = 1e-12
    end

    @testset "lower-dimensional ellipsoids: disk, segment" begin
        Dk = ellipsoid_term([0.0, 0, 1], D3(1, 1, 0))                      # unit disk in the plane z = 1
        r = support(Dk, [1, 0, 0.0]); @test isapprox(r.value, 1; atol = 1e-12)&& r.face_dim == 0
        r = support(Dk, [0, 0, 1.0]); @test isapprox(r.value, 1; atol = 1e-12)&& r.face_dim == 2 # the whole disk is the support face
        h = affine_hull(Dk)
        @test ndims(h) == 2 && !is_full_dimensional(h) && abs(h.origin[3] - 1) < 1e-12
        @test [0.5, 0.5, 1.0] in Dk
        @test !([0.5, 0.5, 1.1] in Dk) && !([1.0, 1.0, 1.0] in Dk)
        Sg = ellipsoid_term(zeros(3), D3(4, 0, 0))                          # segment [-2, 2] on the x-axis
        @test isapprox(support(Sg, [1, 0, 0.0]).value, 2; atol = 1e-12)&& ndims(affine_hull(Sg)) == 1
        pt0 = ellipsoid_term([1.0, 2, 3], zeros(3, 3))                        # zero shape: the point {center} (degenerate is fine)
        @test isapprox(support(pt0, [1, 1, 1.0]).value, 6; atol = 1e-12)&& ndims(affine_hull(pt0)) == 0 && [1.0, 2, 3] in pt0
    end

    @testset "unbounded quadrics: cylinder, paraboloid, parabolic slab" begin
        cyl = QuadricTerm(D3(1, 1, 0), zeros(3), -1.0)
        r = support(cyl, [0, 0, 1.0])
        @test r.status === :infinite && r.ray !== nothing && dot([0, 0, 1.0], r.ray) > 0
        r = support(cyl, [1, 0, 0.0]); @test r.status === :finite && isapprox(r.value, 1; atol = 1e-12)&& r.face_dim == 1 # a whole line
        @test support(cyl, [1, 0, 1.0]).status === :infinite
        rec = recession_cone(cyl)
        @test support(rec, [0, 0, 1.0]).status === :infinite
        @test support(rec, [1, 0, 0.0]).value ≈ 0 atol = 1e-9                  # rec = the z-axis (a line)
        @test is_full_dimensional(affine_hull(cyl))

        par = QuadricTerm(D3(1, 1, 0), [0, 0, 1.0], 0.0)                    # x² + y² + 2z ≤ 0
        for _ in 1:5
            u1, u2 = randn(rng, 2); τ = 0.3 + rand(rng)
            r = support(par, [u1, u2, τ])
            @test r.status === :finite && isapprox(r.value, (u1^2 + u2^2) / (2τ); atol = 1e-9)
            @test r.point in par && isapprox(dot([u1, u2, τ], r.point), r.value; atol = 1e-9)
        end
        @test support(par, [0, 0, 1.0]).value ≈ 0 atol = 1e-12
        r = support(par, [0.3, 0.1, -0.5])                                       # τ < 0: infinite, with a recession ray
        @test r.status === :infinite && r.ray !== nothing
        @test dot([0.3, 0.1, -0.5], r.ray) > 0 && r.ray[3] < 0 && norm(r.ray[1:2]) < 1e-12
        r = support(par, [1.0, 0, 0])                                            # τ = 0, u₊ ≠ 0: infinite but NO recession ray
        @test r.status === :infinite && r.ray === nothing
        @test support(par, zeros(3)).status === :finite

        slab = QuadricTerm(D3(1, 0, 0), zeros(3), -1.0)                      # x² ≤ 1: a rank-one quadric (not a linear constraint)
        @test isapprox(support(slab, [1, 0, 0.0]).value, 1; atol = 1e-12)&& support(slab, [0, 1, 0.0]).status === :infinite
    end

    @testset "restriction by affine equalities" begin
        disk = QuadricTerm(D3(1, 1, 0), zeros(3), -1.0; E = [0.0 0 1], e = [0.0])     # cylinder ∩ {z = 0}
        @test isapprox(support(disk, [0, 0, 1.0]).value, 0; atol = 1e-12)&& support(disk, [0, 0, 1.0]).face_dim == 2
        @test isapprox(support(disk, [1, 0, 0.0]).value, 1; atol = 1e-12)&& ndims(affine_hull(disk)) == 2
        strip = QuadricTerm(D3(1, 1, 0), zeros(3), -1.0; E = [1.0 0 0], e = [0.0])    # cylinder ∩ {x = 0}: |y| ≤ 1, z free
        @test isapprox(support(strip, [0, 1, 0.0]).value, 1; atol = 1e-12)&& support(strip, [0, 0, 1.0]).status === :infinite
        @test ndims(affine_hull(strip)) == 2
        tangent = QuadricTerm(D3(1, 0, 0), zeros(3), -1.0; E = [1.0 0 0], e = [1.0])  # slab ∩ {x = 1}: the whole plane x = 1
        @test support(tangent, [0, 1, 0.0]).status === :infinite && isapprox(support(tangent, [1, 0, 0.0]).value, 1; atol = 1e-12)
        @test support(QuadricTerm(D3(1, 0, 0), zeros(3), -1.0; E = [1.0 0 0], e = [2.0]), [1, 0, 0.0]).status === :empty
    end

    @testset "empty, point-like, validation" begin
        e = QuadricTerm(D3(1, 1, 1), zeros(3), 1.0)                          # x² + 1 ≤ 0
        r = support(e, [1, 0, 0.0]); @test r.status === :empty && r.value == -Inf && affine_hull(e) === nothing
        pt = QuadricTerm(D3(1, 1, 1), zeros(3), 0.0)                          # {0}
        @test isapprox(support(pt, [1, 2, 3.0]).value, 0; atol = 1e-12)&& ndims(affine_hull(pt)) == 0
        hq = QuadricTerm(zeros(3, 3), [0.0, 0, 0.5], 0.0)                        # A = 0: the halfspace z ≤ 0 as a degenerate quadric
        @test isapprox(support(hq, [0, 0, 1.0]).value, 0; atol = 1e-12)&& support(hq, [1, 0, 0.0]).status === :infinite
        @test support(hq, [0, 0, -1.0]).status === :infinite && is_full_dimensional(affine_hull(hq))
        @test support(QuadricTerm(zeros(2, 2), zeros(2), 1.0), [1.0, 0]).status === :empty      # 1 ≤ 0
        @test support(QuadricTerm(zeros(2, 2), zeros(2), -1.0), [1.0, 0]).status === :infinite   # whole space
        @test_throws ArgumentError QuadricTerm(D3(1, -1, 1), zeros(3), -1.0)     # not convex
        @test_throws ArgumentError QuadricTerm([1.0 1; 0 1], zeros(2), -1.0)      # not symmetric
        @test_throws DimensionMismatch support(ball_term(zeros(3), 1.0), [1.0, 0])
    end

    @testset "polyhedra (LP supports)" begin
        bx = box_term([-1.0, -1, -1], [1.0, 1, 1])
        for _ in 1:5
            u = randn(rng, 3); r = support(bx, u)
            @test r.status === :finite && isapprox(r.value, sum(abs, u); atol = 1e-7)&& r.point in bx
        end
        @test support(bx, [1, 0, 0.0]).face_dim == 2 && support(bx, [1, 1, 0.0]).face_dim == 1 && support(bx, [1, 1, 1.0]).face_dim == 0
        recb = recession_cone(bx)
        @test support(recb, [0.3, -1, 2.0]).value ≈ 0 atol = 1e-7                   # rec(box) = {0}
        @test ndims(affine_hull(recb)) == 0

        hs = halfspace_term([0, 0, 1.0], 0.0)                                      # z ≤ 0
        @test support(hs, [0, 0, 1.0]).value ≈ 0 atol = 1e-8
        for u in ([1, 0, 0.0], [0, 0, -1.0], [0.5, 0.2, -1.0])
            r = support(hs, u)
            @test r.status === :infinite && r.ray !== nothing && dot(u, r.ray) > 0
            @test r.ray[3] ≤ 1e-9                                                    # the ray lies in rec = {d_z ≤ 0}
        end
        slab = box_term([-Inf, -Inf, -1.0], [Inf, Inf, 1.0])
        @test isapprox(support(slab, [0, 0, 1.0]).value, 1; atol = 1e-8)&& support(slab, [1, 0, 0.0]).status === :infinite
        @test is_full_dimensional(affine_hull(slab)) && ndims(affine_hull(recession_cone(slab))) == 2

        none = PolyhedronTerm([1.0; -1.0;;], [-1.0, -1.0])                          # x ≤ -1 and x ≥ 1
        @test support(none, [1.0]).status === :empty && affine_hull(none) === nothing

        seg = box_term([-1.0, 0, 0], [1.0, 0, 0])                                  # equalities: a segment
        @test ndims(affine_hull(seg)) == 1
        @test isapprox(support(seg, [0, 1, 0.0]).value, 0; atol = 1e-8)&& support(seg, [0, 1, 0.0]).face_dim == 1
        pinched = PolyhedronTerm([1.0 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1], [1.0, 1, 0, 0, 1, 1])   # y ≤ 0 and −y ≤ 0
        @test ndims(affine_hull(pinched)) == 2                                         # implicit equality y = 0 detected
        ln = affine_term([0.0, 0, 0], reshape([1.0, 0, 0], 3, 1))                      # the x-axis
        @test support(ln, [1, 0, 0.0]).status === :infinite && isapprox(support(ln, [0, 1, 0.0]).value, 0; atol = 1e-8)
        @test support(whole_space_term(2), [1.0, 0]).status === :infinite && support(whole_space_term(2), [0.0, 0]).value == 0
        t = [0.5, -1, 2.0]; u = [1, 2, 3.0]
        @test support(translate(bx, t), u).value ≈ support(bx, u).value + dot(u, t) atol = 1e-7
        @test_throws ArgumentError box_term([1.0], [0.0])
    end

    emb(f, idx, N = 3) = embed_term(f, idx, N)

    @testset "Minkowski sums: terms in a common space (no coordinate tracking)" begin
        disk = emb(ball_term([0.0, 0], 1.0), [1, 2]); seg = emb(interval_term(-1.0, 2.0), [3])
        P = MinkowskiSum(disk, seg)                                              # disk × interval, as a flat disk plus a segment
        @test ambient_dim(P) == 3 && length(P) == 2
        for _ in 1:5
            u = randn(rng, 3); r = support(P, u)
            @test r.status === :finite && isapprox(r.value, norm(u[1:2]) + (u[3] > 0 ? 2u[3] : -u[3]); atol = 1e-7)
            @test all(r.parts[i].point in P.terms[i] for i in 1:2) && isapprox(dot(u, r.point), r.value; atol = 1e-7)
            @test r.point ≈ sum(p.point for p in r.parts)                           # support point of a sum = sum of support points
        end
        @test isapprox(support(disk, [0.3, 0, 5.0]).value, 0.3; atol = 1e-12)       # embedded: u outside the indices is ignored (those coordinates are 0)
        # non-contiguous coordinates, and OVERLAPPING terms (a Minkowski sum) are fine
        N = MinkowskiSum(emb(ball_term([0.0, 0], 1.0), [1, 3]), emb(interval_term(0.0, 1.0), [2]))
        @test isapprox(support(N, [3, 5, 4.0]).value, 5 + 5; atol = 1e-7)
        ov = MinkowskiSum(emb(interval_term(0.0, 1.0), [1], 2), emb(interval_term(0.0, 1.0), [1], 2))
        @test isapprox(support(ov, [2.0, 0]).value, 4; atol = 1e-7) && ndims(affine_hull(ov)) == 1
        bb = MinkowskiSum(ball_term([0.0, 0, 0], 1.0), ball_term([1.0, 0, 0], 0.5))   # sum of two balls: radius 1.5 about (1,0,0)
        u = [0.3, -0.4, 0.5]
        @test isapprox(support(bb, u).value, u[1] + 1.5 * norm(u); atol = 1e-9)
        @test_throws DimensionMismatch MinkowskiSum(ball_term([0.0, 0], 1.0), interval_term(0.0, 1.0))   # terms must share one space
        @test_throws DimensionMismatch embed_term(ball_term([0.0, 0], 1.0), [1], 3)
        @test_throws ArgumentError embed_term(ball_term([0.0, 0], 1.0), [1, 1], 3)
        # an unbounded term: infinite only along directions it can reach
        Q = MinkowskiSum(disk, emb(whole_space_term(1), [3]))                  # a cylinder: disk + line
        @test isapprox(support(Q, [1, 0, 0.0]).value, 1; atol = 1e-12)
        r = support(Q, [0.2, 0, 1.0]); @test r.status === :infinite && r.ray !== nothing && dot([0.2, 0, 1.0], r.ray) > 0
        # precedence: empty > infinite > failed > finite
        emp = QuadricTerm(D3(1, 1), zeros(2), 1.0)
        W2 = MinkowskiSum(whole_space_term(2), whole_space_term(2))
        @test support(MinkowskiSum(emp, whole_space_term(2)), [0.0, 1]).status === :empty
        @test QuadricLab._combine(W2, [QuadricLab._sup_failed("x"), QuadricLab._sup_infinite([0.0, 1.0])]).status === :infinite   # +∞ is not a failure
        @test QuadricLab._combine(W2, [QuadricLab._sup_failed("x"), QuadricLab._sup_finite(1.0, [0.0, 0.0], 0)]).status === :failed
        # hull / recession cone / translation
        @test is_full_dimensional(P) && !is_full_dimensional(MinkowskiSum(disk, emb(affine_term([0.0], zeros(1, 0)), [3])))
        @test ndims(affine_hull(MinkowskiSum(ellipsoid_term([0.0, 0, 0], D3(1, 1, 0)), emb(interval_term(0.0, 1.0), [3])))) == 3
        @test support(recession_cone(Q), [0, 0, 1.0]).status === :infinite && isapprox(support(recession_cone(P), [1, 1, 1.0]).value, 0; atol = 1e-7)
        t = [1.0, 2, 3]; u = [1, -1, 2.0]
        @test support(translate(P, t), u).value ≈ support(P, u).value + dot(u, t) atol = 1e-7
        @test sprint(show, P) |> s -> occursin("QuadricTerm", s) && occursin("PolyhedronTerm", s)
    end

    @testset "Minkowski difference support h_A(u) + h_B(−u)" begin
        cA, cB = [1.0, 0, 2], [-1.0, 0.5, 0]
        A, B = ball_term(cA, 1.5), ball_term(cB, 0.5)
        for _ in 1:5
            u = randn(rng, 3); r = support_difference(A, B, u)
            @test r.status === :finite && isapprox(r.value, dot(cA - cB, u) + 2.0 * norm(u); atol = 1e-9)
        end
        ba, bb = box_term([0.0, 0, 0], [2.0, 2, 2]), box_term([-1.0, -1, -1], [1.0, 1, 1])
        u = [1, -2, 0.5]
        @test support_difference(ba, bb, u).value ≈ dot([1.0, 1, 1], u) + sum(abs, u) * 2 atol = 1e-7   # centres (1,1,1),(0,0,0); half-sizes 1,1
        fl = halfspace_term([0, 0, 1.0], 0.0); bl = ball_term([0.0, 0, 0.3], 1.0)                   # ball − floor (z ≤ 0): finite only for u = (0,0,−μ)
        r = support_difference(bl, fl, [0, 0, -1.0]); @test r.status === :finite && isapprox(r.value, 1 - 0.3; atol = 1e-7)
        @test support_difference(bl, fl, [1, 0, 0.0]).status === :infinite
        PA = MinkowskiSum(emb(ball_term([0.0, 0], 1.0), [1, 2]), emb(interval_term(0.0, 1.0), [3]))
        PB = MinkowskiSum(emb(ball_term([1.0, 0], 0.5), [1, 2]), emb(interval_term(-1.0, 0.5), [3]))
        u = [0.3, -0.4, 0.8]
        want = (dot([-1.0, 0], u[1:2]) + 1.5 * norm(u[1:2])) + (support(interval_term(0.0, 1.0), [0.8]).value + support(interval_term(-1.0, 0.5), [-0.8]).value)
        @test support_difference(PA, PB, u).value ≈ want atol = 1e-7
        Bb = MinkowskiSum(ball_term([0.5, 0, 0], 1.0))                          # decompositions need not match
        @test support_difference(PA, Bb, u).value ≈ support(PA, u).value + support(Bb, -u).value atol = 1e-7
        @test_throws DimensionMismatch support_difference(PA, MinkowskiSum(ball_term([0.0, 0], 1.0)), [1.0, 0, 0])
    end

    @testset "transformations act on the terms (any invertible affine map)" begin
        P = MinkowskiSum(emb(ball_term([0.0, 0], 1.0), [1, 2]), emb(interval_term(-1.0, 2.0), [3]))
        Hm = translation([1.0, -2, 0.5]) * rotation([1, 2, 3.0], 0.9)               # a general rotation mixes the coordinates: fine, no block structure
        L, t = Hm[1:3, 1:3], Hm[1:3, 4]
        PT = transform(P, Hm)
        for _ in 1:5
            u = randn(rng, 3)
            @test support(PT, u).value ≈ support(P, L' * u).value + dot(t, u) atol = 1e-7        # h_{LK+t}(u) = h_K(Lᵀu) + t·u
            @test isapprox(dot(u, support(PT, u).point), support(PT, u).value; atol = 1e-7)
        end
        @test ndims(affine_hull(PT)) == 3 && length(PT) == 2
        Ls = [2.0 0.5 0; 0 1 0; 0 0 3]; PS = transform(P, Ls, [0.0, 0, 1]); u = [0.4, -0.7, 0.2]      # non-orthogonal: still exact
        @test isapprox(support(PS, u).value, support(P, Ls' * u).value + u[3]; atol = 1e-7)
        @test_throws ArgumentError transform(P, [1.0 0 0; 0 1 0; 0 0 0], zeros(3))                     # a singular map is not representable per term
        @test_throws ArgumentError transform(P, [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0.1 0 0 1])             # projective: rejected
        @test_throws DimensionMismatch transform(P, Matrix(1.0I, 2, 2), zeros(2))
        H2 = translation([0.0, 1, 0]) * rotation(zaxis, 0.5)                                            # composition
        @test support(transform(transform(P, Hm), H2), u).value ≈ support(transform(P, H2 * Hm), u).value atol = 1e-7
        Lf = [1.5 0.2 0; 0 0.8 0.1; 0 0 1.2]; tf = [0.3, -0.4, 0.2]; uu = [0.2, 0.5, -0.8]             # term transforms: quadric and polyhedron
        for f in (ball_term([0.5, 0, 1.0], 1.2), box_term([-1.0, -2, 0], [1.0, 1, 3]), ellipsoid_term([0.0, 1, 0], D3(2, 1, 0.5)))
            @test support(transform(f, Lf, tf), uu).value ≈ support(f, Lf' * uu).value + dot(tf, uu) atol = 1e-6
        end
        @test_throws ArgumentError transform(ball_term(zeros(3), 1.0), zeros(3, 3), zeros(3))
    end
end

@testset "Phase 4: exact support of GeoObjects and bound containment" begin
    # exact support of quadric intersections against closed forms
    ball = intersection(sphere([1.0, 2, 3], 2.0))
    box = intersection(halfspace(xaxis, 1.0), halfspace(-xaxis, 1.0), halfspace(yaxis, 2.0), halfspace(-yaxis, 2.0), halfspace(zaxis, 3.0), halfspace(-zaxis, 3.0))
    for u in ([0, 0, 1.0], [1, -2, 0.5], [-0.3, 0.4, -2.0])
        @test support(ball, u).value ≈ dot([1, 2, 3], u) + 2norm(u) atol = 1e-7
        @test support(box, u).value ≈ abs(u[1]) + 2abs(u[2]) + 3abs(u[3]) atol = 1e-7
    end
    @test support(intersection(halfspace(zaxis, 0.0)), [1, 0, 0.0]).status === :infinite
    @test support(intersection(halfspace(zaxis, 0.0)), [0, 0, 1.0]).value ≈ 0 atol = 1e-8
    @test support(intersection(halfspace(zaxis, -1.0), halfspace(-zaxis, -1.0)), [0, 0, 1.0]).status === :empty
    @test_throws ArgumentError support(intersection(Quadric(Matrix(Diagonal([1.0, -1.0, 1.0, -1.0])))), [1, 0, 0.0])
    # an ellipsoid (a = 1.4, b = 0.6) inside the capsule marrow ⊕ ball with half-length a − b and radius b + 0.05
    ell = transform(intersection(sphere(origin, 1.0)), scaling([1.4, 0.6, 0.6]))
    cap(r) = MinkowskiSum(segment_locus([-0.8, 0, 0], [0.8, 0, 0]).set, ball_term(zeros(3), r))
    @test bound_violation(ell, cap(0.65)).violation ≤ 1e-6                           # contained
    @test bound_violation(ell, cap(0.5)).violation > 0.05                            # too thin: proven not to contain it
    @test bound_violation(ell, MinkowskiSum(ball_term(zeros(3), 1.0))).violation > 0.3   # a ball too short along the long axis
    @test bound_violation(intersection(halfspace(zaxis, 0.0)), MinkowskiSum(ball_term(zeros(3), 5.0))).violation == Inf   # an unbounded object cannot fit
end
