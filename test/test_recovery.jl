@testset "Phase 4: support-based conservative recovery" begin
    rng = MersenneTwister(31)
    P1(f) = MinkowskiSum(f)
    emb(f, idx) = embed_term(f, idx, 3)
    cyl(c, r) = MinkowskiSum(emb(ball_term(c, r), [1, 2]), emb(whole_space_term(1), [3]))                       # disk + line
    prism(lo2, hi2, zf) = MinkowskiSum(emb(box_term(lo2, hi2), [1, 2]), emb(zf, [3]))                                # polygon × z-term
    bx(lo, hi) = MinkowskiSum(box_term(lo, hi))
    bl(c, r) = MinkowskiSum(ball_term(c, r))
    D3(v...) = Matrix(Diagonal(Float64[v...]))
    hD(A, B, u) = support_difference(A, B, u).value
    # after applying t to A and B, the bound's difference support in direction u (should be ≤ 0 for a sufficient displacement)
    moved(A, B, tr) = (translate(A, tr.t_a), translate(B, tr.t_b))

    @testset "the guarantee holds along EVERY direction (sign convention)" begin
        Rm = rotation([1, 2, 3.0], 0.8)
        bounds = [bl([0.0, 0, 0], 1.2), bx([-1.0, -1, -1], [1.0, 1, 1]), P1(ellipsoid_term([0.3, 0, 0], D3(2, 1, 0.5))),
                  transform(bx([-0.5, -1, -0.7], [0.5, 1, 0.7]), Rm * scaling([1.0, 1.3, 0.9])),
                  prism([-1.0, -1], [1.0, 1], ball_term([0.0], 1.0))]
        for A in bounds, B in (translate(bl([0.0, 0, 0], 0.8), [0.9, 0.2, -0.3]), translate(bx([0.0, 0, 0], [1.0, 2, 1]), [0.5, 0, 0]))
            for _ in 1:4
                u = normalize(randn(rng, 3)); s = hD(A, B, u)
                tA = -s * u                                                   # A moves by −h_D(u)·u, B stays
                @test hD(translate(A, tA), B, u) ≈ 0 atol = 1e-6                # exactly touching along u
                tB = s * u                                                    # or B moves by +h_D(u)·u
                @test hD(A, translate(B, tB), u) ≈ 0 atol = 1e-6
                v = -u                                                        # in terms of A's own direction v: magnitude h_A(−v) + h_B(v)
                @test support(A, -v).value + support(B, v).value ≈ s atol = 1e-9
            end
        end
    end

    @testset "reference loci: types, closest points, non-unique / meeting loci" begin
        pa, pb = point_locus([0.0, 0, 0]), point_locus([3.0, 4, 0])
        r = locus_vector(pa, pb); @test r.status === :separate && isapprox(r.distance, 5; atol = 1e-6)&& norm(r.vector - [3, 4, 0]) < 1e-5
        sg = segment_locus([-1.0, 0, 0], [1.0, 0, 0])
        r = locus_vector(sg, point_locus([0.4, 2, 0])); @test norm(r.vector - [0, 2, 0]) < 1e-5 && abs(r.x[1] - 0.4) < 1e-5     # interior closest point
        r = locus_vector(sg, point_locus([4.0, 3, 0])); @test norm(r.vector - [3, 3, 0]) < 1e-5 && abs(r.x[1] - 1) < 1e-5       # endpoint
        skew = locus_vector(sg, segment_locus([0.0, -1, 2], [0.0, 1, 2]))                                                       # skew segments
        @test skew.status === :separate && norm(skew.vector - [0, 0, 2]) < 1e-5
        par = locus_vector(line_locus([0.0, 0, 0], [1.0, 0, 0]), line_locus([5.0, 1, 1], [1.0, 0, 0]))                         # parallel: points non-unique, vector unique
        @test par.status === :separate && norm(par.vector - [0, 1, 1]) < 1e-5
        hl = locus_vector(halfline_locus([0.0, 0, 0], [1.0, 0, 0]), point_locus([-2.0, 1, 0]))
        @test norm(hl.vector - [-2, 1, 0]) < 1e-5                                                                              # nearest point is the halfline's origin
        pl = locus_vector(affine_locus([0.0, 0, 0], [1.0 0; 0 1; 0 0]), point_locus([1.0, 2, 3]))                              # the plane z = 0
        @test norm(pl.vector - [0, 0, 3]) < 1e-5
        @test [sg.kind, line_locus([0.0, 0, 0], [1.0, 0, 0]).kind, halfline_locus([0.0, 0, 0], [1.0, 0, 0]).kind, pa.kind] == [:segment, :line, :halfline, :point]
        # meeting loci: zero vector is never normalised; the complement of their directions supplies candidates
        xl, yl = line_locus([0.0, 0, 0], [1.0, 0, 0]), line_locus([0.0, 0, 0], [0.0, 1.0, 0])
        @test locus_vector(xl, yl).status === :coincident
        comp = QuadricLab._locus_complement(xl, yl)
        @test length(comp) == 2 && all(c -> abs(abs(c[3]) - 1) < 1e-9, comp)
        @test locus_vector(pa, point_locus([0.0, 0, 0])).status === :coincident && length(QuadricLab._locus_complement(pa, pa)) == 6
        @test locus_vector(sg, segment_locus([0.5, 0, 0], [3.0, 0, 0])).status === :coincident                              # overlapping segments
        @test_throws DimensionMismatch locus_vector(pa, point_locus([1.0, 2]))
        # loci transform like the objects they belong to
        Hm = translation([1.0, 2, 3]) * rotation(zaxis, π / 2)
        @test norm(locus_vector(point_locus([0.0, 0, 0]), transform(point_locus([1.0, 0, 0]), Hm)).vector - (Hm * [1.0, 0, 0, 1])[1:3]) < 1e-5
    end

    @testset "smooth supports: ball–ball, loci guide but do not constrain" begin
        for _ in 1:6
            c = randn(rng, 3); r1, r2 = 0.8 + rand(rng), 0.8 + rand(rng); d = 0.3 + rand(rng) * (r1 + r2 - 0.6)
            cb = c + d * normalize(randn(rng, 3)); A, B = bl(c, r1), bl(cb, r2)
            res = conservative_displacement(A, B; locus_a = point_locus(c), locus_b = point_locus(cb))
            @test res.status === :sufficient && res.seed === :locus && res.optimality === :stationary && res.locally_optimized
            @test isapprox(res.raw, r1 + r2 - d; atol = 1e-8)&& isapprox(res.magnitude, r1 + r2 - d + 1e-8; atol = 1e-8)
            @test norm(res.direction - normalize(cb - c)) < 1e-6 && res.evaluations ≤ 6 && res.nonsmooth_steps == 0      # 1 + the 4-point escape probe
            # a locus pointing exactly away: stationary at the MAXIMUM; the probe must escape (loci are guides, not constraints)
            wrong = conservative_displacement(A, B; locus_a = point_locus(cb), locus_b = point_locus(c))
            @test wrong.status === :sufficient && isapprox(wrong.raw, r1 + r2 - d; atol = 1e-6)
            # no loci at all: deterministic fallback seeds
            none = conservative_displacement(A, B)
            @test none.status === :sufficient && isapprox(none.raw, r1 + r2 - d; atol = 1e-6)&& none.seed === :fallback
        end
    end

    @testset "mixed smooth / polyhedral: rounded box edge (analytic)" begin
        A = bx([-1.0, -1, -1], [1.0, 1, 1]); B = bl([1.2, 1.2, 0.0], 0.7)
        res = conservative_displacement(A, B; locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.2, 1.2, 0.0]))
        want = 0.7 - norm([0.2, 0.2])                                         # curved patch beats the flat faces (0.5)
        @test res.status === :sufficient && isapprox(res.raw, want; atol = 1e-6)&& norm(res.direction - normalize([1.0, 1, 0])) < 1e-4
        @test res.raw ≥ want - 1e-9                                           # never below the true minimum
        tr = separation_translations(res)
        @test norm(tr.t_a + res.magnitude * res.direction) < 1e-12 && tr.t_b == zeros(3)
    end

    @testset "polyhedral nonsmoothness and exposed faces" begin
        A = bx([-1.0, -1, -1], [1.0, 1, 1]); B = bx([0.5, -0.8, -1.3], [2.5, 1.2, 0.7])
        r = support_difference(A, B, [1.0, 0, 0])                              # facet normal: whole faces are exposed
        @test r.status === :finite && nonsmooth(r) && r.face_dim ≥ 2
        @test length(r.parts) == 2 && all(!isempty(p.parts) for p in r.parts)    # per-sum results retained
        @test !isempty(r.parts[1].parts[1].active)                              # active polyhedron rows of A's term at the face
        rg = support_difference(A, B, normalize([1.0, 0.3, 0.2]))               # generic direction: a vertex pair, differentiable
        @test rg.face_dim == 0 && !nonsmooth(rg)
        res = conservative_displacement(A, B; locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.5, 0.2, -0.3]))
        @test res.status === :sufficient && isapprox(res.raw, 0.5; atol = 1e-6)&& norm(res.direction - [1, 0, 0]) < 1e-4
        @test res.nonsmooth_steps > 0 && res.optimality === :stationary
        # stationary exactly at a facet normal (nonsmooth minimiser): the bundle's min-norm element is zero
        res = conservative_displacement(A, B; direction = [1.0, 0, 0])
        @test isapprox(res.raw, 0.5; atol = 1e-9)&& res.optimality === :stationary
        tr = separation_translations(res); @test norm(tr.t_a - [-res.magnitude, 0, 0]) < 1e-6
        # equal-size boxes sharing no gap direction: magnitude is a valid (maybe larger) local value, never below the true minimum
        for _ in 1:5
            lo = randn(rng, 3) .* 0.3; B2 = bx(lo .- 0.6 .+ 1.0, lo .+ 0.9 .+ 1.0)
            rr = conservative_displacement(A, B2; locus_a = point_locus(zeros(3)), locus_b = point_locus(lo .+ 1.2))
            axis_min = minimum(vcat([hD(A, B2, e) for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])], [hD(A, B2, -e) for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]))
            if axis_min > 1e-9                                                    # AABBs overlap: the minimum is attained on an axis
                @test rr.status === :sufficient && rr.raw ≥ axis_min - 1e-6
            else
                @test rr.status === :separated || rr.status === :sufficient            # separated boxes: a certified negative support (or touching)
            end
        end
    end

    @testset "mixed sums: polygon × ball-interval terms, different term depths" begin
        mk(zlo, zhi, c) = (prism([-1.0, -1], [1.0, 1], ball_term([0.0], 1.0)), prism([0.5, -0.5], [2.5, 0.5], interval_term(zlo, zhi)))
        A, B = mk(0.7, 2.7, 0)                                                   # p_xy = 0.5 (x), p_z = 0.3: the minimum lives in the z block
        res = conservative_displacement(A, B; extra_starts = 8)
        @test res.status === :sufficient && isapprox(res.raw, 0.3; atol = 1e-6)&& norm(res.direction - [0, 0, 1]) < 1e-4
        cheap = conservative_displacement(A, B; locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.5, 0, 1.7]))
        @test cheap.status === :sufficient && cheap.raw ≥ 0.3 - 1e-6 && cheap.raw ≤ 0.5 + 1e-6         # local result: one of the two block minima
        A, B = mk(0.2, 2.2, 0)                                                   # now p_z = 0.8 > p_xy = 0.5: x block wins
        res = conservative_displacement(A, B; extra_starts = 8)
        @test isapprox(res.raw, 0.5; atol = 1e-6)&& norm(res.direction - [1, 0, 0]) < 1e-4
    end

    @testset "rotated and translated sums (affine maps)" begin
        A0 = bx([-1.0, -1, -1], [1.0, 1, 1]); B0 = bx([0.5, -0.8, -1.3], [2.5, 1.2, 0.7])
        base = conservative_displacement(A0, B0; locus_a = point_locus(zeros(3)), locus_b = point_locus([1.5, 0.2, -0.3]))
        Hm = translation([3.0, -2, 5]) * rotation([1, -2, 0.5], 1.1)
        L = Hm[1:3, 1:3]
        A1, B1 = transform(A0, Hm), transform(B0, Hm)                            # the same rigid motion for both
        r1 = conservative_displacement(A1, B1; locus_a = transform(point_locus(zeros(3)), Hm), locus_b = transform(point_locus([1.5, 0.2, -0.3]), Hm))
        @test r1.status === :sufficient && isapprox(r1.raw, base.raw; atol = 1e-6)&& norm(r1.direction - L * base.direction) < 1e-4
        # different motions for A and B: valid (sufficient) and locally optimal
        Hb = translation([3.5, -2.2, 5.3]) * rotation([0.3, 1, 0.2], -0.4)
        A2, B2 = transform(A0, Hm), transform(B0, Hb)
        r2 = conservative_displacement(A2, B2; locus_a = transform(point_locus(zeros(3)), Hm), locus_b = transform(point_locus([1.5, 0.2, -0.3]), Hb))
        @test r2.status === :sufficient
        u = r2.direction; tr = separation_translations(r2)
        @test hD(translate(A2, tr.t_a), translate(B2, tr.t_b), u) ≤ -r2.margin + 1e-9 + 1e-6       # separated along u by at least the margin
        for _ in 1:8                                                              # local minimality: no nearby direction is lower
            w = normalize(u + 1e-3 * normalize(randn(rng, 3))); @test hD(A2, B2, w) ≥ r2.raw - 1e-5
        end
        # a sheared / non-orthogonal image: the support law is exact, so recovery still works
        Ls = [1.0 0.4 0; 0 1.0 0; 0 0 1.5]
        r3 = conservative_displacement(transform(A0, Ls, zeros(3)), transform(B0, Ls, [0.0, 0, 0]); locus_a = point_locus(zeros(3)), locus_b = point_locus(Ls * [1.5, 0.2, -0.3]))
        @test r3.status === :sufficient && r3.raw ≥ 0 && isfinite(r3.magnitude)
    end

    @testset "unbounded geometry: recession rays, projection onto the finite-support domain" begin
        cylA, cylB = cyl([0.0, 0], 1.0), cyl([1.5, 0], 1.0)
        r = conservative_displacement(cylA, cylB; direction = [0.3, 0.0, 1.0])       # a seed with a z component (infinite support)
        @test r.status === :sufficient && isapprox(r.raw, 0.5; atol = 1e-6)&& abs(r.direction[3]) < 1e-9 && r.direction[1] > 0.99
        slab = bx([-Inf, -Inf, -1.0], [Inf, Inf, 1.0])
        r = conservative_displacement(slab, bl([0.0, 0, 0.0], 1.5); direction = [1.0, 1, 1])
        @test r.status === :sufficient && isapprox(r.raw, 2.5; atol = 1e-6)&& abs(abs(r.direction[3]) - 1) < 1e-9
        # a ball sinking into a floor: the finite-support directions are a single ray
        fl = MinkowskiSum(halfspace_term([0.0, 0, 1.0], 0.0))                   # z ≤ 0
        ball = bl([0.0, 0, 0.3], 1.0)
        r = conservative_displacement(ball, fl; direction = [1.0, 0.2, 0.0])
        @test r.status === :sufficient && isapprox(r.raw, 0.7; atol = 1e-6)&& norm(r.direction - [0, 0, -1]) < 1e-6
        tr = separation_translations(r)
        @test tr.t_a[3] > 0.69 && tr.t_b == zeros(3)                                    # the ball is lifted
        tb = separation_translations(r; mode = :b); @test tb.t_b[3] < -0.69 && tb.t_a == zeros(3)    # or the floor lowered
        # impossible: both z ≤ 0 — every direction has infinite support; certified
        r = conservative_displacement(fl, fl)
        @test r.status === :impossible && r.magnitude == Inf && separation_translations(r).t_a == zeros(3)
        r = conservative_displacement(MinkowskiSum(whole_space_term(3)), bl([0.0, 0, 0], 1.0)); @test r.status === :impossible
        # separated: z ≤ 0 against z ≥ 1 — a separating direction has h < 0
        r = conservative_displacement(fl, MinkowskiSum(halfspace_term([0.0, 0, -1.0], -1.0)))
        @test r.status === :separated && r.raw < 0 && r.magnitude == 0
        # parabolic term: infinite support without a recession ray exists on the boundary of the barrier cone; a finite direction is still found
        par = MinkowskiSum(QuadricTerm(D3(1, 1, 0), [0.0, 0, 1.0], 0.0))
        r = conservative_displacement(par, bl([0.0, 0, 0.5], 0.5))
        @test r.status === :sufficient && isfinite(r.magnitude) && r.direction[3] > 0
        # an exhausted budget is reported as an insufficient search, not as impossibility
        @test conservative_displacement(slab, bl([0.0, 0, 0.0], 1.5); direction = [1.0, 1, 1], max_evals = 1).status in (:no_direction_found, :sufficient)
    end

    @testset "degenerate and empty bounds" begin
        sx, sy = bx([-1.0, 0, 0], [1.0, 0, 0]), bx([0.0, -1, 0], [0.0, 1, 0])      # two crossing segments: D′ is a square in a plane
        r = conservative_displacement(sx, sy)
        @test r.status === :degenerate && r.magnitude == 1e-8 && r.direction === nothing
        @test conservative_displacement(sx, sy; margin = 0.0).magnitude == 0.0
        emp = MinkowskiSum(QuadricTerm(D3(1, 1, 1), zeros(3), 1.0))
        @test conservative_displacement(emp, bl([0.0, 0, 0], 1.0)).status === :empty
        @test_throws DimensionMismatch conservative_displacement(bl([0.0, 0, 0], 1.0), bl([0.0, 0], 1.0))
        @test_throws ArgumentError conservative_displacement(bl([0.0, 0, 0], 1.0), bl([1.0, 0, 0], 1.0); margin = -1.0)
        # coincident reference loci / identical objects: no zero vector is normalised; the complement/fallback seeds give a valid answer
        A = bl([0.0, 0, 0], 1.0)
        r = conservative_displacement(A, A; locus_a = point_locus(zeros(3)), locus_b = point_locus(zeros(3)))
        @test r.status === :sufficient && isapprox(r.raw, 2; atol = 1e-6)&& r.seed === :complement && all(isfinite, r.direction)
        @test occursin("meet", r.note)
        # parallel lines as loci of two parallel cylinders: the closest-point VECTOR is unique, so it seeds the search
        cA = cyl([0.0, 0], 1.0)
        cB = cyl([1.2, 0.5], 1.0)
        r = conservative_displacement(cA, cB; locus_a = line_locus([0.0, 0, 0], [0.0, 0, 1.0]), locus_b = line_locus([1.2, 0.5, 7.0], [0.0, 0, 1.0]))
        @test r.seed === :locus && r.status === :sufficient && isapprox(r.raw, 2 - norm([1.2, 0.5]); atol = 1e-6)
    end

    @testset "user control: margin, budget, accuracy" begin
        A, B = bx([-1.0, -1, -1], [1.0, 1, 1]), bx([0.5, -0.8, -1.3], [2.5, 1.2, 0.7])
        loc = (locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.5, 0.2, -0.3]))
        full = conservative_displacement(A, B; loc..., margin = 0.0)
        mm = conservative_displacement(A, B; loc..., margin = 0.05, rel_margin = 0.1)
        @test isapprox(mm.magnitude, full.raw * 1.1 + 0.05; atol = 1e-6)&& isapprox(mm.raw, full.raw; atol = 1e-9)
        lim = conservative_displacement(A, B; loc..., margin = 0.0, max_iter = 1)
        @test lim.status === :sufficient && lim.optimality === :budget && !lim.locally_optimized
        @test lim.magnitude ≥ full.magnitude - 1e-9 && lim.evaluations < full.evaluations            # cheaper, never smaller than the local minimum
        few = conservative_displacement(A, B; loc..., margin = 0.0, max_evals = 12)
        @test few.status in (:sufficient, :no_direction_found) && few.evaluations ≤ 14
        coarse = conservative_displacement(A, B; loc..., margin = 0.0, tol = 1e-2)
        @test coarse.evaluations ≤ full.evaluations && coarse.raw ≥ full.raw - 1e-9
        ex = conservative_displacement(A, B; loc..., margin = 0.0, extra_starts = 4)
        @test ex.raw ≤ full.raw + 1e-9                                                                # more starts never worsen the result
    end

    @testset "translation distribution (weights / inverse masses)" begin
        A, B = bl([0.0, 0, 0], 1.0), bl([1.5, 0, 0], 1.0)
        r = conservative_displacement(A, B; locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.5, 0, 0]), margin = 0.0)
        s, u = r.magnitude, r.direction
        for (kw, wa) in (((; mode = :a), 1.0), ((; mode = :b), 0.0), ((; mode = :equal), 0.5), ((; inv_mass_a = 1.0, inv_mass_b = 3.0), 0.25),
                         ((; inv_mass_a = 2.0, inv_mass_b = 0.0), 1.0), ((; inv_mass_a = 0.0, inv_mass_b = 5.0), 0.0))
            tr = separation_translations(r; kw...)
            @test tr.w_a ≈ wa && tr.t_a ≈ -wa * s * u && tr.t_b ≈ (1 - wa) * s * u
            @test tr.t_b - tr.t_a ≈ s * u atol = 1e-12                                    # only the relative translation matters
            A2, B2 = moved(A, B, tr)
            @test hD(A2, B2, u) ≈ 0 atol = 1e-9                                           # exactly touching in the support sense
        end
        @test_throws ArgumentError separation_translations(r; inv_mass_a = 0.0, inv_mass_b = 0.0)
        @test_throws ArgumentError separation_translations(r; mode = :nope)
        @test_throws ArgumentError separation_translations(r; inv_mass_a = -1.0)
    end

    @testset "sufficiency against exact geometry (independent of the bounds' own support)" begin
        # random AABB / ball pairs: after the displacement the exact interiors are disjoint
        function overlaps(a, b)                                                           # exact interior overlap of two (center, kind, size) shapes
            (ca, ka, sa), (cb, kb, sb) = a, b
            if ka === :ball && kb === :ball; return norm(ca - cb) < sa + sb - 1e-7
            elseif ka === :box && kb === :box; return all(abs.(ca - cb) .< sa .+ sb .- 1e-7)
            else
                (cbx, hb, cbl, r) = ka === :box ? (ca, sa, cb, sb) : (cb, sb, ca, sa)
                return norm(max.(abs.(cbl - cbx) .- hb, 0.0)) < r - 1e-7
            end
        end
        shape(c, k, s) = k === :ball ? bl(c, s) : bx(c .- s, c .+ s)
        n_ok = 0
        for _ in 1:40
            ka, kb = rand(rng, (:ball, :box)), rand(rng, (:ball, :box))
            sa, sb = 0.6 + rand(rng), 0.6 + rand(rng); ca = randn(rng, 3); cb = ca + 0.8 * (sa + sb) * rand(rng) * normalize(randn(rng, 3))
            overlaps((ca, ka, sa), (cb, kb, sb)) || continue
            A, B = shape(ca, ka, sa), shape(cb, kb, sb)
            r = conservative_displacement(A, B; locus_a = point_locus(ca), locus_b = point_locus(cb))
            @test r.status === :sufficient
            tr = separation_translations(r; mode = :equal)
            @test !overlaps((ca + tr.t_a, ka, sa), (cb + tr.t_b, kb, sb))                  # overlap eliminated
            n_ok += 1
        end
        @test n_ok ≥ 20
        # an outer bound that is larger than the actual objects: the displacement is sufficient for the exact geometry (narrow phase, Phase 3)
        exactA = Constraint(sphere([0.0, 0, 0], 1.0)); exactB = Constraint(sphere([1.4, 0.3, 0], 1.0))
        outerA, outerB = bl([0.0, 0, 0], 1.25), bl([1.4, 0.3, 0], 1.1)                  # radii exceed the true ones: conservative
        @test distance(exactA, exactB).status === :intersecting
        r = conservative_displacement(outerA, outerB; locus_a = point_locus([0.0, 0, 0]), locus_b = point_locus([1.4, 0.3, 0]))
        tr = separation_translations(r; mode = :equal)
        eA, eB = transform(intersection(exactA), translation(tr.t_a)), transform(intersection(exactB), translation(tr.t_b))
        dist = distance(eA, eB)
        @test dist.status in (:separated, :touching)
        @test r.raw ≥ 2.0 - norm([1.4, 0.3]) - 1e-9                                       # at least the exact penetration depth
        @test dist.distance ≤ (1.25 + 1.1) - 2.0 + 1e-6                                   # over-correction is bounded by the bound slack
    end

    @testset "display" begin
        r = conservative_displacement(bl([0.0, 0, 0], 1.0), bl([1.5, 0, 0], 1.0); locus_a = point_locus(zeros(3)), locus_b = point_locus([1.5, 0, 0]))
        s = sprint(show, MIME("text/plain"), r)
        @test occursin("sufficient", s) && occursin("magnitude", s) && occursin("not claimed global", s) && occursin("seed locus", s)
        @test occursin("impossible", sprint(show, MIME("text/plain"), conservative_displacement(MinkowskiSum(halfspace_term([0.0, 0, 1.0], 0.0)), MinkowskiSum(halfspace_term([0.0, 0, 1.0], 0.0)))))
    end
end
