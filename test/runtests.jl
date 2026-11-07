using Test, LinearAlgebra, StaticArrays, Random
using QuadricLab

@testset "QuadricLab Phase 1" begin
    @testset "validation" begin
        @test_throws DimensionMismatch Quadric(zeros(3, 3))
        M = Matrix(1.0I, 4, 4); M[1, 2] = 1.0
        @test_throws ArgumentError Quadric(M)
        @test_throws ArgumentError Quadric(fill(NaN, 4, 4))
    end

    @testset "sphere" begin
        S = sphere([1.0, 2.0, 3.0], 2.0)
        @test S([1, 2, 3]) ≈ -4
        @test S([3, 2, 3]) ≈ 0 atol = 1e-12
        @test S([4, 2, 3]) > 0
        @test gradient(S, [3.0, 2.0, 3.0]) ≈ [4, 0, 0]
        @test normal(S, [1.0, 2.0, 5.0]) ≈ [0, 0, 1]
        @test normal(S, [1.0, 2.0, 3.0]) === nothing
        @test is_convex(S)
        @test_throws ArgumentError sphere(origin, -1.0)
    end

    @testset "cylinder" begin
        C = cylinder(zaxis, 1.0)
        @test C([1, 0, 99]) ≈ 0 atol = 1e-12
        @test C([0, 0, -5]) ≈ -1
        @test C([2, 0, 0]) ≈ 3
        @test cylinder([0, 0, 7.0], 1.0).M ≈ C.M
        C3 = cylinder(xaxis, 2.0; point = [0, 1, 0])
        @test C3([5, 3, 0]) ≈ 0 atol = 1e-12
        @test C3([-9, 1, 0]) ≈ -4
        @test is_convex(C)
        @test normal(C, [1.0, 0.0, 3.0]) ≈ [1, 0, 0]
    end

    @testset "halfspace" begin
        F = halfspace(up, 0.0)
        @test F([0, 0, -2]) ≈ -2
        @test F([0, 0, 3]) ≈ 3
        H = halfspace([0, 0, 5.0], 2.0)
        @test H([7, 7, 2]) ≈ 0 atol = 1e-12
        @test normal(H, [0.0, 0.0, 0.0]) ≈ [0, 0, 1]
        @test iszero(affine_part(F)[1])
        @test is_convex(F)
    end

    @testset "affine transforms: Q'(Hx) == Q(x)" begin
        rng = MersenneTwister(1)
        objs = [sphere([1, -1, 2.0], 1.5), cylinder([1, 2, 3.0], 0.7; point = [1, 0, 0.0]),
                halfspace([1, -1, 2.0], 0.3)]
        Hs = [translation([1, 2, 3.0]),
              rotation([1, 1, 0.0], 0.8),
              scaling([2.0, 0.5, 3.0]),
              translation([0.5, 0, -1]) * rotation([0, 1, 1.0], -1.1) * scaling(1.7)]
        for Q in objs, H in Hs, _ in 1:20
            x = randn(rng, 3)
            y = (H * SVector(x[1], x[2], x[3], 1.0))[1:3]
            @test transform(Q, H)(y) ≈ Q(x) atol = 1e-9 rtol = 1e-9
        end
        @test is_convex(transform(cylinder(zaxis, 1.0), scaling([2.0, 3.0, 1.0])))
    end

    @testset "rotation / translation geometry" begin
        R = rotation(zaxis, π / 2)
        @test R[1:3, 1:3] * [1, 0, 0] ≈ [0, 1, 0]
        @test det(R[1:3, 1:3]) ≈ 1
        S = transform(sphere(origin, 1.0), translation([3, 0, 0.0]))
        @test S([3, 0, 0]) ≈ -1
        @test S.M ≈ sphere([3, 0, 0.0], 1.0).M
        Cr = transform(cylinder(zaxis, 1.0), rotation(yaxis, π / 2))
        @test Cr.M ≈ cylinder(xaxis, 1.0).M atol = 1e-12
    end

    @testset "affine vs projective" begin
        P = SMatrix{4,4}(1.0, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0.1,  0, 0, 0, 1)
        S = sphere(origin, 1.0)
        @test !is_affine(P)
        @test_throws ArgumentError transform(S, P)
        Sp = projective_transform(S, P)
        X = SVector(0.3, -0.2, 0.5, 1.0)
        @test homogeneous_eval(Sp, P * X) ≈ homogeneous_eval(S, X) atol = 1e-10
        @test_throws ArgumentError transform(S, zeros(4, 4))
    end

    @testset "non-convex detection" begin
        @test !is_convex(Quadric(Matrix(Diagonal([1.0, -1.0, 1.0, -1.0]))))
    end
end

include("test_phase2.jl")
include("test_phase3.jl")
include("test_bounds.jl")
include("test_recovery.jl")

# OpenGL rendering tests need a GL 3.3 context (a display, or `xvfb-run` with Mesa).
if get(ENV, "QUADRICLAB_GL_TESTS", "0") == "1"
    include("test_render.jl")
else
    @info "Skipping OpenGL tests (set QUADRICLAB_GL_TESTS=1, e.g. under xvfb-run, to run them)"
end
