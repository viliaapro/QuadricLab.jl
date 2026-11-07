"""
QuadricLab — interactive geometry laboratory for intersections of convex quadratic regions.

Phase 1: homogeneous quadrics, evaluation, gradients, primitives, affine transforms.
Phase 3: distance QCQP (witness points, active constraints, normal cones), contact.
Phase 2: Style / Constraint / GeoObject / Scene, feasibility and boundedness, analytic
OpenGL renderer (shaders/quadric.frag) with a CPU reference ray caster.

Conventions (see README.md):
- Quadric: Q(X) = Xᵀ M X, X = (x,y,z,w); affine chart w = 1.
- Region: { x : q(x) ≤ 0 }, with q(x) = xᵀAx + 2bᵀx + c.
- Transform: M' = H⁻ᵀ M H⁻¹, so Q'(H X) = Q(X).
"""
module QuadricLab

using LinearAlgebra
using SparseArrays
using StaticArrays
using Colors
import Clarabel
using GLFW
using ModernGL

include("quadrics.jl")
include("transforms.jl")
include("style.jl")
include("constraints.jl")
include("objects.jl")
include("primitives.jl")
include("collision/feasibility.jl")
include("collision/normalcone.jl")
include("collision/distance.jl")
include("collision/contact.jl")
include("bounds/support.jl")
include("bounds/terms.jl")
include("bounds/minkowski_sum.jl")
include("bounds/exact.jl")
include("bounds/locus.jl")
include("bounds/recovery.jl")
include("rendering/camera.jl")
include("rendering/textures.jl")
include("rendering/raycast.jl")
include("rendering/renderer.jl")

export Quadric, evaluate, gradient, normal, affine_part, is_convex,
       homogeneous_eval,
       transform, projective_transform, is_affine,
       translation, rotation, scaling, affine,
       sphere, cylinder, halfspace,
       origin, xaxis, yaxis, zaxis, up,
       # Phase 2
       Style, Texture, checker, Mapping, NoMapping, PlanarMapping, CylindricalMapping, SphericalMapping,
       Constraint, reid, GeoObject, intersection, constraints, Scene, inside, violation, active_constraints,
       FeasibilityResult, feasibility, is_feasible, is_bounded,
       # Phase 3
       distance, contact, DistanceResult, ContactResult, NormalCone, normal_cone, cone_membership, nnls,
       # Phase 4: bounding Minkowski sums of terms (support layer)
       SupportResult, AffineHull, Term, PolyhedronTerm, QuadricTerm, MinkowskiSum,
       box_term, interval_term, halfspace_term, whole_space_term, affine_term, ball_term, ellipsoid_term,
       support, support_difference, recession_cone, affine_hull, translate, ambient_dim, is_full_dimensional,
       embed_term,
       bound_violation, nonsmooth, ReferenceLocus, point_locus, segment_locus, halfline_locus, line_locus, affine_locus, locus_vector,
       Displacement, Translations, conservative_displacement, separation_translations,
       Camera, orbit!, zoom!, pan!, eye, basis, camera_ray,
       ray_hit, RayHit, pack, render, render_image, update!, write_ppm, Viewer

end # module
