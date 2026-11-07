# ---------------------------------------------------------------------------
# Orbit camera, Z-up, right-handed.  eye = target + distance·(cos p cos y, cos p sin y, sin p)
# yaw y about +z, pitch p = elevation above the xy-plane (clamped away from the poles).
# Pure Julia; no GL dependency.
# ---------------------------------------------------------------------------

mutable struct Camera
    target::SVector{3,Float64}
    distance::Float64
    yaw::Float64
    pitch::Float64
    fov::Float64          # vertical field of view, radians
end

function Camera(; target = (0.0, 0.0, 0.0), distance::Real = 10.0, yaw::Real = -0.6,
                pitch::Real = 0.45, fov::Real = deg2rad(45))
    distance > 0 || throw(ArgumentError("distance must be positive"))
    0 < fov < π || throw(ArgumentError("fov must be in (0, π)"))
    Camera(SVector{3,Float64}(target), Float64(distance), Float64(yaw), Float64(pitch), Float64(fov))
end

const _MAX_PITCH = π / 2 - 1e-3

eye(c::Camera) = c.target + c.distance * SVector(cos(c.pitch) * cos(c.yaw), cos(c.pitch) * sin(c.yaw), sin(c.pitch))

"Orthonormal (right, up, forward) camera basis; forward points from eye to target."
function basis(c::Camera)
    fwd = normalize(c.target - eye(c))
    right = normalize(cross(fwd, SVector(0.0, 0.0, 1.0)))
    up = cross(right, fwd)
    (right, up, fwd)
end

"Orbit by (dyaw, dpitch) radians."
function orbit!(c::Camera, dyaw::Real, dpitch::Real)
    c.yaw += dyaw
    c.pitch = clamp(c.pitch + dpitch, -_MAX_PITCH, _MAX_PITCH)
    c
end

"Dolly: multiply distance by `factor` (>0)."
zoom!(c::Camera, factor::Real) = (c.distance = max(c.distance * factor, 1e-6); c)

"Pan the target in the image plane by (dx, dy) world-units-at-target."
function pan!(c::Camera, dx::Real, dy::Real)
    r, u, _ = basis(c)
    c.target += dx * r + dy * u
    c
end

"World-space unit ray direction through NDC point (x, y) ∈ [-1,1]², given the aspect ratio w/h."
function camera_ray(c::Camera, x::Real, y::Real, aspect::Real)
    r, u, f = basis(c)
    th = tan(c.fov / 2)
    d = normalize(f + x * th * aspect * r + y * th * u)
    (eye(c), d)
end
