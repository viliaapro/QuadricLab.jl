# ---------------------------------------------------------------------------
# Style: appearance of one constraint surface (material + texture + mapping).
# Pure data; the geometry kernel never reads it, the renderer does.
# ---------------------------------------------------------------------------

# ---- Texture ---------------------------------------------------------------

"""
    Texture(data)

Image texture; `data` is any matrix of colours (rows = v, columns = u).
Stored as `Matrix{RGB{Float32}}`. The renderer resamples every texture to a
common resolution (see `TEXTURE_RES`) when packing for the GPU.
"""
struct Texture
    data::Matrix{RGB{Float32}}
    function Texture(data::AbstractMatrix{<:Colorant})
        isempty(data) && throw(ArgumentError("texture must be non-empty"))
        new(Matrix{RGB{Float32}}(data))
    end
end

"Procedural n×n checkerboard texture of `size`×`size` pixels (default 256)."
function checker(c1::Colorant = RGB(1, 1, 1), c2::Colorant = RGB(0.1, 0.1, 0.1);
                 n::Integer = 8, size::Integer = 256)
    px = size ÷ n
    px ≥ 1 || throw(ArgumentError("size must be ≥ n"))
    Texture([iseven((i - 1) ÷ px + (j - 1) ÷ px) ? RGB{Float32}(c1) : RGB{Float32}(c2)
             for i in 1:size, j in 1:size])
end

# ---- Texture-coordinate mappings -------------------------------------------
#
# Each mapping is a frame (origin, u, v) in world space plus a 2-vector scale.
# kind codes are shared with the shader:  0 none, 1 planar, 2 cylindrical, 3 spherical.
#   planar:       (s,t) = (l·u, l·v) .* scale,                       l = x - origin
#   cylindrical:  (s,t) = (atan(l·v̂, l·û)/2π + ½, l·ŵ) .* scale,      ŵ = û × v̂
#   spherical:    (s,t) = (atan(l·v̂, l·û)/2π + ½, acos(l·ŵ/‖l‖)/π) .* scale
# Textures repeat. Mappings are transformed together with their constraint.

abstract type Mapping end

struct NoMapping <: Mapping end

struct PlanarMapping <: Mapping
    origin::SVector{3,Float32}
    u::SVector{3,Float32}
    v::SVector{3,Float32}
    scale::SVector{2,Float32}
end
struct CylindricalMapping <: Mapping
    origin::SVector{3,Float32}
    u::SVector{3,Float32}
    v::SVector{3,Float32}
    scale::SVector{2,Float32}
end
struct SphericalMapping <: Mapping
    origin::SVector{3,Float32}
    u::SVector{3,Float32}
    v::SVector{3,Float32}
    scale::SVector{2,Float32}
end

_sv(v) = SVector{3,Float32}(v)
_sc(s::Real) = SVector{2,Float32}(s, s)
_sc(s) = SVector{2,Float32}(s)

function _frame(axis, ref)
    w = _unit(axis, "mapping axis")
    r = SVector{3,Float64}(ref)
    u = r - dot(r, w) * w
    norm(u) > 1e-8 || throw(ArgumentError("mapping reference direction is parallel to the axis"))
    u = u / norm(u)
    (u, cross(w, u))
end

"Planar mapping onto the plane through `origin` spanned by `u`, `v` (lengths set the texture period)."
PlanarMapping(; origin = (0, 0, 0), u = (1, 0, 0), v = (0, 1, 0), scale = 1) =
    PlanarMapping(_sv(origin), _sv(u), _sv(v), _sc(scale))

"Cylindrical mapping around `axis` through `origin`; angle measured from `ref`."
function CylindricalMapping(; origin = (0, 0, 0), axis = (0, 0, 1), ref = (1, 0, 0), scale = 1)
    u, v = _frame(axis, ref)
    CylindricalMapping(_sv(origin), _sv(u), _sv(v), _sc(scale))
end

"Spherical mapping about `origin` with pole along `axis`; longitude measured from `ref`."
function SphericalMapping(; origin = (0, 0, 0), axis = (0, 0, 1), ref = (1, 0, 0), scale = 1)
    u, v = _frame(axis, ref)
    SphericalMapping(_sv(origin), _sv(u), _sv(v), _sc(scale))
end

mapping_kind(::NoMapping) = 0
mapping_kind(::PlanarMapping) = 1
mapping_kind(::CylindricalMapping) = 2
mapping_kind(::SphericalMapping) = 3

"(origin, u, v, scale) of a mapping; zeros for `NoMapping`."
mapping_frame(::NoMapping) = (zero(SVector{3,Float32}), zero(SVector{3,Float32}),
                              zero(SVector{3,Float32}), SVector{2,Float32}(1, 1))
mapping_frame(m::Mapping) = (m.origin, m.u, m.v, m.scale)

# Affine transform of a mapping (origin moves with H, directions with its linear part).
transform(m::NoMapping, H) = m
function transform(m::M, H::AbstractMatrix) where {M<:Union{PlanarMapping,CylindricalMapping,SphericalMapping}}
    L = SMatrix{3,3,Float64}(H[1:3, 1:3]); t = SVector{3,Float64}(H[1:3, 4])
    M(_sv(L * m.origin + t), _sv(L * m.u), _sv(L * m.v), m.scale)
end

# ---- Style -----------------------------------------------------------------

"""
    Style(; albedo, roughness, metallic, opacity, texture, mapping)

Appearance of one constraint surface.
- `albedo`: any `Colorant` (default light grey); `roughness`, `metallic`, `opacity` ∈ [0,1].
- `texture`: `nothing` or a `Texture`; multiplies the albedo. Needs a `mapping` ≠ `NoMapping`.
- `Style(s; kwargs...)` copies `s` with fields replaced.
"""
struct Style
    albedo::RGB{Float32}
    roughness::Float32
    metallic::Float32
    opacity::Float32
    texture::Union{Nothing,Texture}
    mapping::Mapping
end

function Style(; albedo::Colorant = RGB(0.8, 0.8, 0.8), roughness::Real = 0.5,
               metallic::Real = 0.0, opacity::Real = 1.0,
               texture::Union{Nothing,Texture} = nothing, mapping::Mapping = NoMapping())
    for (n, x) in (:roughness => roughness, :metallic => metallic, :opacity => opacity)
        0 ≤ x ≤ 1 || throw(ArgumentError("$n must lie in [0, 1], got $x"))
    end
    texture !== nothing && mapping isa NoMapping &&
        throw(ArgumentError("a texture needs a mapping (PlanarMapping, CylindricalMapping or SphericalMapping)"))
    Style(RGB{Float32}(albedo), Float32(roughness), Float32(metallic), Float32(opacity), texture, mapping)
end

Style(s::Style; kwargs...) = Style(; albedo = s.albedo, roughness = s.roughness, metallic = s.metallic,
                                   opacity = s.opacity, texture = s.texture, mapping = s.mapping, kwargs...)

function Base.show(io::IO, s::Style)
    print(io, "Style(albedo=", s.albedo, ", roughness=", s.roughness, ", metallic=", s.metallic,
          ", opacity=", s.opacity, s.texture === nothing ? "" : ", textured", ")")
end
