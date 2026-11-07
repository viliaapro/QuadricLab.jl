# ---------------------------------------------------------------------------
# GPU data packing (pure Julia — no OpenGL calls).
#
# Constraint table: Float32 array (4, TEXELS, N); one row (texture row) per constraint.
#   texel 0..3  M columns (M symmetric)           (m[1:4, k])
#   texel 4     albedo.rgb, opacity
#   texel 5     roughness, metallic, texture layer (-1 none), mapping kind
#   texel 6     mapping origin.xyz, scale.x
#   texel 7     mapping u.xyz,      scale.y
#   texel 8     mapping v.xyz,      constraint id
# Object table: Float32 (4, nobj): (first row, count, 0, 0).
# Texture array: UInt8 (4, RES, RES, layers) RGBA, nearest-resampled to RES×RES.
# ---------------------------------------------------------------------------

const TEXELS_PER_CONSTRAINT = 9
const TEXTURE_RES = 256

"Resample a Texture to RES×RES RGBA8 (nearest neighbour). Result indexed [comp, u, v]."
function resample(t::Texture, res::Integer = TEXTURE_RES)
    h, w = size(t.data)
    out = Array{UInt8}(undef, 4, res, res)
    for j in 1:res, i in 1:res                 # i → u (columns), j → v (rows)
        px = t.data[clamp(ceil(Int, j * h / res), 1, h), clamp(ceil(Int, i * w / res), 1, w)]
        out[1, i, j] = round(UInt8, 255 * clamp(red(px), 0, 1))
        out[2, i, j] = round(UInt8, 255 * clamp(green(px), 0, 1))
        out[3, i, j] = round(UInt8, 255 * clamp(blue(px), 0, 1))
        out[4, i, j] = 0xff
    end
    out
end

struct PackedScene
    constraints::Array{Float32,3}        # (4, TEXELS_PER_CONSTRAINT, N)
    objects::Array{Float32,2}            # (4, nobj)
    textures::Array{UInt8,4}             # (4, RES, RES, layers ≥ 1)
    ids::Vector{Int}                     # row → constraint id
    ntextures::Int
end

"Pack a Scene for the GPU. Identical `Texture` objects (===) share one layer."
function pack(scene::Scene)
    rows = Constraint[]; objtab = Vector{NTuple{4,Float32}}()
    for o in scene.objects
        push!(objtab, (Float32(length(rows)), Float32(length(o)), 0f0, 0f0))
        append!(rows, o.constraints)
    end
    isempty(rows) && throw(ArgumentError("empty scene"))
    layer_of = IdDict{Texture,Int}(); layers = Texture[]
    data = zeros(Float32, 4, TEXELS_PER_CONSTRAINT, length(rows))
    for (r, c) in enumerate(rows)
        M = c.quadric.M; s = c.style
        for k in 1:4, i in 1:4
            data[i, k, r] = Float32(M[i, k])
        end
        layer = -1
        if s.texture !== nothing
            layer = get!(layer_of, s.texture) do
                push!(layers, s.texture); length(layers) - 1
            end
        end
        o, u, v, sc = mapping_frame(s.mapping)
        data[:, 5, r] .= (red(s.albedo), green(s.albedo), blue(s.albedo), s.opacity)
        data[:, 6, r] .= (s.roughness, s.metallic, Float32(layer), Float32(mapping_kind(s.mapping)))
        data[:, 7, r] .= (o[1], o[2], o[3], sc[1])
        data[:, 8, r] .= (u[1], u[2], u[3], sc[2])
        data[:, 9, r] .= (v[1], v[2], v[3], Float32(c.id))
    end
    tex = zeros(UInt8, 4, TEXTURE_RES, TEXTURE_RES, max(length(layers), 1))
    for (l, t) in enumerate(layers)
        tex[:, :, :, l] = resample(t)
    end
    PackedScene(data, reduce(hcat, [collect(t) for t in objtab]), tex, [c.id for c in rows], length(layers))
end
