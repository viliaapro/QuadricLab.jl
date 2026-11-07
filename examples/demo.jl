# julia --project examples/demo.jl            → interactive window (Esc to close)
# julia --project examples/demo.jl out.ppm    → offscreen image
using QuadricLab, Colors

tex   = checker(colorant"white", colorant"navy"; n = 8)
stone = Style(albedo = colorant"burlywood", roughness = 0.8)
gold  = Style(albedo = colorant"goldenrod", roughness = 0.25, metallic = 1.0)
glass = Style(albedo = colorant"lightskyblue", roughness = 0.1, opacity = 0.35)
ball  = Style(albedo = colorant"white", texture = tex, mapping = SphericalMapping())
pipe  = Style(albedo = colorant"white", texture = checker(colorant"white", colorant"firebrick"; n = 4),
              mapping = CylindricalMapping(scale = (4, 1)))
floor_style = Style(albedo = colorant"white", roughness = 0.9, texture = checker(colorant"gray90", colorant"gray40"; n = 2),
                    mapping = PlanarMapping(u = (2, 0, 0), v = (0, 2, 0)))

# An infinite floor (unbounded — no bounding volume anywhere).
ground = intersection(halfspace(up, -2.0, floor_style))

# Sphere ∩ vertical tube ∩ halfspace: a drum with a flat top, each surface with its own Style.
drum = intersection(sphere(origin, 2.0, ball), cylinder(zaxis, 1.5, gold), halfspace(up, 1.0, stone))

# Half-infinite elliptic tube (stretched cylinder), textured, standing on the right.
tube = transform(intersection(cylinder(zaxis, 0.6, pipe), halfspace(up, 3.0, stone)),
                 translation([4.0, 1.0, 0.0]) * scaling([1.0, 1.6, 1.0]))

# Translucent ball in front.
bubble = sphere([-2.0, -3.2, -0.8], 1.1, glass)

scene = Scene(ground, drum, tube, bubble)
cam = Camera(target = (1, 0, -0.3), distance = 13, yaw = -1.0, pitch = 0.4)

if isempty(ARGS)
    render(scene; camera = cam, block = true)
else
    write_ppm(ARGS[1], render_image(scene; width = 960, height = 600, camera = cam))
end
