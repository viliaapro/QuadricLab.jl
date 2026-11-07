using Colors

# Compare the GLSL renderer against the CPU reference ray caster, pixel by pixel.
@testset "OpenGL renderer vs CPU ray caster" begin
    W, H = 160, 120
    s_red   = Style(albedo = colorant"red")
    s_green = Style(albedo = colorant"green")
    s_blue  = Style(albedo = colorant"blue")
    obj = intersection(sphere(origin, 2.0, s_red), cylinder(zaxis, 1.0, s_green), halfspace(up, 0.5, s_blue))
    cam = Camera(distance = 9, yaw = -0.7, pitch = 0.5)
    img = render_image(obj; width = W, height = H, camera = cam)
    @test size(img) == (H, W)

    # classify by dominant colour channel; the grey background has low saturation
    function cls(c)
        r, g, b = red(c), green(c), blue(c)
        m = max(r, g, b)
        m - min(r, g, b) < 0.15 ? :bg : (r == m ? :red : (g == m ? :green : :blue))
    end
    names = Dict(obj[1].id => :red, obj[2].id => :green, obj[3].id => :blue)
    agree = 0; total = 0; hitpix = 0
    for j in 1:H, i in 1:W
        x = 2 * (i - 0.5) / W - 1; y = 1 - 2 * (j - 0.5) / H
        o, d = camera_ray(cam, x, y, W / H)
        h = ray_hit(obj, o, d)
        want = h === nothing ? :bg : names[h.id]
        got = cls(img[j, i])
        total += 1; agree += (want == got); hitpix += (h !== nothing)
    end
    @test hitpix > 0.05 * W * H
    @test agree / total > 0.985            # disagreements only at silhouettes / seams / background shading
end

@testset "OpenGL: unbounded, translucent, textured" begin
    cam = Camera(distance = 8)
    # half-infinite tube rendered with no bounding volume: pixels must be non-background
    img = render_image(intersection(cylinder(zaxis, 1.0), halfspace(up, 0.0)); width = 64, height = 48, camera = cam)
    @test any(c -> c != img[1, 1], img)
    # translucent sphere over an opaque one: centre pixel mixes both colours
    sc = Scene(sphere([0, 0, 0.0], 1.0, Style(albedo = colorant"blue")),
               sphere([3, 0, 0.0], 1.0, Style(albedo = colorant"red", opacity = 0.5)))
    cam2 = Camera(target = (0, 0, 0), distance = 12, yaw = 0, pitch = 0)  # eye on +x → red sphere in front
    img = render_image(sc; width = 64, height = 48, camera = cam2)
    c = img[24, 32]
    @test blue(c) > 0.2 && red(c) > 0.2
    # texture changes pixel values relative to untextured
    plain = render_image(sphere(origin, 2.0, Style(albedo = colorant"white")); width = 64, height = 48)
    tex = render_image(sphere(origin, 2.0, Style(albedo = colorant"white", texture = checker(n = 4),
                                                 mapping = SphericalMapping())); width = 64, height = 48)
    @test plain != tex
end
