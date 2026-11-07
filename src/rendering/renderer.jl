# ---------------------------------------------------------------------------
# OpenGL front end (ModernGL + GLFW). Julia builds scene/camera state and uploads it;
# the fragment shader does ray generation, intersection, validation, normals, shading.
# Requires an OpenGL 3.3 core context.
# ---------------------------------------------------------------------------

const SHADER_DIR = normpath(joinpath(@__DIR__, "..", "..", "shaders"))

# ---- small GL helpers --------------------------------------------------------

function _compile_shader(src::String, typ)
    sh = glCreateShader(typ)
    ptrs = Ptr{GLchar}[Ptr{GLchar}(pointer(src))]
    GC.@preserve src ptrs glShaderSource(sh, 1, ptrs, C_NULL)
    glCompileShader(sh)
    ok = Ref{GLint}(0)
    glGetShaderiv(sh, GL_COMPILE_STATUS, ok)
    if ok[] != GL_TRUE
        len = Ref{GLint}(0); glGetShaderiv(sh, GL_INFO_LOG_LENGTH, len)
        buf = zeros(UInt8, max(len[], 1)); glGetShaderInfoLog(sh, length(buf), C_NULL, buf)
        error("GLSL compile error:\n", String(buf))
    end
    sh
end

function _link_program(vs, fs)
    p = glCreateProgram()
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p)
    ok = Ref{GLint}(0); glGetProgramiv(p, GL_LINK_STATUS, ok)
    if ok[] != GL_TRUE
        len = Ref{GLint}(0); glGetProgramiv(p, GL_INFO_LOG_LENGTH, len)
        buf = zeros(UInt8, max(len[], 1)); glGetProgramInfoLog(p, length(buf), C_NULL, buf)
        error("GLSL link error:\n", String(buf))
    end
    p
end

function _gen(f)               # glGen* helper
    r = Ref{GLuint}(0); f(1, r); r[]
end

# ---- GPU state ---------------------------------------------------------------

mutable struct GLState
    program::GLuint
    vao::GLuint
    tex_data::GLuint
    tex_objects::GLuint
    tex_array::GLuint
    nobjects::Int
end

function GLState()
    vs = _compile_shader(read(joinpath(SHADER_DIR, "quadric.vert"), String), GL_VERTEX_SHADER)
    fs = _compile_shader(read(joinpath(SHADER_DIR, "quadric.frag"), String), GL_FRAGMENT_SHADER)
    prog = _link_program(vs, fs)
    glDeleteShader(vs); glDeleteShader(fs)
    GLState(prog, _gen(glGenVertexArrays), _gen(glGenTextures), _gen(glGenTextures), _gen(glGenTextures), 0)
end

function _tex_params(target, filter, wrap)
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter); glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter)
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap); glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap)
end

function upload!(st::GLState, p::PackedScene)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    n = size(p.constraints, 3)
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, st.tex_data)
    _tex_params(GL_TEXTURE_2D, GL_NEAREST, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXELS_PER_CONSTRAINT, n, 0, GL_RGBA, GL_FLOAT, p.constraints)
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, st.tex_objects)
    _tex_params(GL_TEXTURE_2D, GL_NEAREST, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size(p.objects, 2), 1, 0, GL_RGBA, GL_FLOAT, p.objects)
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D_ARRAY, st.tex_array)
    _tex_params(GL_TEXTURE_2D_ARRAY, GL_LINEAR, GL_REPEAT)
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, TEXTURE_RES, TEXTURE_RES, size(p.textures, 4), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, p.textures)
    st.nobjects = size(p.objects, 2)
    nothing
end

_u(st, name) = glGetUniformLocation(st.program, name)

function draw!(st::GLState, cam::Camera, light, width::Integer, height::Integer)
    glViewport(0, 0, width, height)
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(st.program)
    r, u, f = basis(cam); e = eye(cam)
    glUniform3f(_u(st, "uEye"), e[1], e[2], e[3])
    glUniform3f(_u(st, "uRight"), r[1], r[2], r[3])
    glUniform3f(_u(st, "uUp"), u[1], u[2], u[3])
    glUniform3f(_u(st, "uFwd"), f[1], f[2], f[3])
    glUniform1f(_u(st, "uTanHalfFov"), tan(cam.fov / 2))
    glUniform1f(_u(st, "uAspect"), width / height)
    glUniform3f(_u(st, "uLightDir"), light[1], light[2], light[3])
    glUniform1i(_u(st, "uNumObjects"), st.nobjects)
    glUniform1i(_u(st, "uData"), 0); glUniform1i(_u(st, "uObjects"), 1); glUniform1i(_u(st, "uTex"), 2)
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, st.tex_data)
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, st.tex_objects)
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D_ARRAY, st.tex_array)
    glBindVertexArray(st.vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    nothing
end

# ---- windows -----------------------------------------------------------------

function _make_window(width, height, title; visible::Bool = true)
    GLFW.Init()
    GLFW.DefaultWindowHints()
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    GLFW.WindowHint(GLFW.VISIBLE, visible)
    win = GLFW.CreateWindow(width, height, title)
    GLFW.MakeContextCurrent(win)
    win
end

_unitlight(l) = (v = normalize(SVector{3,Float32}(l)); v)

# ---- offscreen image ---------------------------------------------------------

"""
    render_image(x; width = 800, height = 600, camera = Camera(), light = (0.4, 0.6, 0.8))

Render a Quadric / Constraint / GeoObject / Scene offscreen (hidden window + framebuffer
object) and return a `Matrix{RGB{Float32}}` (row 1 = top of the image).
"""
function render_image(x; width::Integer = 800, height::Integer = 600, camera::Camera = Camera(),
                      light = (0.4, 0.6, 0.8))
    packed = pack(Scene(x))
    win = _make_window(width, height, "QuadricLab"; visible = false)
    try
        st = GLState()
        upload!(st, packed)
        fbo = _gen(glGenFramebuffers); rb = _gen(glGenRenderbuffers)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb)
        glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE || error("framebuffer incomplete")
        draw!(st, camera, _unitlight(light), width, height)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        buf = Array{UInt8}(undef, 4, width, height)
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        [RGB{Float32}(buf[1, i, height - j + 1] / 255f0, buf[2, i, height - j + 1] / 255f0,
                      buf[3, i, height - j + 1] / 255f0) for j in 1:height, i in 1:width]
    finally
        GLFW.DestroyWindow(win)
    end
end

"Write an image (matrix of colours) as binary PPM (P6)."
function write_ppm(path::AbstractString, img::AbstractMatrix{<:Colorant})
    h, w = size(img)
    open(path, "w") do io
        write(io, "P6\n$w $h\n255\n")
        for j in 1:h, i in 1:w
            c = RGB{Float32}(img[j, i])
            write(io, round(UInt8, 255 * clamp(red(c), 0, 1)), round(UInt8, 255 * clamp(green(c), 0, 1)),
                  round(UInt8, 255 * clamp(blue(c), 0, 1)))
        end
    end
    path
end

# ---- interactive viewer ------------------------------------------------------

mutable struct Viewer
    window::GLFW.Window
    state::GLState
    camera::Camera
    light::SVector{3,Float32}
    pending::Union{Nothing,PackedScene}
    redraw::Bool
    closed::Bool
    lastx::Float64
    lasty::Float64
end

"""
    render(x; width = 1000, height = 700, camera = Camera(), light = (0.4,0.6,0.8), block = !isinteractive())

Open an interactive window showing a Quadric / Constraint / GeoObject / Scene.
Mouse: left-drag orbit, right/middle-drag pan, scroll zoom; Esc closes. With `block=false`
(the REPL default) the render loop runs as an `@async` task and a `Viewer` handle is returned:
`update!(viewer, newscene)` replaces the content, `close(viewer)` closes the window.
"""
function render(x; width::Integer = 1000, height::Integer = 700, camera::Camera = Camera(),
                light = (0.4, 0.6, 0.8), block::Bool = !isinteractive())
    packed = pack(Scene(x))
    win = _make_window(width, height, "QuadricLab")
    GLFW.SwapInterval(1)
    v = Viewer(win, GLState(), camera, _unitlight(light), nothing, true, false, 0.0, 0.0)
    upload!(v.state, packed)
    _install_callbacks!(v)
    block ? _run!(v) : (@async _run!(v))
    v
end

"Replace the content of a running viewer (applied on the viewer's next frame)."
function update!(v::Viewer, x)
    v.pending = pack(Scene(x)); v.redraw = true; v
end

Base.close(v::Viewer) = (v.closed = true; nothing)
Base.isopen(v::Viewer) = !v.closed

function _install_callbacks!(v::Viewer)
    w = v.window
    GLFW.SetCursorPosCallback(w, (_, x, y) -> begin
        dx, dy = x - v.lastx, y - v.lasty
        v.lastx, v.lasty = x, y
        if GLFW.GetMouseButton(w, GLFW.MOUSE_BUTTON_LEFT)
            orbit!(v.camera, -0.008dx, 0.008dy); v.redraw = true
        elseif GLFW.GetMouseButton(w, GLFW.MOUSE_BUTTON_RIGHT) || GLFW.GetMouseButton(w, GLFW.MOUSE_BUTTON_MIDDLE)
            s = v.camera.distance * 0.0015
            pan!(v.camera, -s * dx, s * dy); v.redraw = true
        end
    end)
    GLFW.SetScrollCallback(w, (_, _, yoff) -> (zoom!(v.camera, 0.9^yoff); v.redraw = true))
    GLFW.SetKeyCallback(w, (_, key, _, action, _) ->
        (key == GLFW.KEY_ESCAPE && action == GLFW.PRESS) && (v.closed = true))
    GLFW.SetWindowRefreshCallback(w, _ -> v.redraw = true)
    GLFW.SetFramebufferSizeCallback(w, (_, _, _) -> v.redraw = true)
end

function _run!(v::Viewer)
    GLFW.MakeContextCurrent(v.window)
    try
        while !v.closed && !GLFW.WindowShouldClose(v.window)
            GLFW.PollEvents()
            if v.pending !== nothing
                upload!(v.state, v.pending); v.pending = nothing
            end
            if v.redraw
                fb = GLFW.GetFramebufferSize(v.window)
                draw!(v.state, v.camera, v.light, fb.width, fb.height)
                GLFW.SwapBuffers(v.window)
                v.redraw = false
            end
            sleep(0.008)                       # yields to the REPL in async mode
        end
    finally
        v.closed = true
        GLFW.DestroyWindow(v.window)
    end
end
