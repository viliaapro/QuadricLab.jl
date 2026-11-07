#version 330 core
// Fullscreen triangle; no vertex buffers. vNDC spans [-1,1]^2 over the visible area.
out vec2 vNDC;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // (0,0) (2,0) (0,2)
    vNDC = p * 2.0 - 1.0;
    gl_Position = vec4(vNDC, 0.0, 1.0);
}
