#version 330 core
// Analytic ray–quadric renderer.
//
// Scene = union of objects; object = intersection of constraints  q_i(x) <= 0,
// q_i(X) = Xᵀ M_i X with X = (x,1).  For a ray X(t) = O + tD, (O=(o,1), D=(d,0)):
//     (DᵀMD) t² + 2 (OᵀMD) t + OᵀMO = 0.
// Candidate roots of every constraint are tested against all other inequalities;
// the nearest valid candidate is the visible boundary. Mirrors src/rendering/raycast.jl.

in vec2 vNDC;
out vec4 fragColor;

uniform vec3  uEye;
uniform vec3  uRight;
uniform vec3  uUp;
uniform vec3  uFwd;
uniform float uTanHalfFov;
uniform float uAspect;
uniform vec3  uLightDir;          // direction TOWARDS the light, normalised
uniform int   uNumObjects;

uniform sampler2D      uData;     // constraint table, 9 texels per row
uniform sampler2D      uObjects;  // (first row, count) per object
uniform sampler2DArray uTex;

const float INF        = 1e30;
const int   MAX_LAYERS = 6;       // translucent surfaces composited per pixel
const float PI         = 3.14159265358979;

struct Constraint {
    mat4  M;
    vec3  albedo; float opacity;
    float roughness, metallic, layer, mapKind;
    vec3  mapOrigin; vec2 mapScale;
    vec3  mapU, mapV;
};

mat4 fetchM(int row) {
    return mat4(texelFetch(uData, ivec2(0, row), 0), texelFetch(uData, ivec2(1, row), 0),
                texelFetch(uData, ivec2(2, row), 0), texelFetch(uData, ivec2(3, row), 0));
}

Constraint fetchConstraint(int row) {
    Constraint c;
    c.M = fetchM(row);
    vec4 t4 = texelFetch(uData, ivec2(4, row), 0);
    vec4 t5 = texelFetch(uData, ivec2(5, row), 0);
    vec4 t6 = texelFetch(uData, ivec2(6, row), 0);
    vec4 t7 = texelFetch(uData, ivec2(7, row), 0);
    vec4 t8 = texelFetch(uData, ivec2(8, row), 0);
    c.albedo = t4.rgb; c.opacity = t4.a;
    c.roughness = t5.x; c.metallic = t5.y; c.layer = t5.z; c.mapKind = t5.w;
    c.mapOrigin = t6.xyz; c.mapScale = vec2(t6.w, t7.w);
    c.mapU = t7.xyz; c.mapV = t8.xyz;
    return c;
}

float quad(mat4 M, vec4 X)  { return dot(X, M * X); }
vec3  grad(mat4 M, vec3 x)  { return 2.0 * (mat3(M) * x + M[3].xyz); }   // ∇q = 2(Ax + b)

// Real roots of a t² + 2 b t + c = 0 (ascending). Returns count in {0,1,2}.
int solveQuadratic(float a, float b, float c, float scale, out float t0, out float t1) {
    t0 = INF; t1 = INF;
    if (abs(a) <= 1e-6 * scale) {                       // linear (halfspace, or ray ∥ cylinder axis)
        if (abs(b) <= 1e-12 * max(scale, abs(c))) return 0;
        t0 = -c / (2.0 * b);
        return 1;
    }
    float disc = b * b - a * c;
    if (disc < 0.0) return 0;
    float sq = sqrt(disc);
    float q = -(b + (b >= 0.0 ? sq : -sq));
    if (q == 0.0) { t0 = 0.0; return 1; }
    float r1 = q / a, r2 = c / q;
    t0 = min(r1, r2); t1 = max(r1, r2);
    return 2;
}

// Is x inside (within tolerance) every constraint of the object except `skip`?
bool insideOthers(vec3 x, int first, int count, int skip, float tol) {
    vec4 X = vec4(x, 1.0);
    for (int j = 0; j < count; ++j) {
        if (j == skip) continue;
        mat4 M = fetchM(first + j);
        float g = length(grad(M, x));
        if (quad(M, X) > tol * g) return false;
    }
    return true;
}

// Nearest valid boundary hit of one object. Ties within tol go to the lowest index.
bool traceObject(vec3 o, vec3 d, float tmin, int first, int count, out float tBest, out int iBest) {
    tBest = INF; iBest = -1;
    vec4 O = vec4(o, 1.0), D = vec4(d, 0.0);
    for (int i = 0; i < count; ++i) {
        mat4 M = fetchM(first + i);
        vec4 MD = M * D;
        float a = dot(D, MD), b = dot(O, MD), c = dot(O, M * O);
        float scale = abs(M[0][0]) + abs(M[1][1]) + abs(M[2][2]) + 1e-20;
        float t0, t1;
        int n = solveQuadratic(a, b, c, scale, t0, t1);
        for (int k = 0; k < 2; ++k) {
            if (k >= n) break;
            float t = (k == 0) ? t0 : t1;
            if (t <= tmin) continue;
            if (t >= tBest - 1e-4) continue;
            float tol = 1e-4 * (1.0 + 0.02 * t);
            if (insideOthers(o + t * d, first, count, i, tol)) { tBest = t; iBest = i; break; }
        }
    }
    return iBest >= 0;
}

// Texture coordinates for the mapping stored with the constraint.
vec2 mapUV(Constraint c, vec3 p) {
    vec3 l = p - c.mapOrigin;
    int kind = int(c.mapKind + 0.5);
    if (kind == 1) return vec2(dot(l, c.mapU), dot(l, c.mapV)) * c.mapScale;
    vec3 u = normalize(c.mapU), v = normalize(c.mapV), w = normalize(cross(u, v));
    float lon = atan(dot(l, v), dot(l, u)) / (2.0 * PI) + 0.5;
    if (kind == 2) return vec2(lon, dot(l, w)) * c.mapScale;
    float r = max(length(l), 1e-9);
    return vec2(lon, acos(clamp(dot(l, w) / r, -1.0, 1.0)) / PI) * c.mapScale;   // kind 3
}

// Cook–Torrance (GGX) with one directional light and a hemispherical ambient term.
vec3 shade(Constraint c, vec3 p, vec3 n, vec3 viewDir) {
    vec3 albedo = c.albedo;
    if (c.layer >= 0.0 && c.mapKind > 0.5)
        albedo *= textureLod(uTex, vec3(mapUV(c, p), c.layer), 0.0).rgb;
    float rough = clamp(c.roughness, 0.05, 1.0);
    vec3 L = uLightDir, V = viewDir, H = normalize(L + V);
    float NL = max(dot(n, L), 0.0), NV = max(dot(n, V), 1e-4), NH = max(dot(n, H), 0.0), VH = max(dot(V, H), 0.0);
    float a = rough * rough, a2 = a * a;
    float dd = NH * NH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G = (NL / (NL * (1.0 - k) + k)) * (NV / (NV * (1.0 - k) + k));
    vec3 F0 = mix(vec3(0.04), albedo, c.metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VH, 5.0);
    vec3 spec = D * G * F / max(4.0 * NL * NV, 1e-4);
    vec3 diff = (1.0 - F) * (1.0 - c.metallic) * albedo / PI;
    vec3 sky = mix(vec3(0.20, 0.17, 0.15), vec3(0.45, 0.52, 0.65), 0.5 + 0.5 * n.z);
    vec3 ambient = sky * albedo * (1.0 - 0.5 * c.metallic);
    return ambient + (diff + spec) * NL * 3.0;
}

vec3 background(vec3 d) {
    return mix(vec3(0.10, 0.10, 0.12), vec3(0.34, 0.42, 0.55), clamp(0.5 + 0.5 * d.z, 0.0, 1.0));
}

void main() {
    vec3 d = normalize(uFwd + vNDC.x * uTanHalfFov * uAspect * uRight + vNDC.y * uTanHalfFov * uUp);
    vec3 o = uEye;
    vec3 color = vec3(0.0);
    float transmit = 1.0;
    float tmin = 0.0;

    for (int layer = 0; layer < MAX_LAYERS; ++layer) {
        float bestT = INF; int bestRow = -1;
        for (int k = 0; k < uNumObjects; ++k) {
            vec4 ob = texelFetch(uObjects, ivec2(k, 0), 0);
            float t; int i;
            if (traceObject(o, d, tmin, int(ob.x + 0.5), int(ob.y + 0.5), t, i) && t < bestT) {
                bestT = t; bestRow = int(ob.x + 0.5) + i;
            }
        }
        if (bestRow < 0) break;

        Constraint c = fetchConstraint(bestRow);
        vec3 p = o + bestT * d;
        vec3 n = normalize(grad(c.M, p));
        if (dot(n, d) > 0.0) n = -n;                    // two-sided: light the face we see
        vec3 lit = shade(c, p, n, -d);
        color += transmit * c.opacity * lit;
        transmit *= (1.0 - c.opacity);
        if (transmit < 0.01) { transmit = 0.0; break; }
        tmin = bestT + 1e-4 * (1.0 + 0.02 * bestT);     // continue behind this translucent surface
    }
    color += transmit * background(d);
    fragColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
