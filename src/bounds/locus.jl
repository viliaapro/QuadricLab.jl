# ---------------------------------------------------------------------------
# Reference loci: a positional reference of an object, independent of its support function.
#
# Not necessarily a centre of mass: a point, segment, halfline, line, or any higher-dimensional affine/polyhedral locus
# (stored as a PolyhedronTerm). It only guides the optimiser: the relative locus geometry gives an initial direction.
# The vector between the closest points of two loci is unique even when the closest points are not (parallel loci, a locus
# lying along another), so it is used whenever it is nonzero; if it vanishes (intersecting or coincident loci) there is no
# direction to normalise, and the orthogonal complement of the loci's directions supplies deterministic candidates instead.
# ---------------------------------------------------------------------------

struct ReferenceLocus
    set::PolyhedronTerm
    kind::Symbol                 # :point :segment :halfline :line :affine :polyhedral (descriptive)
end
ambient_dim(L::ReferenceLocus) = ambient_dim(L.set)
ReferenceLocus(f::PolyhedronTerm) = ReferenceLocus(f, :polyhedral)

function _line_rows(origin::AbstractVector, direction::AbstractVector)
    d = Vector{Float64}(direction); nd = norm(d); nd > 0 || throw(ArgumentError("direction must be nonzero"))
    d ./= nd; o = Vector{Float64}(origin)
    E = Matrix(nullspace(reshape(d, 1, :))'); (d, o, E, E * o)
end
"The point `p`."
point_locus(p::AbstractVector) = ReferenceLocus(affine_term(p), :point)
"The segment from `a` to `b` (a point if a == b)."
function segment_locus(a::AbstractVector, b::AbstractVector)
    norm(b - a) == 0 && return point_locus(a)
    d, o, E, e = _line_rows(a, b - a); n = length(a)
    ReferenceLocus(PolyhedronTerm(n, [-d'; d'], [-dot(d, a), dot(d, b)], E, e), :segment)
end
"The halfline `origin + t·direction`, t ≥ 0."
function halfline_locus(origin::AbstractVector, direction::AbstractVector)
    d, o, E, e = _line_rows(origin, direction)
    ReferenceLocus(PolyhedronTerm(length(o), reshape(-d, 1, :), [-dot(d, o)], E, e), :halfline)
end
"The line through `point` with `direction`."
line_locus(point::AbstractVector, direction::AbstractVector) = ReferenceLocus(affine_term(point, reshape(Float64.(direction), :, 1)), :line)
"The affine locus `point + span(basis)`."
affine_locus(point::AbstractVector, basis::AbstractMatrix) = ReferenceLocus(affine_term(point, basis), :affine)

translate(L::ReferenceLocus, t::AbstractVector) = ReferenceLocus(translate(L.set, t), L.kind)
transform(L::ReferenceLocus, M::AbstractMatrix, t::AbstractVector) = ReferenceLocus(transform(L.set, M, t), L.kind)
function transform(L::ReferenceLocus, H::AbstractMatrix)
    (size(H) == (4, 4) && ambient_dim(L) == 3) || throw(ArgumentError("a homogeneous transform needs a 4×4 matrix and a 3-D locus"))
    is_affine(H) || throw(ArgumentError("transform requires an affine H"))
    transform(L, Matrix{Float64}(H[1:3, 1:3]), Vector{Float64}(H[1:3, 4]))
end

"""
    locus_vector(La, Lb)

Closest points x ∈ La, y ∈ Lb (a convex QP) and the vector `y − x` from A's locus to B's. Returns
`(vector, distance, x, y, status)`, status `:separate` (distance > tol), `:coincident` (loci meet: no direction) or `:failed`.
"""
function locus_vector(La::ReferenceLocus, Lb::ReferenceLocus; tol::Real = 1e-9)
    fa, fb = La.set, Lb.set; n = fa.n; n == fb.n || throw(DimensionMismatch("loci live in different spaces"))
    pa, pb, ma, mb = size(fa.E, 1), size(fb.E, 1), size(fa.G, 1), size(fb.G, 1)
    Id = Matrix{Float64}(I, n, n)
    rows = vcat(hcat(fa.E, zeros(pa, n)), hcat(zeros(pb, n), fb.E), hcat(fa.G, zeros(ma, n)), hcat(zeros(mb, n), fb.G))
    rhs = vcat(fa.e, fb.e, fa.g, fb.g)
    cones = Clarabel.SupportedCone[]
    pa + pb > 0 && push!(cones, Clarabel.ZeroConeT(pa + pb)); ma + mb > 0 && push!(cones, Clarabel.NonnegativeConeT(ma + mb))
    if isempty(cones)
        return (vector = zeros(n), distance = 0.0, x = zeros(n), y = zeros(n), status = :coincident)
    end
    sol = _solve_conic(sparse(triu([Id -Id; -Id Id])), zeros(2n), sparse(rows), rhs, cones; tols = (1e-12, 1e-10, 1e-8))
    _solved(sol) || return (vector = zeros(n), distance = NaN, x = zeros(n), y = zeros(n), status = :failed)
    x, y = Vector{Float64}(sol.x[1:n]), Vector{Float64}(sol.x[n+1:2n]); w = y - x; d = norm(w)
    (vector = w, distance = d, x = x, y = y, status = d > tol * max(1.0, norm(x), norm(y)) ? :separate : :coincident)
end

# Unit vectors orthogonal to the directions of both loci (± each basis vector): deterministic candidates when the loci meet.
function _locus_complement(La::ReferenceLocus, Lb::ReferenceLocus)
    ha, hb = affine_hull(La.set), affine_hull(Lb.set); n = ambient_dim(La)
    (ha === nothing || hb === nothing) && return Vector{Float64}[]
    S = hcat(ha.basis, hb.basis)
    C = size(S, 2) == 0 ? Matrix{Float64}(I, n, n) : Matrix{Float64}(nullspace(S'; atol = 1e-9))
    out = Vector{Float64}[]
    for j in 1:size(C, 2); push!(out, C[:, j]); push!(out, -C[:, j]); end
    out
end

Base.show(io::IO, L::ReferenceLocus) = print(io, "ReferenceLocus(", L.kind, " in ℝ^", ambient_dim(L), ")")
