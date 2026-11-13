#*******************************************************************************
#    					 Transfer between meshes and physics
#*******************************************************************************
#
# Values of a finite-element function of a source space at the dofs (node, component) of a destination: the
# Dirichlet dofs of a tag of another space (Coupling) or the dofs of another space (Transfer).
#
# Two phases, kept apart so that the second one can run on GPU arrays and on distributed vectors:
#   - construction (once, CPU, the only place where the geometry is used): every destination node is located in the
#     source mesh and the result is stored in flat arrays of concrete type (`PointLocation`);
#   - application (every stage): linear algebra on vectors only, a gather when all the destination nodes coincide
#     with source nodes (4 bytes per destination dof) or a CSR product otherwise. No allocations.
#
# Columns are numbered over the concatenation [free dofs; Dirichlet dofs] of the source space, the layout of a
# distributed sparse matrix: `findnz(sparse(L))` gives the triplets to assemble one.

"""
    PointLocation

Linear map from the dof values of a source space (free and Dirichlet) to the values of the finite-element function
at `n` destination dofs: `y = L * [x_free; x_dirichlet]`. Built once by [`PointLocation(V, points, node, comp)`](@ref)
and applied with [`transfer!`](@ref) and [`transfer_transpose!`](@ref).

Storage: when every destination dof coincides with a source dof the map is a gather and only `col` is stored
(`is_gather(L)`); otherwise CSR (`rowptr`, `col`, `val`). The arrays are flat and of concrete type and the
application loops are generic in the array type.
"""
struct PointLocation{VI<:AbstractVector{<:Integer},VV<:AbstractVector{<:Real}}
  rowptr::VI          # CSR row pointers; empty for a gather
  col::VI             # source dof in [free; Dirichlet] numbering
  val::VV             # weights; empty for a gather
  ntargets::Int
  nfree::Int
  ndirichlet::Int
  nmatched::Int       # destination dofs matched to a coincident source node (exact unit rows)
end

is_gather(L::PointLocation) = isempty(L.val)
Base.size(L::PointLocation) = (L.ntargets, L.nfree + L.ndirichlet)
Base.size(L::PointLocation, i::Integer) = size(L)[i]
SparseArrays.nnz(L::PointLocation) = length(L.col)

"The map as a sparse matrix `ntargets × (nfree + ndirichlet)` (triplets for a distributed assembly: `findnz`)."
function SparseArrays.sparse(L::PointLocation)
  n, m = size(L)
  if is_gather(L)
    sparse(collect(1:n), collect(Int, L.col), ones(n), n, m)
  else
    I = Vector{Int}(undef, nnz(L))
    for i in 1:n, k in L.rowptr[i]:L.rowptr[i+1]-1
      I[k] = i
    end
    sparse(I, collect(Int, L.col), collect(Float64, L.val), n, m)
  end
end

"""
    transfer!(y, L::PointLocation, x_free, x_dirichlet)

`y = L * [x_free; x_dirichlet]`: the source function at the destination dofs. No allocations.
"""
function transfer!(y::AbstractVector, L::PointLocation, xf::AbstractVector, xd::AbstractVector)
  nf = L.nfree
  col = L.col
  if is_gather(L)
    @inbounds for i in eachindex(col)
      c = col[i]
      y[i] = c ≤ nf ? xf[c] : xd[c-nf]
    end
  else
    rowptr, val = L.rowptr, L.val
    @inbounds for i in 1:L.ntargets
      s = zero(eltype(y))
      for k in rowptr[i]:rowptr[i+1]-1
        c = col[k]
        s += val[k] * (c ≤ nf ? xf[c] : xd[c-nf])
      end
      y[i] = s
    end
  end
  y
end

"""
    transfer_transpose!(f_free, L::PointLocation, g)

`f_free = (Lᵀ g)` restricted to the free dofs of the source: the conservative transfer of a load `g` given at the
destination dofs (the transpose of the transfer of the displacements; the resultant is kept because the rows of `L`
add up to one). The part of `Lᵀ g` on the Dirichlet dofs of the source is dropped (reactions). No allocations.
"""
function transfer_transpose!(ff::AbstractVector, L::PointLocation, g::AbstractVector)
  nf = L.nfree
  col = L.col
  fill!(ff, zero(eltype(ff)))
  if is_gather(L)
    @inbounds for i in eachindex(col)
      c = col[i]
      c ≤ nf && (ff[c] += g[i])
    end
  else
    rowptr, val = L.rowptr, L.val
    @inbounds for i in 1:L.ntargets
      gi = g[i]
      for k in rowptr[i]:rowptr[i+1]-1
        c = col[k]
        c ≤ nf && (ff[c] += val[k] * gi)
      end
    end
  end
  ff
end

# ---------------------------------------------------------------------------------------------------------------------
# Construction

_bucket(x, s) = map(xk -> floor(Int, xk / s), Tuple(x))

"Source nodes inside the bounding box of the destination points, hashed by coordinates (buckets of side 4h)."
struct _NodeHash{D}
  first::Dict{NTuple{D,Int},Int32}    # bucket → first node of its chain
  next::Vector{Int32}                 # chain
  x::Vector{VectorValue{D,Float64}}
  cell::Vector{Int32}
  lnode::Vector{Int16}
  dof::Vector{Int}                    # global dof of the first component (identity of the node)
  ambiguous::Vector{Bool}             # two different dofs at the same coordinates (DG, cracks)
  h::Float64
end

function _find(H::_NodeHash{D}, x) where {D}
  s = 4 * H.h
  b = _bucket(x, s)
  best = 0
  for δ in Iterators.product(ntuple(_ -> -1:1, D)...)
    i = get(H.first, b .+ δ, Int32(0))
    while i != 0
      if norm(H.x[i] - x) ≤ H.h
        best = Int(i)
        break
      end
      i = H.next[i]
    end
    best != 0 && break
  end
  best
end

function _node_hash(cmap, ids, nodes, ldof, lo, hi, h, ::Val{D}) where {D}
  H = _NodeHash{D}(Dict{NTuple{D,Int},Int32}(), Int32[], VectorValue{D,Float64}[], Int32[], Int16[], Int[], Bool[], h)
  s = 4h
  cmap_cache = array_cache(cmap)
  ids_cache = array_cache(ids)
  for cell in 1:length(cmap)
    xs = evaluate(getindex!(cmap_cache, cmap, cell), nodes)
    idc = getindex!(ids_cache, ids, cell)
    for n in eachindex(xs)
      x = xs[n]
      all(k -> lo[k] - h ≤ x[k] ≤ hi[k] + h, 1:D) || continue
      g = Int(idc[ldof[n, 1]])
      j = _find(H, x)
      if j != 0
        H.dof[j] == g || (H.ambiguous[j] = true)
        continue
      end
      push!(H.x, x); push!(H.cell, cell); push!(H.lnode, n); push!(H.dof, g); push!(H.ambiguous, false)
      b = _bucket(x, s)
      push!(H.next, get(H.first, b, Int32(0)))
      H.first[b] = Int32(length(H.x))
    end
  end
  H
end

# Reference coordinates of `x` in a cell: Newton iterations on top of Gridap's inverse map, whose iterative solver
# (non-affine cells) stops at ~1e-8. Cells of the dimension of the space only (square Jacobian); a no-op when the
# inverse map is exact (affine cells).
function _polish(φ, ξ::Point{D}, x::Point{D}, scale) where {D}
  for _ in 1:3
    r = x - evaluate(φ, ξ)
    norm(r) ≤ 1e-15 * scale && break
    ξ = ξ + r ⋅ inv(evaluate(∇(φ), ξ))
  end
  ξ
end
_polish(φ, ξ, x, scale) = ξ

"""
    PointLocation(V, points, node, comp; ncomp=…, match=:auto, tol=1e-10)

Locate the destination dofs in the source space `V` (Lagrangian, scalar or vector valued; free and Dirichlet dofs).
Destination dof `j` is the component `comp[j]` at `points[node[j]]`.

- `match = :auto`: a destination node that coincides with a source node (distance ≤ `tol` × size of the
  coordinates) is matched to it, with an exact unit weight; the others are searched in the cells of the source
  mesh. `match = :search`: every node is searched in the cells.
- A searched node must lie in a source cell: the point mapped back from the reference coordinates is compared with
  the node and an error is thrown if they differ (a node outside the source mesh is never extrapolated).
- `ncomp`: number of components of the destination, checked against the source.

The search in the cells uses Gridap's KD-tree point search, once, here; nothing of it is kept.
"""
function PointLocation(V::SingleFieldFESpace, points::AbstractVector{<:Point}, node::AbstractVector{<:Integer},
                       comp::AbstractVector{<:Integer}; ncomp::Integer=maximum(comp; init=1), match::Symbol=:auto, tol::Real=1e-10)
  match in (:auto, :search) || error("HyperFEM: PointLocation: match = :$match is not supported (:auto, :search)")
  length(node) == length(comp) || error("HyperFEM: PointLocation: node and comp must have the same length")
  nodes, ldof, ncomp_src = _nodal_layout(V, "source")
  ncomp == ncomp_src ||
    error("HyperFEM: PointLocation: the destination has $ncomp components and the source space has $ncomp_src")
  nfree, ndir = num_free_dofs(V), num_dirichlet_dofs(V)
  nfree + ndir ≤ typemax(Int32) || error("HyperFEM: PointLocation: too many source dofs for Int32 indices")
  n = length(node)
  npts = length(points)
  trian = get_triangulation(V)
  cmap = get_cell_map(trian)
  ids = get_cell_dof_ids(V)
  D = length(eltype(points))
  # per point: either a matched (cell, local node), or a cell and the weights of its nodes
  pcell = zeros(Int32, npts)
  plnode = zeros(Int16, npts)                 # > 0: matched node
  pw = Vector{Vector{Float64}}(undef, npts)   # weights of the searched points
  nmatched_pts = 0
  if npts > 0
    lo = ntuple(k -> minimum(x -> x[k], points), D)
    hi = ntuple(k -> maximum(x -> x[k], points), D)
    scale = max(maximum(abs, lo), maximum(abs, hi), 1.0)
    h = tol * scale
    if match === :auto
      H = _node_hash(cmap, ids, nodes, ldof, lo, hi, h, Val(D))
      for (p, x) in enumerate(points)
        j = _find(H, x)
        j == 0 && continue
        H.ambiguous[j] &&
          error("HyperFEM: PointLocation: the source space has several dofs at the node $(H.x[j]) (discontinuous space or duplicated nodes); use match = :search")
        pcell[p] = H.cell[j]; plnode[p] = H.lnode[j]
        nmatched_pts += 1
      end
    end
    if nmatched_pts < npts
      search = Gridap.CellData._point_to_cell_cache(Gridap.CellData.KDTreeSearch(), trian)
      cmap_cache = array_cache(cmap)
      for (p, x) in enumerate(points)
        plnode[p] > 0 && continue
        cell = try
          Gridap.CellData._point_to_cell!(search, x)
        catch
          error("HyperFEM: PointLocation: the destination node $x is not in any cell of the source mesh")
        end
        φ = getindex!(cmap_cache, cmap, cell)
        ξ = _polish(φ, evaluate(inverse_map(φ), x), x, scale)
        norm(evaluate(φ, ξ) - x) ≤ 1e-8 * scale ||
          error("HyperFEM: PointLocation: the destination node $x is not on the source mesh (it is at a distance $(norm(evaluate(φ, ξ) - x)) of its closest cell)")
        pcell[p] = cell
        pw[p] = vec(_scalar_shapefuns(V, [ξ], ldof))
      end
    end
  end
  gather = nmatched_pts == npts
  ids_cache = array_cache(ids)
  _col(g) = Int32(g > 0 ? g : nfree - g)
  if gather
    col = Vector{Int32}(undef, n)
    for j in 1:n
      p = node[j]
      idc = getindex!(ids_cache, ids, pcell[p])
      col[j] = _col(idc[ldof[plnode[p], comp[j]]])
    end
    return PointLocation(Int32[], col, Float64[], n, nfree, ndir, n)
  end
  rowptr = Vector{Int32}(undef, n + 1)
  col = Int32[]; val = Float64[]
  sizehint!(col, n * length(nodes)); sizehint!(val, n * length(nodes))
  nmatched = 0
  for j in 1:n
    rowptr[j] = length(col) + 1
    p = node[j]
    idc = getindex!(ids_cache, ids, pcell[p])
    if plnode[p] > 0
      push!(col, _col(idc[ldof[plnode[p], comp[j]]])); push!(val, 1.0)
      nmatched += 1
    else
      w = pw[p]
      for m in eachindex(w)
        abs(w[m]) ≤ 1e-14 && continue
        push!(col, _col(idc[ldof[m, comp[j]]])); push!(val, w[m])
      end
    end
  end
  rowptr[n+1] = length(col) + 1
  PointLocation(rowptr, col, val, n, nfree, ndir, nmatched)
end

PointLocation(U::TrialFESpace, args...; kwargs...) = PointLocation(U.space, args...; kwargs...)

# ---------------------------------------------------------------------------------------------------------------------
# Transfer between two spaces

"Node and component of every free dof of a Lagrangian space: (points, node of each dof, component of each dof, ncomp)."
function _free_dof_targets(V::SingleFieldFESpace)
  nodes, ldof, ncomp = _nodal_layout(V, "destination")
  cmap = get_cell_map(get_triangulation(V))
  ids = get_cell_dof_ids(V)
  nf = num_free_dofs(V)
  P = eltype(evaluate(Gridap.Arrays.testitem(cmap), nodes))
  points = P[]
  index = Dict{P,Int}()
  node = zeros(Int, nf); comp = zeros(Int, nf)
  cmap_cache = array_cache(cmap); ids_cache = array_cache(ids)
  for cell in 1:length(cmap)
    xs = evaluate(getindex!(cmap_cache, cmap, cell), nodes)
    idc = getindex!(ids_cache, ids, cell)
    for c in 1:ncomp, m in eachindex(xs)
      g = idc[ldof[m, c]]
      (g > 0 && node[g] == 0) || continue
      node[g] = get!(index, xs[m]) do
        push!(points, xs[m]); length(points)
      end
      comp[g] = c
    end
  end
  points, node, comp, ncomp
end

"""
    Transfer(U_src, V_dst; match=:auto, tol=1e-10)

Transfer of finite-element functions from the space `U_src` to the free dofs of the space `V_dst` (both
Lagrangian, on any two meshes or triangulations: a volume and an interface, two non-matching meshes), built once:

    T = Transfer(U_src, V_dst)
    transfer!(y, T, x_free, x_dirichlet)     # y: free dof values in V_dst of the function of U_src
    transfer!(y, T, uh)                      # the same from an FEFunction of U_src
    transfer_transpose!(f, T, g)             # loads: g at the free dofs of V_dst → f at the free dofs of U_src

See [`PointLocation`](@ref) for `match` and `tol`.
"""
struct Transfer{L<:PointLocation}
  location::L
end

function Transfer(U_src::SingleFieldFESpace, V_dst::SingleFieldFESpace; kwargs...)
  points, node, comp, ncomp = _free_dof_targets(V_dst)
  Transfer(PointLocation(U_src, points, node, comp; ncomp, kwargs...))
end

Base.size(T::Transfer, args...) = size(T.location, args...)
SparseArrays.sparse(T::Transfer) = sparse(T.location)
is_gather(T::Transfer) = is_gather(T.location)

transfer!(y::AbstractVector, T::Transfer, xf::AbstractVector, xd::AbstractVector) = transfer!(y, T.location, xf, xd)
transfer!(y::AbstractVector, T::Union{Transfer,PointLocation}, uh::FEFunction) =
  transfer!(y, T, get_free_dof_values(uh), uh.dirichlet_values)
transfer_transpose!(ff::AbstractVector, T::Transfer, g::AbstractVector) = transfer_transpose!(ff, T.location, g)
