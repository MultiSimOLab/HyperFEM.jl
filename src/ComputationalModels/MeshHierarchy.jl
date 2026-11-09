# Mesh hierarchy by uniform refinement and the prolongation matrices between its levels (h-multigrid, S4-B;
# design/solvers-s4b-plan.md §3.1–3.2). Gridap only: no PETSc, no assembly engine.

"""
    MeshHierarchy(coarse::DiscreteModel, nlevels; refinement = 2)

`nlevels` nested meshes, coarse first, obtained by refining `coarse` uniformly `nlevels - 1` times with Gridap's
`refine(model, refinement)` (every cell split into `refinement^D` children; tetrahedra, hexahedra, triangles and
quadrilaterals, Gmsh or Cartesian). The models are plain (`UnstructuredDiscreteModel`), so drivers, assembly backends
and post-processing see what they see today; the refinement glues are kept apart (`level_glue(h, l)` maps level `l + 1`
onto level `l`). Boundary labels are inherited by every level. Dirichlet tags must be **closed** sets (a face with
its edges and vertices under the same name) so that the Dirichlet nodes are the same geometric set on every level;
`prolongation` checks it.

    h = MeshHierarchy(GmshDiscreteModel("coarse.msh"), 4)      # 4 levels, ×512 cells
    model = fine_model(h)                                      # the working mesh of the driver
"""
struct MeshHierarchy{M<:DiscreteModel,G}
  models::Vector{M}
  glues::Vector{G}
  function MeshHierarchy(coarse::DiscreteModel, nlevels::Integer; refinement::Integer=2)
    nlevels ≥ 1 || error("HyperFEM: MeshHierarchy needs at least one level (got $nlevels)")
    refinement ≥ 2 || error("HyperFEM: MeshHierarchy: refinement must be ≥ 2 (got $refinement)")
    models = DiscreteModel[coarse]
    glues = Any[]
    m = coarse
    for _ in 2:nlevels
      a = Gridap.Adaptivity.refine(m, refinement)
      push!(glues, Gridap.Adaptivity.get_adaptivity_glue(a))
      m = Gridap.Adaptivity.get_model(a)
      push!(models, m)
    end
    M = mapreduce(typeof, typejoin, models)
    G = isempty(glues) ? Any : mapreduce(typeof, typejoin, glues)
    new{M,G}(convert(Vector{M}, models), convert(Vector{G}, glues))
  end
end

num_levels(h::MeshHierarchy) = length(h.models)
"The model of level `l` (1 = coarsest)."
level_model(h::MeshHierarchy, l::Integer) = h.models[l]
"The finest model: the working mesh."
fine_model(h::MeshHierarchy) = last(h.models)
"The glue of level `l + 1` onto level `l`."
level_glue(h::MeshHierarchy, l::Integer) = h.glues[l]

function Base.show(io::IO, h::MeshHierarchy)
  n = [num_cells(m) for m in h.models]
  print(io, "MeshHierarchy with $(num_levels(h)) levels: ", join(n, " → "), " cells")
end

# ---------------------------------------------------------------------------------------------------------------------
# Prolongation between the free dofs of two consecutive levels

"Lagrangian dof basis of a nodal single-field space, and its node/component layout."
function _nodal_layout(V::SingleFieldFESpace, what::String)
  db = Gridap.Arrays.testitem(Gridap.CellData.get_data(get_fe_dof_basis(V)))
  db isa Gridap.ReferenceFEs.LagrangianDofBasis ||
    error("HyperFEM: prolongation: the $what space must be Lagrangian (nodal), got $(nameof(typeof(db)))")
  n2d = db.node_and_comp_to_dof
  ncomp = n2d[1] isa Integer ? 1 : length(n2d[1])
  ldof = [ncomp == 1 ? Int(n2d[n]) : Int(n2d[n][c]) for n in eachindex(n2d), c in 1:ncomp]   # node × comp → local dof
  collect(db.nodes), ldof, ncomp
end

"Scalar shape functions of `V` (one component) at the reference points `pts`: npts × nnodes."
function _scalar_shapefuns(V::SingleFieldFESpace, pts, ldof::Matrix{Int})
  b = Gridap.Arrays.testitem(Gridap.CellData.get_data(get_fe_basis(V)))
  vals = evaluate(b, pts)                                 # npts × ndofs
  N = zeros(length(pts), size(ldof, 1))
  for n in axes(ldof, 1), q in eachindex(pts)
    v = vals[q, ldof[n, 1]]
    N[q, n] = v isa Real ? v : v[1]                      # Gridap's MultiValue <: Number
  end
  N
end

_snap(e) = abs(e) < 1e-7 ? 0.0 : abs(e - 1) < 1e-7 ? 1.0 : e

"""
    prolongation(Vc, Vf, glue) -> SparseMatrixCSC

Prolongation matrix from the free dofs of the coarse space `Vc` to the free dofs of the fine space `Vf`, the two
levels of a `MeshHierarchy` related by `glue` (fine → coarse). `P * xc` is the exact interpolation of the coarse
finite-element function `xc` (free values, homogeneous Dirichlet data) in the fine space: for every fine node the
coarse shape functions are evaluated at it through the refinement rule, the values are **inserted** (a fine node
shared by several cells is written once), coarse Dirichlet columns are dropped (the correction equation is
homogeneous) and fine Dirichlet rows are never written. The restriction is `transpose(P)`. State-independent:
built once per hierarchy. Multi-field spaces (`ConsecutiveMultiFieldStyle`) give the block-diagonal matrix of
their fields.

Errors if the two levels have different Dirichlet boundaries (a fine Dirichlet node interpolated from a free coarse
node): Dirichlet tags must be closed sets on the coarse mesh (face + edges + vertices), as the labels are inherited.
A fine free node whose coarse parents are all Dirichlet gets a zero row (the smoother handles it).
"""
function prolongation(Vc::SingleFieldFESpace, Vf::SingleFieldFESpace, glue)
  nodes_f, ldof_f, ncomp_f = _nodal_layout(Vf, "fine")
  nodes_c, ldof_c, ncomp_c = _nodal_layout(Vc, "coarse")
  ncomp_f == ncomp_c || error("HyperFEM: prolongation: the two levels have different numbers of components")
  ncomp = ncomp_f
  NF, NC = length(nodes_f), length(nodes_c)
  # the fine basis is nodal at its own nodes (so a fine node is a fine dof)
  Nf = _scalar_shapefuns(Vf, nodes_f, ldof_f)
  all(isapprox(Nf[q, n], q == n ? 1.0 : 0.0; atol=1e-8) for q in 1:NF, n in 1:NF) ||
    error("HyperFEM: prolongation: the fine space is not nodal at its dof-basis nodes")
  f2c = glue.n2o_faces_map[end]
  child = glue.n2o_cell_to_child_id
  rules = glue.refinement_rules
  nfine = length(f2c)
  ids_f = get_cell_dof_ids(Vf)
  ids_c = get_cell_dof_ids(Vc)
  length(ids_f) == nfine || error("HyperFEM: prolongation: the fine space does not live on the fine level of the glue")
  length(ids_c) ≥ maximum(f2c) ||
    error("HyperFEM: prolongation: the coarse space does not live on the coarse level of the glue")
  # embedding per (refinement rule, child): coarse shape functions at the fine nodes mapped into the coarse cell
  cache = Dict{Tuple{UInt,Int},Matrix{Float64}}()
  function embedding(c, k)
    rr = rules[c]
    get!(cache, (objectid(rr), k)) do
      cmaps = Gridap.Geometry.get_cell_map(rr)
      pts = evaluate(cmaps[k], nodes_f)
      E = _scalar_shapefuns(Vc, pts, ldof_c)
      E .= _snap.(E)
      E
    end
  end
  # geometry check on a sample of fine cells: fine node = coarse cell map ∘ child map (the glue and the grids agree)
  cmf = get_cell_map(get_triangulation(Vf)); cmc = get_cell_map(get_triangulation(Vc))
  for f in unique((1, nfine ÷ 2 + 1, nfine))
    c, k = f2c[f], child[f]
    pts = evaluate(Gridap.Geometry.get_cell_map(rules[c])[k], nodes_f)
    xf = evaluate(cmf[f], nodes_f); xc = evaluate(cmc[c], pts)
    all(norm(xf[q] - xc[q]) ≤ 1e-10 * (1 + norm(xf[q])) for q in 1:NF) ||
      error("HyperFEM: prolongation: fine cell $f is not the child $k of coarse cell $c (inconsistent hierarchy)")
  end
  I = Int[]; J = Int[]; W = Float64[]
  sizehint!(I, nfine * NF * ncomp); sizehint!(J, nfine * NF * ncomp); sizehint!(W, nfine * NF * ncomp)
  idf = Vector{Int}(undef, NF * ncomp); idc = Vector{Int}(undef, NC * ncomp)
  for f in 1:nfine
    c, k = f2c[f], child[f]
    E = embedding(c, k)
    copyto!(idf, ids_f[f]); copyto!(idc, ids_c[c])
    for comp in 1:ncomp, n in 1:NF
      gf = idf[ldof_f[n, comp]]
      if gf < 0                                           # fine Dirichlet node: no free coarse parent may reach it
        for m in 1:NC
          E[n, m] != 0 && idc[ldof_c[m, comp]] > 0 &&
            error("HyperFEM: prolongation: the two levels have different Dirichlet boundaries (fine Dirichlet node interpolated from a free coarse node). Tag the closure of the Dirichlet boundary on the coarse mesh: the face with its edges and vertices under the same name.")
        end
        continue
      end
      for m in 1:NC
        w = E[n, m]
        w == 0 && continue
        gc = idc[ldof_c[m, comp]]
        gc > 0 || continue                                # coarse Dirichlet: homogeneous, dropped
        push!(I, gf); push!(J, gc); push!(W, w)
      end
    end
  end
  sparse(I, J, W, num_free_dofs(Vf), num_free_dofs(Vc), (a, b) -> a)      # insertion: shared nodes written once
end

function prolongation(Vc::MultiFieldFESpace, Vf::MultiFieldFESpace, glue)
  length(Vc.spaces) == length(Vf.spaces) || error("HyperFEM: prolongation: the two levels have different numbers of fields")
  for V in (Vc, Vf)
    MultiFieldStyle(V) isa ConsecutiveMultiFieldStyle || error("HyperFEM: prolongation: only ConsecutiveMultiFieldStyle multi-field spaces")
  end
  blockdiag([prolongation(sc, sf, glue) for (sc, sf) in zip(Vc.spaces, Vf.spaces)]...)
end
prolongation(Uc::TrialFESpace, Uf::TrialFESpace, glue) = prolongation(Uc.space, Uf.space, glue)

"""
    prolongations(h::MeshHierarchy, space::Function) -> Vector{SparseMatrixCSC}

The prolongation matrices of the hierarchy, `P[l]` from level `l` to level `l + 1`, for the test spaces
`space(model)` built on every level (the same `BoundaryConditions`, inherited tags).

    Ps = prolongations(h, m -> TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D; conformity = :H1))
"""
function prolongations(h::MeshHierarchy, space::Function)
  spaces = [space(m) for m in h.models]
  [prolongation(spaces[l], spaces[l+1], level_glue(h, l)) for l in 1:num_levels(h)-1]
end

# ---------------------------------------------------------------------------------------------------------------------
# The h-multigrid solver from a hierarchy

"""
    HMultigridSolver(h::MeshHierarchy, space::Function; kwargs...)

The [`HMultigridSolver`](@ref) of the hierarchy `h` for the test spaces `space(model)` built on every level (the same
`BoundaryConditions`, inherited tags): the prolongations are built here ([`prolongations`](@ref)).

    space(m) = TestFESpace(Triangulation(m), ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D; conformity = :H1)
    ls = HMultigridSolver(h, space; krylov = :cg, coarse = PETScSolver(:preonly, :gamg; nullspace = near_nullspace(TrialFESpace(space(level_model(h, 1))))))
"""
function Solvers.HMultigridSolver(h::MeshHierarchy, space::Function; kwargs...)
  spaces = [space(m) for m in h.models]
  Ps = [prolongation(spaces[l], spaces[l+1], level_glue(h, l)) for l in 1:num_levels(h)-1]
  Solvers.HMultigridSolver(spaces, Ps; kwargs...)
end
