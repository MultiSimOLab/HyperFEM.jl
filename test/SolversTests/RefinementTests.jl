# S4-B step 0 (design/solvers-s4b-status.md): Gridap's `refine` gives a usable mesh hierarchy on Gmsh meshes of
# every element type HyperFEM uses. Checks per level: cell count ×2^D, volume preserved, every child inside its parent
# (the glue is complete), no degenerate cell, boundary labels inherited, conforming P1/Q1 spaces on every level, exact
# coarse-to-fine interpolation of a linear (simplices) / multilinear (n-cubes) field, multi-field spaces. The meshes
# are small boxes generated with Gmsh and kept in test/models (s4b_*.msh): tetrahedra, hexahedra (affine and with
# displaced interior nodes), triangles and quadrilaterals, with the Dirichlet group "left" tagged as a closed set
# (face + edges + vertices), which the hierarchy needs: with an open face tag the Dirichlet sets differ across levels.
using Gridap.Adaptivity, Gridap.Geometry, Gridap.CellData, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Arrays
using GridapGmsh
using LinearAlgebra

const REF_MODELS = joinpath(@__DIR__, "..", "models")

ref_cellvols(model) = get_array(∫(1)Measure(Triangulation(model), 2))

"det J of the cell map at the reference vertices of every cell: (min, max, number of sign-changing cells)."
function ref_detJ(model)
  cm = get_cell_map(get_grid(model))
  pts = get_vertex_coordinates(first(get_polytopes(model)))
  d = [extrema(det, c) for c in lazy_map(evaluate, lazy_map(∇, cm), fill(pts, num_cells(model)))]
  (minimum(first, d), maximum(last, d), count(x -> x[1] * x[2] <= 0, d))
end

"Largest ratio longest/shortest edge over the cells."
function ref_edge_ratio(model)
  X = get_node_coordinates(get_grid(model))
  edges = get_faces(first(get_polytopes(model)), 1, 0)
  maximum(get_cell_node_ids(get_grid(model))) do nodes
    l = [norm(X[nodes[e[1]]] - X[nodes[e[2]]]) for e in edges]
    maximum(l) / minimum(l)
  end
end

ref_nodes_on_left(model) = count(x -> abs(x[1]) < 1e-12, get_node_coordinates(get_grid(model)))

function check_refined_level(cmodel, fmodel, D, ncube)
  nc, nf = num_cells(cmodel), num_cells(fmodel)
  @test nf == 2^D * nc
  cv, fv = ref_cellvols(cmodel), ref_cellvols(fmodel)
  @test sum(fv) ≈ sum(cv) rtol = 1e-12
  glue = get_adaptivity_glue(fmodel)
  f2c = glue.n2o_faces_map[end]
  @test length(f2c) == nf
  counts = zeros(Int, nc); pv = zeros(nc)
  for (f, c) in enumerate(f2c)
    counts[c] += 1; pv[c] += fv[f]
  end
  @test all(==(2^D), counts)                              # 2^D children per parent
  @test maximum(abs.(pv .- cv) ./ cv) < 1e-10             # children cover the parent exactly
  @test ref_detJ(fmodel)[3] == 0                          # no degenerate cell
  # labels inherited: the "left" face keeps its measure, and every node on x = 0 is Dirichlet (closed tag)
  area(m) = sum(∫(1)Measure(BoundaryTriangulation(m; tags="left"), 2))
  @test area(fmodel) ≈ area(cmodel) rtol = 1e-12
  # spaces and exact interpolation
  f(x) = D == 3 ? (ncube ? 1 + 2x[1] - x[2] + 3x[3] + x[1] * x[2] * x[3] : 1 + 2x[1] - x[2] + 3x[3]) :
                  (ncube ? 1 + 2x[1] - x[2] + x[1] * x[2] : 1 + 2x[1] - x[2])
  g(x) = VectorValue(ntuple(i -> i * f(x), D))
  dΩf = Measure(Triangulation(fmodel), 4)
  for (T, h, nc_) in ((Float64, f, 1), (VectorValue{D,Float64}, g, D))
    Vc = TestFESpace(cmodel, ReferenceFE(lagrangian, T, 1); dirichlet_tags="left")
    Vf = TestFESpace(fmodel, ReferenceFE(lagrangian, T, 1); dirichlet_tags="left")
    @test num_dirichlet_dofs(Vc) == nc_ * ref_nodes_on_left(cmodel)
    @test num_dirichlet_dofs(Vf) == nc_ * ref_nodes_on_left(fmodel)
    uc = interpolate(h, TrialFESpace(Vc, h))
    uf = interpolate(uc, TrialFESpace(Vf, h))             # coarse FE function into the fine space, through the glue
    e = uf - uc
    @test sqrt(abs(sum(∫(e ⋅ e)dΩf))) < 1e-12
  end
  Vu = TestFESpace(fmodel, ReferenceFE(lagrangian, VectorValue{D,Float64}, 1); dirichlet_tags="left")
  Vφ = TestFESpace(fmodel, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags=["left", "right"])
  @test num_free_dofs(MultiFieldFESpace([Vu, Vφ])) == num_free_dofs(Vu) + num_free_dofs(Vφ)
end

@testset "Refinement hierarchy on Gmsh meshes (S4-B step 0)" begin
  cases = (("s4b_tet.msh", 3, false, 3), ("s4b_hex.msh", 3, true, 2), ("s4b_hex_distorted.msh", 3, true, 2),
           ("s4b_tri.msh", 2, false, 2), ("s4b_quad.msh", 2, true, 2))
  for (file, D, ncube, levels) in cases
    @testset "$file" begin
      model = GmshDiscreteModel(joinpath(REF_MODELS, file))
      m = model
      r0 = ref_edge_ratio(model)
      for l in 1:levels
        fm = refine(m, 2)
        @test fm isa AdaptedDiscreteModel
        check_refined_level(m, fm, D, ncube)
        m = fm
      end
      # shape regularity: n-cubes keep their edge ratio; tetrahedra degrade by a bounded factor per level (no Bey rule)
      r = ref_edge_ratio(m)
      if ncube
        @test r ≈ r0 rtol = 1e-10
      else
        @test r < 3.0 * r0
      end
    end
  end

  @testset "red_green gives the same counts and interpolation on tetrahedra" begin
    model = GmshDiscreteModel(joinpath(REF_MODELS, "s4b_tet.msh"))
    fm = refine(model; refinement_method="red_green")
    check_refined_level(model, fm, 3, false)
  end

  @testset "an open face tag gives different Dirichlet sets across levels" begin
    # the same tetrahedral box with "left" on the face only (no edges, no vertices): the coarse level has no
    # Dirichlet node on x = 0, the fine one has some — the trap of S4-A's Prolongation check, now on a hierarchy
    model = GmshDiscreteModel(joinpath(REF_MODELS, "s4b_tet_open.msh"))
    fm = refine(model, 2)
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags="left")
    Vf = TestFESpace(fm, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags="left")
    @test num_dirichlet_dofs(Vc) < ref_nodes_on_left(model)
    @test num_dirichlet_dofs(Vf) > 0
    @test num_dirichlet_dofs(Vf) < ref_nodes_on_left(fm)
  end
end
