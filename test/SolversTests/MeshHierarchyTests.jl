# S4-B step 1 (design/solvers-s4b-plan.md §3.1–3.2): `MeshHierarchy` by uniform refinement and the prolongation
# matrices between its levels. The prolongation is the exact interpolation of a coarse FE function in the fine space
# (free dofs, homogeneous Dirichlet), built by insertion at shared nodes; the levels must share the closed Dirichlet
# boundary; a fine node whose coarse parents are all Dirichlet gets a zero row; on affine cells the Galerkin product
# PᵀAP equals the rediscretised linear-elasticity operator of the coarse level to round-off.
using Gridap.Adaptivity, Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField
using GridapGmsh
using LinearAlgebra, SparseArrays

const MH_MODELS = joinpath(@__DIR__, "..", "models")

"Vector Lagrangian test space of order `p` with Dirichlet on `tags`."
mh_space(m, p, tags; T=VectorValue{num_cell_dims(m),Float64}) =
  TestFESpace(m, ReferenceFE(lagrangian, T, p); dirichlet_tags=tags, conformity=:H1)

"Linear elasticity operator on `V` (constant coefficients, quadrature degree 2)."
function mh_elasticity(V)
  dΩ = Measure(Triangulation(get_background_model(get_triangulation(V))), 2)
  λ, μ = 2.0, 1.0
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩ
  assemble_matrix(a, TrialFESpace(V), V)
end

@testset "MeshHierarchy and prolongations (S4-B step 1)" begin

  @testset "hierarchy: plain models, cell counts, inherited labels" begin
    coarse = GmshDiscreteModel(joinpath(MH_MODELS, "s4b_tet.msh"))
    h = MeshHierarchy(coarse, 3)
    @test num_levels(h) == 3
    @test all(m -> m isa UnstructuredDiscreteModel, h.models)      # plain models for the drivers
    @test [num_cells(level_model(h, l)) for l in 1:3] == [98, 784, 6272]
    @test fine_model(h) === level_model(h, 3)
    @test length(level_glue(h, 2).n2o_faces_map[end]) == 6272
    @test "left" in get_face_labeling(fine_model(h)).tag_to_name
    @test_throws ErrorException MeshHierarchy(coarse, 0)
    @test_throws ErrorException MeshHierarchy(coarse, 2; refinement=1)
    h1 = MeshHierarchy(coarse, 1)
    @test num_levels(h1) == 1 && isempty(h1.glues)
  end

  @testset "prolongation is the exact interpolation of a coarse function" begin
    for (file, p) in (("s4b_tet.msh", 1), ("s4b_hex.msh", 1), ("s4b_hex_distorted.msh", 1), ("s4b_tri.msh", 1),
                      ("s4b_quad.msh", 1), ("s4b_tet.msh", 2), ("s4b_hex.msh", 2))
      coarse = GmshDiscreteModel(joinpath(MH_MODELS, file))
      D = num_cell_dims(coarse)
      h = MeshHierarchy(coarse, 3)
      # the reference interpolation needs Gridap's adapted models (change of domain through the glue); the spaces on
      # them have the same dofs as on the plain models of the hierarchy
      a1 = refine(coarse, 2); a2 = refine(a1, 2)
      adapted = (coarse, a1, a2)
      for tags in (String[], ["left"])
        Vs = [mh_space(m, p, tags) for m in h.models]
        Va = [mh_space(m, p, tags) for m in adapted]
        # a smooth field, zero on x = 0 when "left" is Dirichlet (the prolongation carries homogeneous data)
        g(x) = (isempty(tags) ? 1.0 : x[1]) * VectorValue(ntuple(i -> i * (1 + sin(2x[1]) * cos(x[2]) + (D == 3 ? x[3]^2 : 0.0)), D))
        for l in 1:2
          P = prolongation(Vs[l], Vs[l+1], level_glue(h, l))
          @test size(P) == (num_free_dofs(Vs[l+1]), num_free_dofs(Vs[l]))
          @test P == prolongation(Va[l], Va[l+1], level_glue(h, l))
          Uc, Uf = TrialFESpace(Va[l], g), TrialFESpace(Va[l+1], g)
          uc = interpolate(g, Uc)                          # a coarse FE function
          uf = interpolate(uc, Uf)                         # the same function in the fine space (Gridap, through the glue)
          @test norm(P * get_free_dof_values(uc) - get_free_dof_values(uf)) ≤ 1e-12 * max(1.0, norm(get_free_dof_values(uf)))
          # insertion: at order 1 every entry is a shape-function value in [0, 1] (an accumulated shared node would
          # give 2s) and no row exceeds one; without Dirichlet dofs every row sums to one (partition of unity)
          rs = vec(sum(P; dims=2))
          if p == 1
            @test all(0 ≤ w ≤ 1 + 1e-12 for w in nonzeros(P))
            @test all(r -> r ≤ 1 + 1e-10, rs)
          end
          isempty(tags) && @test all(r -> abs(r - 1) ≤ 1e-10, rs)
        end
      end
    end
  end

  @testset "prolongations(h, space) and the restriction" begin
    coarse = GmshDiscreteModel(joinpath(MH_MODELS, "s4b_hex.msh"))
    h = MeshHierarchy(coarse, 3)
    Ps = prolongations(h, m -> mh_space(m, 1, ["left"]))
    @test length(Ps) == 2
    Vs = [mh_space(m, 1, ["left"]) for m in h.models]
    @test size(Ps[2]) == (num_free_dofs(Vs[3]), num_free_dofs(Vs[2]))
    x = randn(size(Ps[1], 2)); y = randn(size(Ps[1], 1))
    @test dot(y, Ps[1] * x) ≈ dot(transpose(Ps[1]) * y, x)
  end

  @testset "Galerkin product equals the rediscretised coarse operator on affine cells" begin
    for file in ("s4b_tet.msh", "s4b_tri.msh")
      coarse = GmshDiscreteModel(joinpath(MH_MODELS, file))
      h = MeshHierarchy(coarse, 2)
      Vc, Vf = mh_space(coarse, 1, ["left"]), mh_space(fine_model(h), 1, ["left"])
      P = prolongation(Vc, Vf, level_glue(h, 1))
      Ac, Af = mh_elasticity(Vc), mh_elasticity(Vf)
      @test norm(transpose(P) * Af * P - Ac) ≤ 1e-11 * norm(Ac)
    end
    # Cartesian hexahedra (affine) through Gridap's Cartesian refinement
    coarse = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
    h = MeshHierarchy(coarse, 2)
    Vc, Vf = mh_space(coarse, 1, ["tag_01"]), mh_space(fine_model(h), 1, ["tag_01"])
    P = prolongation(Vc, Vf, level_glue(h, 1))
    Ac, Af = mh_elasticity(Vc), mh_elasticity(Vf)
    @test norm(transpose(P) * Af * P - Ac) ≤ 1e-11 * norm(Ac)
    # non-affine hexahedra: the spaces are nested but the quadrature is not exact for the rational integrand
    coarse = GmshDiscreteModel(joinpath(MH_MODELS, "s4b_hex_distorted.msh"))
    h = MeshHierarchy(coarse, 2)
    Vc, Vf = mh_space(coarse, 1, ["left"]), mh_space(fine_model(h), 1, ["left"])
    P = prolongation(Vc, Vf, level_glue(h, 1))
    Ac, Af = mh_elasticity(Vc), mh_elasticity(Vf)
    @test norm(transpose(P) * Af * P - Ac) ≤ 2e-2 * norm(Ac)
  end

  @testset "multi-field: block-diagonal prolongation" begin
    coarse = GmshDiscreteModel(joinpath(MH_MODELS, "s4b_tet.msh"))
    h = MeshHierarchy(coarse, 2)
    # ("right" is an open face tag in these meshes: only "left" is closed, so only "left" can be Dirichlet)
    mk(m) = MultiFieldFESpace([mh_space(m, 1, ["left"]), mh_space(m, 1, ["left"]; T=Float64)])
    Xc, Xf = mk(coarse), mk(fine_model(h))
    P = prolongation(Xc, Xf, level_glue(h, 1))
    Pu = prolongation(Xc.spaces[1], Xf.spaces[1], level_glue(h, 1))
    Pφ = prolongation(Xc.spaces[2], Xf.spaces[2], level_glue(h, 1))
    @test P == blockdiag(Pu, Pφ)
    @test size(P) == (num_free_dofs(Xf), num_free_dofs(Xc))
  end

  @testset "Dirichlet boundaries: open tag refused, all-Dirichlet parents give a zero row" begin
    # the open face tag of step 0: Dirichlet sets differ across levels
    coarse = GmshDiscreteModel(joinpath(MH_MODELS, "s4b_tet_open.msh"))
    h = MeshHierarchy(coarse, 2)
    Vc, Vf = mh_space(coarse, 1, ["left"]), mh_space(fine_model(h), 1, ["left"])
    @test_throws ErrorException prolongation(Vc, Vf, level_glue(h, 1))
    # Dirichlet on the four corners of x = 0 only (a closed set of vertices) on a one-cell coarse mesh: the fine
    # mid-edge nodes between two corners and the face centre are free with no free parent → zero rows; the
    # prolongation is still exact for a coarse function vanishing at the corners
    coarse = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (1, 1, 1))
    labels = get_face_labeling(coarse)
    add_tag_from_tags!(labels, "corners", [1, 3, 5, 7])
    h = MeshHierarchy(coarse, 2)
    a1 = refine(coarse, 2)
    Vc, Vf = mh_space(coarse, 1, ["corners"]), mh_space(a1, 1, ["corners"])
    P = prolongation(Vc, Vf, level_glue(h, 1))
    @test P == prolongation(Vc, mh_space(fine_model(h), 1, ["corners"]), level_glue(h, 1))
    rs = vec(sum(abs, P; dims=2))
    @test count(iszero, rs) == 3 * 5                      # 4 mid-edge nodes + the centre of the x = 0 face, 3 components
    g(x) = VectorValue(x[2] * (1 - x[2]) + x[3] * (1 - x[3]), x[1], x[2] * x[3] * (1 - x[2]))
    Uc, Uf = TrialFESpace(Vc, g), TrialFESpace(Vf, g)
    uc = interpolate(g, Uc)
    uf = interpolate(uc, Uf)
    @test norm(P * get_free_dof_values(uc) - get_free_dof_values(uf)) ≤ 1e-12
  end
end
