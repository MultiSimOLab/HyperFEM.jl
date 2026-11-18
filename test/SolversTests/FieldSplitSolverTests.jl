# S4-B step 5 (design/solvers-s4b-plan.md §3.5): `FieldSplitSolver`, the pure-Julia block preconditioner with injected
# block solvers — the S2 battery with exact (LU) blocks, and the h-multigrid V-cycle as the block preconditioner of the
# (u, φ) saddle point on a refined hierarchy: same solution as LU, iterations bounded with the levels, the Newton
# trajectory of LU on nonlinear electro-mechanics with lagged multigrid blocks.
include("krylov_common.jl")
include("fieldsplit_common.jl")
using GridapGmsh
using HyperFEM.Solvers: level_spectra

s2_battery("FieldSplitSolver (LU blocks)", (U; kwargs...) -> FieldSplitSolver(U, (LUSolver(), LUSolver()); kwargs...))

"The saddle point of `s2_saddle_problem` on a refined hierarchy, with the level space builders of the two fields."
function fs_saddle_hierarchy(nlevels)
  coarse = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5, 1, 1))
  labels = get_face_labeling(coarse)
  add_tag_from_tags!(labels, "left", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  h = MeshHierarchy(coarse, nlevels)
  model = fine_model(h)
  space_u(m) = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags=["left"])
  space_φ(m) = TestFESpace(m, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags=["left", "top"])
  Vu, Vφ = space_u(model), space_φ(model)
  U = MultiFieldFESpace([TrialFESpace(Vu, x -> zero(VectorValue{3,Float64})), TrialFESpace(Vφ, [x -> 0.0, x -> 1.0])])
  V = MultiFieldFESpace([Vu, Vφ])
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  λ, μ, εr, κ = 1.0, 1.0, 0.5, 0.3
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  a((u, φ), (v, ψ)) = ∫(ε(v) ⊙ (σ ∘ ε(u)) - εr * (∇(φ) ⋅ ∇(ψ)) + κ * (divergence(v) * φ + divergence(u) * ψ))dΩ
  l((v, ψ)) = ∫(v ⋅ VectorValue(0.0, 0.0, -1.0))dΩ
  op = AffineFEOperator(a, l, U, V)
  (; h, U, V, A=get_matrix(op), b=get_vector(op), space_u, space_φ)
end

"Nonlinear electro-mechanics (the EM problem of S1) on a refined hierarchy, with closed Dirichlet tags on every level."
function fs_em_problem(nlevels)
  coarse = CartesianDiscreteModel((0.0, 1.0, 0.0, 0.2, 0.0, 0.2), (5, 1, 2))
  labels = get_face_labeling(coarse)
  add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
  add_tag_from_tags!(labels, "botsuf", CartesianTags.faceXY0⁺)
  h = MeshHierarchy(coarse, nlevels)
  model = fine_model(h)
  physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
  k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
  Dφ = BoundaryConditions(BoundaryCondition("botsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 0.02, Λ -> Λ))
  D = MultiFieldBC([Du, Dφ])
  space_u(m) = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
  space_φ(m) = TestFESpace(m, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
  Vu, Vφ = space_u(model), space_φ(model)
  V = MultiFieldFESpace([Vu, Vφ])
  U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
  res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
  jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
  (; res, jac, U, V, D, h, space_u, space_φ)
end

"The field-split solver with one h-multigrid V-cycle per block (CG-style symmetric products inside, coarse LU)."
function fs_hmg(t; split=:additive, rtol=1e-10, reuse=1, maxiter=300)
  hmg_u = HMultigridSolver(t.h, t.space_u; krylov=:cg, reuse)
  hmg_φ = HMultigridSolver(t.h, t.space_φ; krylov=:cg, reuse)
  FieldSplitSolver(t.U, (hmg_u, hmg_φ); split, rtol, maxiter)
end

@testset "field split with h-multigrid blocks (S4-B step 5)" begin

  @testset "errors and name" begin
    t = fs_saddle_hierarchy(2)
    r = field_ranges(t.U)
    @test_throws ErrorException FieldSplitSolver(r, (LUSolver(), LUSolver()); krylov=:cg)
    @test_throws ErrorException FieldSplitSolver(r, (LUSolver(),))
    @test_throws ErrorException FieldSplitSolver(r, (LUSolver(), LUSolver()); split=:lower)
    @test_throws ErrorException FieldSplitSolver([1:3, 5:7], (LUSolver(), LUSolver()))
    @test_throws ErrorException FieldSplitSolver(r, (LUSolver(), LUSolver()); negate=1:2)
    hmg_u = HMultigridSolver(t.h, t.space_u; krylov=:cg)
    @test_throws ErrorException FieldSplitSolver(t.U, (LUSolver(), hmg_u))        # the multigrid of the wrong field
    ls = fs_hmg(t)
    @test HyperFEM.Solvers._linear_name(ls) == "GMRES+FS-add(hMG(2),hMG(2))"
    @test HyperFEM.Solvers._linear_name(FieldSplitSolver(t.U, (LUSolver(), LUSolver()); split=:schur)) == "GMRES+FS-schur(LUSolver,LUSolver)"
  end

  @testset "same solution as LU, additive and Schur; iterations bounded with the levels" begin
    its = Dict(:additive => Int[], :schur => Int[])
    for nlevels in (3, 4)
      t = fs_saddle_hierarchy(nlevels)
      xlu = t.A \ t.b
      b0 = copy(t.b)
      for split in (:additive, :schur)
        x, ns = s1_solve(fs_hmg(t; split), t.A, t.b)
        @test linear_converged(ns)
        @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
        @test t.b == b0                                               # the negated equations are restored
        push!(its[split], linear_iterations(ns))
        # the blocks the multigrid saw are SPD (the φ block negated) and the fine levels alias the slices
        for bns in ns.block_ns
          @test all(((lo, hi),) -> lo > 0 && hi > lo, level_spectra(bns))
        end
      end
    end
    @info "field split + hMG, iterations with 3 and 4 levels: $(its)"
    for split in (:additive, :schur)
      @test its[split][2] ≤ 1.3 * its[split][1] + 2
    end
  end

  @testset "lagged multigrid blocks, refilled slices, new matrix object" begin
    t = fs_saddle_hierarchy(3)
    xlu = t.A \ t.b
    ls = fs_hmg(t; reuse=3)                                           # the lag lives in the blocks
    ns = numerical_setup(symbolic_setup(ls, t.A), t.A)
    x = zeros(length(t.b))
    solve!(x, ns, t.b)
    @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
    @test all(preconditioner_builds(bns) == 1 for bns in ns.block_ns)
    A2 = t.A
    A2.nzval .*= 2.0                                                  # same object: the slices are refilled in place
    numerical_setup!(ns, A2)
    @test all(preconditioner_builds(bns) == 1 for bns in ns.block_ns) # setup 2 of 3: the coarse levels of the blocks lag
    r = field_ranges(t.U)
    @test ns.diag[1].M ≈ A2[r[1], r[1]] && ns.diag[2].M ≈ -A2[r[2], r[2]]
    fill!(x, 0.0); solve!(x, ns, t.b)
    @test linear_converged(ns) && norm(x - xlu / 2) ≤ 1e-7 * norm(xlu)
    A3 = copy(A2)
    numerical_setup!(ns, A3)                                          # a new matrix object rebuilds everything
    @test all(preconditioner_builds(bns) == 2 for bns in ns.block_ns)
    @test preconditioner_builds(ns) == 3
    fill!(x, 0.0); solve!(x, ns, t.b)
    @test linear_converged(ns) && norm(x - xlu / 2) ≤ 1e-7 * norm(xlu)
  end

  @testset "an unusable block fails the solve at once" begin
    t = fs_saddle_hierarchy(2)
    ls = fs_hmg(t)
    A = copy(t.A)
    ns = numerical_setup(symbolic_setup(ls, A), A)
    x = zeros(length(t.b))
    A.nzval[1] = NaN
    numerical_setup!(ns, A)
    solve!(x, ns, t.b)
    @test !linear_converged(ns) && linear_iterations(ns) == 0
    @test occursin("block 1", HyperFEM.Solvers.linear_info(ns))
    @test !linear_indefinite(ns)                                      # a NaN block is not the instability signal
    copyto!(A.nzval, t.A.nzval)
    numerical_setup!(ns, A)
    solve!(x, ns, t.b)
    @test linear_converged(ns)
    # an indefinite u block (elasticity − s·mass) is reported through the field split (S5)
    r = field_ranges(t.U)
    Vu = t.U[1].space
    dΩ = Measure(Triangulation(fine_model(t.h)), 2)
    Mu = assemble_matrix((u, v) -> ∫(300.0 * (v ⋅ u))dΩ, TrialFESpace(Vu, x -> zero(VectorValue{3,Float64})), Vu)
    Bi = copy(t.A)
    Bi[r[1], r[1]] .-= Mu
    numerical_setup!(ns, Bi)
    solve!(x, ns, t.b)
    @test !linear_converged(ns) && linear_indefinite(ns)
    @test occursin("block 1", HyperFEM.Solvers.linear_info(ns))
  end

  @testset "Newton trajectory of LU on nonlinear electro-mechanics" begin
    p = fs_em_problem(3)
    for backend in S1_BACKENDS
      ref = s1_solve_nl(LUSolver(), p; backend)
      @test ref.flag
      for (name, ls) in (("additive", fs_hmg(p; split=:additive, rtol=1e-8)),
                         ("additive, blocks reuse 3", fs_hmg(p; split=:additive, rtol=1e-8, reuse=3)),
                         ("Schur", fs_hmg(p; split=:schur, rtol=1e-8)))
        @testset "$name ($backend)" begin
          r = s1_solve_nl(ls, p; backend)
          @test r.flag
          @test r.its == ref.its
          @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
        end
      end
      r = s1_solve_nl(fs_hmg(p; split=:additive, rtol=1e-8, reuse=3), p; backend, forcing=EisenstatWalker())
      @test r.flag
      @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
    end
  end
end
