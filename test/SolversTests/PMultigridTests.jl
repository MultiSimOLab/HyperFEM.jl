# p-multigrid inside the drivers (design/solvers-s4a-plan.md, S4-A): `level_spaces` builds the Q1 level of a Q2 working
# space with the same boundary conditions, and HyperFEMKernels' `PMultigrid` preconditions the matrix-free Newton
# (`CompiledBackend(jacobian = :matrixfree)`). The reference is LU on the assembled tangent: same Newton trajectory and
# solution, with the Dirichlet projection (an operator on the increment space) going through the same solver, and the
# levels following the operator from load step to load step. Needs HyperFEMKernels ≥ 0.2.1.
include("krylov_common.jl")
import AlgebraicMultigrid                      # HyperFEMAMGExt: AMGSolver as coarse solver
using HyperFEMKernels: MatrixFreeSolver, PMultigrid, Chebyshev

const PMG_BACKEND = CompiledBackend(jacobian=:matrixfree)

"The S1 mechanical bar of krylov_common.jl at order `p` (neo-Hookean, clamped and sheared, two load steps)."
function pmg_mech_problem(p=2)
  model = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5, 2, 2))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  physmodel = NeoHookean3D(λ=10.0, μ=1.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2p)
  D = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]), BoundaryCondition("top", [0.05, 0.0, 0.0], Λ -> Λ))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, p), D, conformity=:H1)
  U = TrialFESpace(V, D, 0.0)
  k = Kinematics(Mechano, Solid)
  res(Λ) = (u, v) -> HyperFEM.residual(physmodel, k, u, v, dΩ)
  jac(Λ) = (u, du, v) -> HyperFEM.jacobian(physmodel, k, u, du, v, dΩ)
  (; res, jac, U, V, D)
end

@testset "p-multigrid in the drivers (S4-A)" begin
  @testset "level_spaces" begin
    p = pmg_mech_problem(2)
    sp = level_spaces(p.U, p.V, p.D)
    @test length(sp) == 2 && sp[1] == (p.U, p.V)
    U1, V1 = sp[2]
    @test U1 isa TrialFESpace
    @test num_free_dofs(U1) < num_free_dofs(p.U)
    @test num_free_dofs(U1) == num_free_dofs(V1)
    sp3 = level_spaces(pmg_mech_problem(3).U, pmg_mech_problem(3).V, p.D, (2, 1))
    @test length(sp3) == 3
    pe = s1_em_problem()
    @test_throws ErrorException level_spaces(pe.U, pe.V, pe.D)
  end

  @testset "Newton trajectory of LU with PMultigrid (coarse LU and AMG), Dirichlet projection included" begin
    p = pmg_mech_problem(2)
    ref = s1_solve_nl(LUSolver(), p; backend=:compiled)
    @test ref.flag
    sp = level_spaces(p.U, p.V, p.D)
    N1 = near_nullspace(sp[2][1])
    for (name, krylov, coarse, every) in (("coarse LU", :gmres, LUSolver(), 1),
                                          ("coarse LU, every 3", :gmres, LUSolver(), 3),
                                          ("coarse AMG solve, FGMRES", :fgmres, AMGSolver(:cg; rtol=1e-8, nullspace=N1), 1))
      @testset "$name" begin
        ls = MatrixFreeSolver(; krylov, rtol=1e-10, maxiter=200, preconditioner=PMultigrid(sp; coarse, coarse_every=every))
        r = s1_solve_nl(ls, p; backend=PMG_BACKEND)
        @test r.flag
        @test r.its == ref.its
        @test maximum(abs.(log10.(r.hist) .- log10.(ref.hist))) < 0.5
        @test norm(r.x - ref.x) ≤ 1e-7 * norm(ref.x)
        @test all(≤(40), r.lin)                                  # Krylov iterations per Newton iteration
        st = r.ns.prec
        @test st.lv.nform ≥ 1                                    # the levels followed the operator across load steps
        @test st.lv.nskip ≥ 1                                    # the Dirichlet projection operator was recognised and skipped
      end
    end
  end
end
