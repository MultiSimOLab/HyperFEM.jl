# Krylov + AMG on the assembled tangent (design/solvers.md, S1): the near-null space of the tangent and the
# pure-Julia `AMGSolver` (AlgebraicMultigrid + IterativeSolvers), which runs in CI. `PETScSolver` runs the
# same battery in PETScSolverTests.jl (local, needs GridapPETSc).
import AlgebraicMultigrid      # activates HyperFEMAMGExt (imported, not used: it exports its own solve!)
include("krylov_common.jl")

@testset "near_nullspace" begin
  @testset "free body: rigid modes in the kernel" begin
    U, V, A, b = s1_elasticity(2; dirichlet=false)
    N = near_nullspace(U)
    @test size(N) == (num_free_dofs(U), 6)
    @test norm(N' * N - I) < 1e-12
    @test norm(A * N) < 1e-10 * norm(A, Inf)
  end
  @testset "Dirichlet dofs left out, modes restricted" begin
    U, V, A, b = s1_elasticity(2)
    N = near_nullspace(U)
    @test size(N) == (num_free_dofs(U), 6)
    @test norm(N' * N - I) < 1e-12
    @test norm(A * N) > 0                          # the restricted modes are not in the kernel any more
  end
  @testset "2D: 3 modes" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    V = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1))
    N = near_nullspace(TrialFESpace(V))
    @test size(N, 2) == 3
    @test norm(N' * N - I) < 1e-12
  end
  @testset "multi-field: one block per field" begin
    model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 3, 3))
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags=["boundary"])
    Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1))
    U = MultiFieldFESpace([TrialFESpace(Vu, x -> zero(VectorValue{3,Float64})), TrialFESpace(Vφ)])
    N = near_nullspace(U)
    nu = num_free_dofs(U[1])
    @test size(N) == (num_free_dofs(U), 7)
    @test norm(N' * N - I) < 1e-12
    # the rigid modes live on the u block and the constant mode on the φ block (to the round-off of the QR:
    # exact zeros with Accelerate, 1e-18 with OpenBLAS)
    @test maximum(abs, N[nu+1:end, 1:6]) < 1e-14
    @test maximum(abs, N[1:nu, 7]) < 1e-14
    @test all(≈(N[nu+1, 7]), N[nu+1:end, 7])
  end
  @testset "scalar field: the constant" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2))
    N = near_nullspace(TrialFESpace(V))
    @test size(N, 2) == 1
    @test all(≈(N[1]), N)
  end
end

# AlgebraicMultigrid's smoothed aggregation with default parameters is less mesh-independent than GAMG on the
# thin bar: a growth of 2 is admitted here, the plan's ±30 % is tested on PETSc
s1_battery("AMGSolver", (krylov; rtol=1e-6, maxiter=500, reuse=1, nullspace=nothing) ->
  AMGSolver(krylov; rtol, maxiter, reuse, nullspace); growth=2.0)

@testset "AMGSolver errors" begin
  @test_throws ErrorException AMGSolver(:bicg)
  @test_throws ErrorException AMGSolver(:cg; reuse=0)
  U, V, A, b = s1_elasticity(1)
  N = near_nullspace(U)
  @test_throws ErrorException s1_solve(AMGSolver(:cg; nullspace=N[1:end-1, :]), A, b)
  @test HyperFEM.Solvers._linear_name(AMGSolver(:gmres)) == "GMRES+AMG"
end
