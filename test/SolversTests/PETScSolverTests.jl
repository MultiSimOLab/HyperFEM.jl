# `PETScSolver` (design/solvers.md, S1): the S1 battery on PETSc's KSP (GAMG, BoomerAMG) plus MUMPS as the
# direct reference. Needs GridapPETSc, which is not in the CI test environment (installing PETSc there is
# slow, D1): runs locally with HYPERFEM_TEST_PETSC=1 from an environment that has GridapPETSc, e.g.
#   julia --project=HyperFEMKernels.jl/benchmark/solvers HyperFEM.jl/test/SolversTests/PETScSolverTests.jl
using GridapPETSc              # activates HyperFEMPETScExt
include("krylov_common.jl")

s1_battery("PETScSolver CG/GMRES + GAMG", (krylov; rtol=1e-6, maxiter=500, reuse=1, nullspace=nothing) ->
  PETScSolver(krylov, :gamg; rtol, maxiter, reuse, nullspace))

@testset "PETScSolver" begin
  U, V, A, b = s1_elasticity(2)
  N = near_nullspace(U)
  xlu = A \ b

  @testset "MUMPS: the direct reference" begin
    for pc in (:mumps, :mumps_cholesky, :lu)
      x, ns = s1_solve(PETScSolver(:preonly, pc), A, b)
      @test linear_converged(ns)
      @test linear_iterations(ns) == 1
      @test norm(x - xlu) ≤ 1e-10 * norm(xlu)
      @test linear_info(ns) == ""
    end
  end

  @testset "BoomerAMG" begin
    for krylov in (:cg, :gmres)
      x, ns = s1_solve(PETScSolver(krylov, :boomeramg; rtol=1e-12, nullspace=N), A, b)
      @test linear_converged(ns)
      @test norm(x - xlu) ≤ 1e-8 * norm(xlu)
    end
  end

  @testset "options and names" begin
    x, ns = s1_solve(PETScSolver(:gmres, :gamg; rtol=1e-10, nullspace=N, options="-ksp_gmres_restart 50 -pc_gamg_threshold 0.02"), A, b)
    @test linear_converged(ns)
    @test norm(x - xlu) ≤ 1e-8 * norm(xlu)
    @test HyperFEM.Solvers._linear_name(PETScSolver(:cg, :gamg)) == "CG+GAMG"
    @test HyperFEM.Solvers._linear_name(PETScSolver(:preonly, :mumps)) == "MUMPS"
    @test occursin("CONVERGED", Base.get_extension(HyperFEM, :HyperFEMPETScExt).converged_reason(ns))
    @test_throws ErrorException PETScSolver(:cg, :ilu)
    @test_throws ErrorException s1_solve(PETScSolver(:cg, :gamg; options="bad"), A, b)
  end

  @testset "several solvers coexist (own option prefixes)" begin
    _, ns1 = s1_solve(PETScSolver(:cg, :gamg; rtol=1e-10, nullspace=N), A, b)
    _, ns2 = s1_solve(PETScSolver(:gmres, :boomeramg; rtol=1e-10, nullspace=N), A, b)
    x = zeros(length(b)); solve!(x, ns1, b)
    @test norm(x - xlu) ≤ 1e-8 * norm(xlu)
    @test ns1.prefix != ns2.prefix
  end
end
