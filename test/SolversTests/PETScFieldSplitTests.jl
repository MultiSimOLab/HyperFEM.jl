# `PETScSolver(U; …)` (design/solvers-s2-plan.md, S2): the S2 battery on PETSc's PCFIELDSPLIT with GAMG on every
# block. Local, as PETScSolverTests.jl (HYPERFEM_TEST_PETSC=1 or an environment with GridapPETSc).
using GridapPETSc
isdefined(@__MODULE__, :s1_battery) || include("krylov_common.jl")   # already included by the S1 tests of the same session
include("fieldsplit_common.jl")

s2_battery("PETScSolver fieldsplit", (U; kwargs...) -> PETScSolver(U; kwargs...))

@testset "PETScSolver fieldsplit" begin
  U, V, A, b = s2_saddle_problem(2)
  xlu = A \ b
  r = field_ranges(U)

  @testset "Schur variants and block solvers" begin
    for (fact, prec) in ((:lower, :a11), (:full, :a11), (:upper, :selfp))
      x, ns = s1_solve(PETScSolver(U; schur=(fact=fact, precondition=prec), rtol=1e-10), A, b)
      @test linear_converged(ns)
      @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
    end
    x, ns = s1_solve(PETScSolver(U; blocks=[:mumps, :gamg], rtol=1e-10), A, b)   # exact u block
    @test linear_converged(ns)
    @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
    x, ns = s1_solve(PETScSolver(U; ksp=:fgmres, rtol=1e-10, options="-fieldsplit_u_ksp_type cg -fieldsplit_u_ksp_rtol 1e-2"), A, b)
    @test linear_converged(ns)                                                   # inexact inner solve, FGMRES outside
    @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
  end

  @testset "names, errors" begin
    @test HyperFEM.Solvers._linear_name(PETScSolver(U)) == "GMRES+FS-schur(GAMG,GAMG)"
    @test PETScSolver(U).fieldnames == ["u", "phi"]
    @test_throws ErrorException PETScSolver(:gmres, :fieldsplit)                                 # no fields
    @test_throws ErrorException PETScSolver(:gmres, :fieldsplit; fields=r, blocks=[:gamg])      # lengths
    @test_throws ErrorException PETScSolver(:gmres, :gamg; fields=r)                            # fields without fieldsplit
    @test_throws ErrorException PETScSolver(:gmres, :fieldsplit; fields=[r[1], r[2], r[2]], split=:schur)
  end
end
