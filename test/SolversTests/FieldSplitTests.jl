# Block preconditioners by fields (design/solvers-s2-plan.md, S2): the pure-Julia `AMGSolver(U; …)` with one
# smoothed-aggregation hierarchy per field, which runs in CI. `PETScSolver(U; …)` runs the same battery in
# PETScFieldSplitTests.jl (local, needs GridapPETSc). krylov_common.jl must be included first (KrylovSolversTests.jl does).
import AlgebraicMultigrid
isdefined(@__MODULE__, :s1_battery) || include("krylov_common.jl")   # already included by the S1 tests of the same session
include("fieldsplit_common.jl")

s2_battery("AMGSolver blocks", (U; kwargs...) -> AMGSolver(U; kwargs...); growth=2.0)

@testset "AMGSolver block errors" begin
  U, V, A, b = s2_saddle_problem(1)
  r = field_ranges(U)
  @test_throws ErrorException AMGSolver(:cg; fields=r)                       # blocks need GMRES
  @test_throws ErrorException AMGSolver(:gmres; fields=[r[1]])               # one field is not a split
  @test_throws ErrorException AMGSolver(:gmres; fields=r, split=:lower)
  @test_throws ErrorException AMGSolver(:gmres; fields=[1:3, 5:7])           # not consecutive
  @test_throws ErrorException AMGSolver(:gmres; fields=[2:4, 5:7])           # not from 1
  @test HyperFEM.Solvers._linear_name(AMGSolver(U)) == "GMRES+FS-schur(AMG)"
end
