# IterativeSolver (deprecated): still works, warns when deprecation warnings are on.
using Gridap, Gridap.Algebra
using HyperFEM
using LinearAlgebra
using Test

@testset "IterativeSolver (deprecated)" begin
  n = 50
  A = diagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
  b = ones(n)
  ls = Base.JLOptions().depwarn == 0 ? IterativeSolver("cg"; reltol=1e-12) :
       (@test_logs (:warn, r"deprecated") match_mode = :any IterativeSolver("cg"; reltol=1e-12))
  ns = numerical_setup(symbolic_setup(ls, A), A)
  x = zeros(n)
  solve!(x, ns, b)
  @test norm(A * x - b) < 1e-10 * norm(b)
end

@testset "set_tolerance! is a no-op for direct solvers" begin
  A = [4.0 1.0; 1.0 3.0]; b = [1.0, 2.0]
  ns = numerical_setup(symbolic_setup(LUSolver(), A), A)
  @test set_tolerance!(ns, 1e-2) === ns
  x = zeros(2); solve!(x, ns, b)
  @test x ≈ A \ b
end
