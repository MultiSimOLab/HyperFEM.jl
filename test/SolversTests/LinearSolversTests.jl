# Hooks of the linear solvers used by the Newton.
using Gridap, Gridap.Algebra
using HyperFEM
using LinearAlgebra
using Test

@testset "set_tolerance! is a no-op for direct solvers" begin
  A = [4.0 1.0; 1.0 3.0]; b = [1.0, 2.0]
  ns = numerical_setup(symbolic_setup(LUSolver(), A), A)
  @test set_tolerance!(ns, 1e-2) === ns
  x = zeros(2); solve!(x, ns, b)
  @test x ≈ A \ b
end
