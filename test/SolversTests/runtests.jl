using HyperFEM
using Gridap
using Test

@testset "Solvers" begin

  @time begin
    include("NewtonRaphsonTests.jl")
  end

end
