using HyperFEM
using Gridap
using Test

@testset "Solvers" begin

  @time begin
    include("NewtonRaphsonTests.jl")
  end

  @time begin
    include("AllocationTests.jl")
  end

end
