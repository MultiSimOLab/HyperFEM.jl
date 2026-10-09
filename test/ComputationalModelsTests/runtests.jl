using HyperFEM
using Gridap
using Test

@testset "ComputationalModels" begin

  @time begin
    include("BoundaryConditionsTests.jl")
  end

end
