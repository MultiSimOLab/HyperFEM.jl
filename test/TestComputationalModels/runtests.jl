using HyperFEM
using Gridap
using Test

@testset "ComputationalModels" begin

  @time begin
    include("PostProcessorsTests.jl")
  end

end
