using HyperFEM
using Gridap
using Test

@testset "ComputationalModels" begin

  @time begin
    include("PostProcessorsTests.jl")
  end

  @time begin
    include("BoundaryConditionsTests.jl")
  end

  @time begin
    include("DriversTests.jl")
  end

  @time begin
    include("BackendsTests.jl")
  end

  @time begin
    include("LoadSteppingTests.jl")
  end

end
