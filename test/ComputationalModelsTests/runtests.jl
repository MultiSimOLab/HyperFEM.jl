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

  @time begin
    include("TimeSteppingTests.jl")
  end

  @time begin
    include("TimeSchemesTests.jl")
  end

  @time begin
    include("TransientModelTests.jl")
  end

  @time begin
    include("LinearModelTests.jl")
  end

end
