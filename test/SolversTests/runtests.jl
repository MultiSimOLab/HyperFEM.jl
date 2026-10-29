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

  @time begin
    include("LineSearchTests.jl")
  end

  @time begin
    include("EnergyTests.jl")
  end

  @time begin
    include("HybridTests.jl")
  end

  @time begin
    include("VerboseTests.jl")
  end

end
