using HyperFEM
using Gridap
using Test

@testset "TensorAlgebra" begin

  @time begin
    include("TensorAlgebraTests.jl")
    include("Hessian.jl")
  end

end
