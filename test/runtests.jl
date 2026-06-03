using Gridap
using HyperFEM
using Test

@testset "HyperFEMTests" verbose = true begin

  include("TestConstitutiveModels/runtests.jl")

  include("TestTensorAlgebra/runtests.jl")

  include("TestWeakForms/runtests.jl")

  include("SimulationsTests/runtests.jl")

end;
