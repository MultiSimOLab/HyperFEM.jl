using Gridap
using HyperFEM
using Test

@testset "HyperFEMTests" verbose = true begin

  include("ConstitutiveModelsTests/runtests.jl")

  include("TensorAlgebraTests/runtests.jl")

  include("WeakFormsTests/runtests.jl")

  include("ComputationalModelsTests/runtests.jl")

  include("SolversTests/runtests.jl")

  include("SimulationsTests/runtests.jl")

end;
