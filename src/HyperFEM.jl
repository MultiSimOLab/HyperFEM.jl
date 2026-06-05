module HyperFEM

include("TensorAlgebra/TensorAlgebra.jl")
include("PhysicalModels/PhysicalModels.jl")
include("WeakForms/WeakForms.jl")
include("Solvers/Solvers.jl")
include("DiscreteModeling/DiscreteModeling.jl")
include("ComputationalModels/ComputationalModels.jl")
include("Io.jl")
include("Exports.jl")

end
