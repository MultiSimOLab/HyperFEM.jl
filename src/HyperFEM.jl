module HyperFEM

include("GridapPerformancePatches.jl")   # temporary: performance of Gridap 0.19.9 on Gridap 0.20
include("TensorAlgebra/TensorAlgebra.jl")
include("Backends.jl")
include("PhysicalModels/PhysicalModels.jl")
include("WeakForms/WeakForms.jl")
include("Solvers/Solvers.jl")
include("DiscreteModeling/DiscreteModeling.jl")
include("ComputationalModels/ComputationalModels.jl")
include("Io.jl")
include("Exports.jl")

end
