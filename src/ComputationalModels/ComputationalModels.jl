module ComputationalModels
using HyperFEM.Backends
using HyperFEM.PhysicalModels
using HyperFEM.Solvers

using Gridap
using Gridap.Helpers
using Gridap.TensorValues, Gridap.FESpaces, Gridap.Algebra
using Gridap: createvtk
using Gridap.MultiField
using Gridap.FESpaces: get_assembly_strategy
import Gridap: solve!

using BlockArrays

using LinearAlgebra
using Printf
using WriteVTK

using GridapGmsh 
using GridapGmsh: GmshDiscreteModel

# Deprecation exports
using HyperFEM.DiscreteModeling
Base.@deprecate_binding CartesianTags DiscreteModeling.CartesianTags
Base.@deprecate_binding EvolutionFunctions DiscreteModeling.EvolutionFunctions

include("BoundaryConditions.jl")
export Prescription
export Separable
export NonSeparable
export Coupling
export BoundaryCondition
export BoundaryConditions
export MultiFieldBoundaryConditions
export get_tags
export get_masks
export get_space_functions
export get_time_derivative
export central_difference
export get_Neumann_dΓ
export NothingBC
export NothingTC
export MultiFieldBC
export SingleFieldTC
export MultiFieldTC
export residual_Neumann
export NeumannConditions
export add_tag_from_vertex_filter!


include("DirichletApplicator.jl")
export dirichlet_applicator
export DirichletApplicator
include("FESpaces.jl")
export TrialFESpace
export TrialFESpace!
export TestFESpace
export TestFESpace!

include("GridapExtras.jl")
export GmshDiscreteModel
export evaluate!

include("Drivers.jl")
include("TimeStepping.jl")
include("LoadStepping.jl")
export TimeStepping
export AdaptiveStepping
export TimeStep
export LoadStepping
export LoadStep
export step!
export StaggeredModel
export StaticNonlinearModel
export DynamicNonlinearModel
export StaticLinearModel
export NonlinearStepCache
export LinearSystemCache
export solve!
export dirichlet_preconditioning!
export get_state
export get_measure
export get_spaces
export get_applicators
export get_assemblers
export get_trial_space
export get_test_space
export update_velocity!
export update_displacements!

include("PostProcessors.jl")
export PostProcessor
export vtk_save
export get_pvd
export Cauchy
export Piola
export Jacobian
export Entropy
export D0
export reset!
export interpolate_L2_tensor
export interpolate_L2_vector
export interpolate_L2_scalar
export interpolate_L2_field
export L2_projection
export L2_Projection
export component_LInf

end
