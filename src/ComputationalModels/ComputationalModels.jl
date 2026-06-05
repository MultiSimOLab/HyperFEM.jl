module ComputationalModels
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
using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.BlockSolvers
using GridapSolvers.SolverInterfaces: SolverVerboseLevel, SOLVER_VERBOSE_NONE, SOLVER_VERBOSE_LOW, SOLVER_VERBOSE_HIGH
using GridapSolvers.SolverInterfaces: finished, print_message, converged

using LinearAlgebra
using WriteVTK

using GridapGmsh 
using GridapGmsh: GmshDiscreteModel

import Base.getindex

# Deprecation exports
using HyperFEM.DiscreteModeling
Base.@deprecate_binding CartesianTags DiscreteModeling.CartesianTags
Base.@deprecate_binding EvolutionFunctions DiscreteModeling.EvolutionFunctions

include("BoundaryConditions.jl")
export DirichletBC
export NeumannBC
export get_Neumann_dΓ
export NothingBC
export NothingTC
export MultiFieldBC
export SingleFieldTC
export MultiFieldTC
export residual_Neumann
export updateBC!
export DirichletCoupling
export InterpolableBC
export InterpolableBC!
export add_tag_from_vertex_filter!


include("FESpaces.jl")
export TrialFESpace
export TrialFESpace!
export TestFESpace
export TestFESpace!

include("GridapExtras.jl")
export GmshDiscreteModel
export evaluate!

include("Drivers.jl")
export StaggeredModel
export StaticNonlinearModel
export DynamicNonlinearModel
export StaticLinearModel
export solve!
export dirichlet_preconditioning!
export get_state
export get_measure
export get_spaces
export get_assemblers
export get_trial_space
export get_test_space

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
export L2_Projection
end
