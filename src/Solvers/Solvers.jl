module Solvers

using IterativeSolvers
using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.Arrays: array_cache, getindex!
using LinearAlgebra
using Printf
using AbstractTrees

export NewtonRaphson
export NewtonTolerances
export NewtonLog
export NewtonCache
export Newton_RaphsonSolver
export Newton_RaphsonCache
export solve!
export converged
export residual_history
export step_history
export num_iterations
export stop_reason
export verbosity
export get_tolerances
export linear_converged
export numerical_setup_state
export numerical_setup_state!
export FullStepLS
export BacktrackingLS
export InjectivityLS
export LineSearch
export Roman_LS
export Injectivity_Preserving_LS
export update_cellstate!
export injectivity_step
export ModelOperator
export backend_operator
export energy
export max_step
export set_tangent!
export TangentStrategy
export ConsistentTangent
export HybridTangent
export mode_history
export search!
export AbstractLineSearch

import HyperFEM.WeakForms: energy
using HyperFEM.PhysicalModels: TangentSwitch, set_projected!, is_projected

include("LinearSolvers.jl")
include("Injectivity.jl")
include("ModelOperator.jl")
include("LineSearches.jl")
include("NonlinearSolvers.jl")

# Deprecated names (one release)
Base.@deprecate_binding Newton_RaphsonSolver NewtonRaphson
Base.@deprecate_binding Newton_RaphsonCache NewtonCache

end
