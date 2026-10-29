module Solvers

using IterativeSolvers
using Gridap
using Gridap.Algebra
using Gridap.CellData
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
export Injectivity_Preserving_LS
export Roman_LS
export update_cellstate!
export search!
export AbstractLineSearch

include("LinearSolvers.jl")
include("LineSearches.jl")
include("NonlinearSolvers.jl")

# Deprecated names (one release)
Base.@deprecate_binding Newton_RaphsonSolver NewtonRaphson
Base.@deprecate_binding Newton_RaphsonCache NewtonCache

end
