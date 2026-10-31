module Solvers

using IterativeSolvers
using Gridap
using Gridap.Algebra
import Gridap.Algebra: numerical_setup, numerical_setup!
using Gridap.CellData
using Gridap.FESpaces: SingleFieldFESpace, get_fe_dof_basis, get_free_dof_values, interpolate, num_free_dofs
using Gridap.MultiField: MultiFieldFESpace, num_fields
using Gridap.MultiField: MultiFieldStyle, ConsecutiveMultiFieldStyle
using Gridap.Geometry: num_point_dims, get_triangulation
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
export linear_iterations
export preconditioner_builds
export PETScSolver
export AMGSolver
export near_nullspace
export field_ranges
export split_nullspace
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
export linear_history
export print_summary
export reason_text
export linear_info
export linesearch_info
export injectivity_limit
export search!
export AbstractLineSearch

import HyperFEM.WeakForms: energy
using HyperFEM.PhysicalModels: TangentSwitch, set_projected!, is_projected

include("LinearSolvers.jl")
include("KrylovSolvers.jl")
include("Injectivity.jl")
include("ModelOperator.jl")
include("LineSearches.jl")
include("NonlinearSolvers.jl")

# Deprecated names (one release)
Base.@deprecate_binding Newton_RaphsonSolver NewtonRaphson
Base.@deprecate_binding Newton_RaphsonCache NewtonCache

end
