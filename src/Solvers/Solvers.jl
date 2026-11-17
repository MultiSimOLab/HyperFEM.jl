module Solvers

using Gridap
using Gridap.Algebra
import Gridap.Algebra: numerical_setup, numerical_setup!
using Gridap.CellData
using Gridap.FESpaces: SingleFieldFESpace, get_fe_dof_basis, get_free_dof_values, interpolate, num_free_dofs
using Gridap.FESpaces: get_cell_dof_ids, TrialFESpace
using SparseArrays
import IterativeSolvers
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
export linear_indefinite
export preconditioner_builds
export set_tolerance!
export tolerance
export PETScSolver
export AMGSolver
export HMultigridSolver
export FieldSplitSolver
export near_nullspace
export field_ranges
export split_nullspace
export numerical_setup_state
export numerical_setup_state!
export FullStepLS
export BacktrackingLS
export InjectivityLS
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
export forcing_history
export EisenstatWalker
export print_summary
export reason_text
export linear_info
export linesearch_info
export injectivity_limit
export search!
export AbstractLineSearch

import HyperFEM.WeakForms: energy
using HyperFEM.PhysicalModels: TangentSwitch, set_projected!, is_projected

include("KrylovSolvers.jl")
include("HMultigrid.jl")
include("FieldSplit.jl")
include("Injectivity.jl")
include("ModelOperator.jl")
include("LineSearches.jl")
include("NonlinearSolvers.jl")

end
