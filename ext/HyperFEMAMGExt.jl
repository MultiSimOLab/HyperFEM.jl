module HyperFEMAMGExt

# Pure-Julia Krylov + AMG for HyperFEM's Newton (design/solvers.md, S1): the methods of `AMGSolver` as a
# Gridap `LinearSolver` — IterativeSolvers' `cg!`/`gmres!` preconditioned by one V-cycle of
# AlgebraicMultigrid's smoothed aggregation built with the near-null space. Activated by
# `import AlgebraicMultigrid` (imported, not used: the package exports its own `solve!`). Serial; the CI variant of `PETScSolver` and the reference of the
# Newton integration (`linear_converged`, lagged setups through `reuse`).

using HyperFEM
using HyperFEM.Solvers: AMGSolver
import HyperFEM.Solvers: amg_symbolic_setup, linear_converged, linear_iterations, linear_info, preconditioner_builds
using Gridap.Algebra
import Gridap.Algebra: numerical_setup, numerical_setup!, solve!
using AlgebraicMultigrid: smoothed_aggregation, aspreconditioner, MultiLevel
using IterativeSolvers: cg!, gmres!
using SparseArrays
using LinearAlgebra

struct AMGSolverSS <: SymbolicSetup
  solver::AMGSolver
end

amg_symbolic_setup(s::AMGSolver, ::AbstractMatrix) = AMGSolverSS(s)

mutable struct AMGSolverNS{TA<:AbstractMatrix} <: NumericalSetup
  const solver::AMGSolver
  A::TA
  ml::MultiLevel
  P::Any                                     # AlgebraicMultigrid.Preconditioner (one V-cycle)
  nsetup::Int
  nbuild::Int
  iters::Int
  rnorm::Float64
  converged::Bool
end

function _hierarchy(s::AMGSolver, A::SparseMatrixCSC)
  ml = s.nullspace === nothing ? smoothed_aggregation(A) : smoothed_aggregation(A; B=s.nullspace)
  ml, aspreconditioner(ml)
end
_hierarchy(s::AMGSolver, A::AbstractMatrix) = _hierarchy(s, sparse(A))

function numerical_setup(ss::AMGSolverSS, A::AbstractMatrix)
  s = ss.solver
  s.nullspace === nothing || size(s.nullspace, 1) == size(A, 1) ||
    error("HyperFEM: AMGSolver: the near-null space has $(size(s.nullspace, 1)) rows but the matrix has $(size(A, 1))")
  ml, P = _hierarchy(s, A)
  AMGSolverNS{typeof(A)}(s, A, ml, P, 1, 1, 0, 0.0, true)
end

function numerical_setup!(ns::AMGSolverNS, A::AbstractMatrix)
  ns.nsetup += 1
  rebuild = A !== ns.A || (ns.nsetup - 1) % ns.solver.reuse == 0   # a new matrix object always rebuilds (as the PETSc solver)
  ns.A = A
  if rebuild
    ns.ml, ns.P = _hierarchy(ns.solver, A)   # the hierarchy of the lagged setups keeps the previous tangent
    ns.nbuild += 1
  end
  ns
end

function solve!(x::AbstractVector, ns::AMGSolverNS, b::AbstractVector)
  s = ns.solver
  fill!(x, zero(eltype(x)))
  if s.krylov === :cg
    _, h = cg!(x, ns.A, b; Pl=ns.P, reltol=s.rtol, abstol=s.atol, maxiter=s.maxiter, initially_zero=true, log=true)
  else
    # right preconditioning: the residual the method monitors is the true one, ‖b - A x‖ ≤ rtol ‖b‖, as the PETSc solver
    _, h = gmres!(x, ns.A, b; Pr=ns.P, reltol=s.rtol, abstol=s.atol, maxiter=s.maxiter, restart=min(s.restart, length(b)), initially_zero=true, log=true)
  end
  ns.iters = h.iters
  ns.converged = h.isconverged
  res = h[:resnorm]
  ns.rnorm = isempty(res) ? 0.0 : res[end]
  x
end

linear_converged(ns::AMGSolverNS) = ns.converged
linear_iterations(ns::AMGSolverNS) = ns.iters
preconditioner_builds(ns::AMGSolverNS) = ns.nbuild
linear_info(ns::AMGSolverNS) = " $(ns.iters)"

end # module
