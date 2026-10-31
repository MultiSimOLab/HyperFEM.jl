module HyperFEMAMGExt

# Pure-Julia Krylov + AMG for HyperFEM's Newton (design/solvers.md, S1 and S2): the methods of `AMGSolver` as a
# Gridap `LinearSolver` — IterativeSolvers' `cg!`/`gmres!` preconditioned by one V-cycle of
# AlgebraicMultigrid's smoothed aggregation, or by a block preconditioner with one hierarchy per field
# (block Jacobi, block Gauss–Seidel, upper Schur with S ≈ A₂₂). Activated by `import AlgebraicMultigrid`
# (imported, not used: the package exports its own `solve!`). Serial; the CI variant of `PETScSolver` and
# the reference of the Newton integration (`linear_converged`, lagged setups through `reuse`).

using HyperFEM
using HyperFEM.Solvers: AMGSolver, split_nullspace
import HyperFEM.Solvers: amg_symbolic_setup, linear_converged, linear_iterations, linear_info, preconditioner_builds
using Gridap.Algebra
import Gridap.Algebra: numerical_setup, numerical_setup!, solve!
using AlgebraicMultigrid: smoothed_aggregation, aspreconditioner
using IterativeSolvers: cg!, gmres!
using SparseArrays
using LinearAlgebra

struct AMGSolverSS <: SymbolicSetup
  solver::AMGSolver
end

amg_symbolic_setup(s::AMGSolver, ::AbstractMatrix) = AMGSolverSS(s)

# ------------------------------------------------------------------------------------------------
# The (possibly negated) operator the Krylov method sees

"The matrix the solver works on: `A` itself, or a copy with the rows in `negate` changed of sign."
function _operator(A::SparseMatrixCSC, negate)
  negate === nothing && return A
  B = copy(A)
  _negate_rows!(B, negate)
  B
end
_operator(A::AbstractMatrix, negate) = _operator(sparse(A), negate)

function _negate_rows!(B::SparseMatrixCSC, negate)
  rv, nz = B.rowval, B.nzval
  @inbounds for k in eachindex(nz)
    rv[k] in negate && (nz[k] = -nz[k])
  end
  B
end

"Refreshes the values of the operator copy from `A` (same pattern), negating the rows in `negate`."
function _update_operator!(B::SparseMatrixCSC, A::SparseMatrixCSC, negate)
  B === A && return B
  copyto!(B.nzval, A.nzval)
  _negate_rows!(B, negate)
end
_update_operator!(B::SparseMatrixCSC, A::AbstractMatrix, negate) = _update_operator!(B, sparse(A), negate)

function negate!(b::AbstractVector, negate)
  negate === nothing && return b
  @inbounds for i in negate
    b[i] = -b[i]
  end
  b
end

# ------------------------------------------------------------------------------------------------
# Preconditioners: one V-cycle, or a block preconditioner with one V-cycle per field

"""
Block preconditioner on the field ranges: `P[i]` one V-cycle on the diagonal block, `A[i, j]` the off-diagonal
blocks the triangular splits need (empty for the additive one and on the diagonal).
"""
struct BlockPreconditioner{TP,TA}
  fields::Vector{UnitRange{Int}}
  P::Vector{TP}
  A::Matrix{TA}
  split::Symbol
  r::Vector{Vector{Float64}}                 # work vectors per field: right-hand side of the block solve
  z::Vector{Vector{Float64}}                 # and its solution (AlgebraicMultigrid's V-cycle wants plain vectors)
  w::Vector{Float64}                         # copy of the input of the in-place application
end

function _block_preconditioner(s::AMGSolver, B::SparseMatrixCSC)
  f = s.fields
  N = s.nullspace === nothing ? nothing : split_nullspace(s.nullspace, f)
  P = [begin
    Aii = B[r, r]
    ml = N === nothing || size(N[i], 2) == 0 ? smoothed_aggregation(Aii) : smoothed_aggregation(Aii; B=N[i])
    aspreconditioner(ml)
  end for (i, r) in enumerate(f)]
  n = length(f)
  needed(i, j) = s.split === :multiplicative ? j < i : s.split === :schur ? (i, j) == (1, 2) : false
  blocks = [needed(i, j) ? B[f[i], f[j]] : spzeros(0, 0) for i in 1:n, j in 1:n]
  BlockPreconditioner(f, P, blocks, s.split, [zeros(length(r)) for r in f], [zeros(length(r)) for r in f], zeros(size(B, 1)))
end

# one V-cycle of block i on bp.r[i], result in bp.z[i] and in y
function _block_solve!(y, bp::BlockPreconditioner, i)
  ldiv!(bp.z[i], bp.P[i], bp.r[i])
  copyto!(view(y, bp.fields[i]), bp.z[i])
end

function LinearAlgebra.ldiv!(y::AbstractVector, bp::BlockPreconditioner, x::AbstractVector)
  f, A, r, z = bp.fields, bp.A, bp.r, bp.z
  n = length(f)
  for i in 1:n
    copyto!(r[i], view(x, f[i]))
  end
  if bp.split === :additive
    for i in 1:n
      _block_solve!(y, bp, i)
    end
  elseif bp.split === :multiplicative          # lower block triangular: y_i = P_i⁻¹ (x_i − Σ_{j<i} A_ij y_j)
    for i in 1:n
      for j in 1:i-1
        mul!(r[i], A[i, j], z[j], -1.0, 1.0)
      end
      _block_solve!(y, bp, i)
    end
  else                                         # :schur, upper, S ≈ A₂₂: y_2 = P_2⁻¹ x_2, y_1 = P_1⁻¹ (x_1 − A_12 y_2)
    _block_solve!(y, bp, 2)
    mul!(r[1], A[1, 2], z[2], -1.0, 1.0)
    _block_solve!(y, bp, 1)
  end
  y
end

# in place (the form IterativeSolvers' gmres! uses for Pr): through the work copy, no allocation per application
LinearAlgebra.ldiv!(bp::BlockPreconditioner, x::AbstractVector) = (copyto!(bp.w, x); ldiv!(x, bp, bp.w))

function _preconditioner(s::AMGSolver, B::SparseMatrixCSC)
  isempty(s.fields) || return _block_preconditioner(s, B)
  ml = s.nullspace === nothing ? smoothed_aggregation(B) : smoothed_aggregation(B; B=s.nullspace)
  aspreconditioner(ml)
end

# ------------------------------------------------------------------------------------------------
# Setups and solve

mutable struct AMGSolverNS{TA<:AbstractMatrix,TB<:SparseMatrixCSC} <: NumericalSetup
  const solver::AMGSolver
  A::TA                                      # the matrix of the last setup (the Newton cache's)
  B::TB                                      # the operator (A or its negated copy)
  P::Any
  nsetup::Int
  nbuild::Int
  iters::Int
  rnorm::Float64
  converged::Bool
end

function numerical_setup(ss::AMGSolverSS, A::AbstractMatrix)
  s = ss.solver
  s.nullspace === nothing || size(s.nullspace, 1) == size(A, 1) ||
    error("HyperFEM: AMGSolver: the near-null space has $(size(s.nullspace, 1)) rows but the matrix has $(size(A, 1))")
  isempty(s.fields) || last(s.fields[end]) == size(A, 1) ||
    error("HyperFEM: AMGSolver: the field ranges end at $(last(s.fields[end])) but the matrix has $(size(A, 1)) rows")
  B = _operator(A, s.negate)
  P = _preconditioner(s, B)
  AMGSolverNS{typeof(A),typeof(B)}(s, A, B, P, 1, 1, 0, 0.0, true)
end

function numerical_setup!(ns::AMGSolverNS, A::AbstractMatrix)
  s = ns.solver
  ns.nsetup += 1
  rebuild = A !== ns.A || (ns.nsetup - 1) % s.reuse == 0   # a new matrix object always rebuilds (as the PETSc solver)
  if A !== ns.A
    ns.A = A
    ns.B = _operator(A, s.negate)
  else
    _update_operator!(ns.B, A, s.negate)
  end
  if rebuild
    ns.P = _preconditioner(s, ns.B)        # the hierarchies of the lagged setups keep the previous tangent
    ns.nbuild += 1
  end
  ns
end

function solve!(x::AbstractVector, ns::AMGSolverNS, b::AbstractVector)
  s = ns.solver
  fill!(x, zero(eltype(x)))
  negate!(b, s.negate)
  if s.krylov === :cg
    _, h = cg!(x, ns.B, b; Pl=ns.P, reltol=s.rtol, abstol=s.atol, maxiter=s.maxiter, initially_zero=true, log=true)
  else
    # right preconditioning: the residual the method monitors is the true one, ‖b - A x‖ ≤ rtol ‖b‖, as the PETSc solver
    _, h = gmres!(x, ns.B, b; Pr=ns.P, reltol=s.rtol, abstol=s.atol, maxiter=s.maxiter, restart=min(s.restart, length(b)), initially_zero=true, log=true)
  end
  negate!(b, s.negate)
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
