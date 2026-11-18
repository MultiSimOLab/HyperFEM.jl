# Block (field-split) preconditioner in pure Julia with injected block solvers (design/solvers-s4b-plan.md §3.5, S4-B
# step 5): GMRES on the positive-real form of the saddle-point tangent (the φ rows negated, as S2) preconditioned by one
# application of a Gridap `LinearSolver` per diagonal block — one V-cycle of `HMultigridSolver`, one GAMG cycle of
# `PETScSolver(:preonly, :gamg)`, or a direct solver. The blocks are sliced from the tangent in a fixed pattern and
# refilled without allocation, so the block solvers see one matrix object per block and their own `reuse` applies.

"""
    FieldSplitSolver(fields, blocks; krylov = :gmres, split = :additive, negate = nothing, symmetric = true,
                     rtol = 1e-6, atol = 0.0, maxiter = 500, restart = 30, reuse = 1)
    FieldSplitSolver(U::MultiFieldFESpace, blocks; split = :additive, negate = :last, kwargs...)

Krylov method on a multi-field tangent preconditioned by blocks: `fields` are the dof ranges of the fields
([`field_ranges`](@ref)) and `blocks[i]` a Gridap `LinearSolver` whose **single application** preconditions the
diagonal block `i` — an [`HMultigridSolver`](@ref) (one V-cycle on the block, its own `reuse`), a direct solver
(exact block), or a `PETScSolver(:preonly, :gamg)` (one GAMG cycle). `split = :additive` (block Jacobi),
`:multiplicative` (block Gauss–Seidel, lower triangular) or `:schur` (two fields, upper block triangular with the
Schur complement approximated by its diagonal block). `negate` is the range of equations to negate (the φ rows of a
saddle point: `:last` in the constructor from the space), which gives the positive-real form `[A B; −Bᵀ C]` with
SPD diagonal blocks — what the block V-cycles need; the right-hand side is negated and restored. With
`symmetric = true` the tangent itself is symmetric (the saddle point before the negation) and its products are
threaded. The diagonal blocks are refilled at every setup and the block solvers set up every `reuse` setups (a new
matrix object always rebuilds); an `HMultigridSolver` block should carry the lag itself (`reuse = 1` here, its own
`reuse = 3`), so that its fine smoother follows the current tangent.

Newton hooks: [`linear_converged`](@ref), [`linear_iterations`](@ref), [`preconditioner_builds`](@ref),
[`set_tolerance!`](@ref)/[`tolerance`](@ref). A block whose solver is not usable (an indefinite block for the
multigrid) fails the solve at once, as `HMultigridSolver` does. Block preconditioners are not symmetric: `:gmres`.
"""
struct FieldSplitSolver{B<:Tuple} <: Algebra.LinearSolver
  fields::Vector{UnitRange{Int}}
  blocks::B
  krylov::Symbol
  split::Symbol
  negate::Union{Nothing,UnitRange{Int}}
  symmetric::Bool
  rtol::Float64
  atol::Float64
  maxiter::Int
  restart::Int
  reuse::Int
  function FieldSplitSolver(fields::AbstractVector{<:AbstractUnitRange}, blocks; krylov::Symbol=:gmres, split::Symbol=:additive,
                            negate::Union{Nothing,AbstractUnitRange}=nothing, symmetric::Bool=true, rtol::Real=1e-6,
                            atol::Real=0.0, maxiter::Integer=500, restart::Integer=30, reuse::Integer=1)
    fr = [UnitRange{Int}(r) for r in fields]
    bl = Tuple(blocks)
    length(fr) ≥ 2 || error("HyperFEM: FieldSplitSolver needs at least two fields")
    length(bl) == length(fr) || error("HyperFEM: FieldSplitSolver: $(length(fr)) fields need $(length(fr)) block solvers (got $(length(bl)))")
    all(b -> b isa Algebra.LinearSolver, bl) || error("HyperFEM: FieldSplitSolver: the blocks must be Gridap LinearSolvers")
    krylov === :gmres || error("HyperFEM: FieldSplitSolver: block preconditioners are not symmetric: use krylov = :gmres")
    split in (:additive, :multiplicative, :schur) || error("HyperFEM: FieldSplitSolver: split must be :additive, :multiplicative or :schur")
    split === :schur && length(fr) != 2 && error("HyperFEM: FieldSplitSolver: the Schur split needs exactly two fields")
    first(fr[1]) == 1 || error("HyperFEM: FieldSplitSolver: the field ranges must start at 1 (got $(first(fr[1])))")
    for i in 2:length(fr)
      first(fr[i]) == last(fr[i-1]) + 1 || error("HyperFEM: FieldSplitSolver: the field ranges must be consecutive")
    end
    neg = negate === nothing ? nothing : UnitRange{Int}(negate)
    neg === nothing || neg in fr || error("HyperFEM: FieldSplitSolver: negate must be one of the field ranges")
    reuse ≥ 1 || error("HyperFEM: FieldSplitSolver: reuse must be ≥ 1")
    for (i, b) in enumerate(bl)
      b isa HMultigridSolver && num_free_dofs(b.spaces[end]) != length(fr[i]) &&
        error("HyperFEM: FieldSplitSolver: the multigrid of block $i has $(num_free_dofs(b.spaces[end])) fine dofs but the field has $(length(fr[i]))")
    end
    new{typeof(bl)}(fr, bl, krylov, split, neg, symmetric, rtol, atol, maxiter, restart, reuse)
  end
end

function FieldSplitSolver(U::MultiFieldFESpace, blocks; split::Symbol=:additive, negate=:last, kwargs...)
  fields = field_ranges(U)
  neg = negate === :last ? fields[end] : negate
  FieldSplitSolver(fields, blocks; split, negate=neg, kwargs...)
end

_block_name(s::HMultigridSolver) = "hMG($(length(s.spaces)))"
_block_name(s::Algebra.LinearSolver) = string(nameof(typeof(s)))
_linear_name(s::FieldSplitSolver) = _KSP_NAMES[s.krylov] * "+FS-" * _SPLIT_NAMES[s.split] * "(" * join(_block_name.(s.blocks), ",") * ")"

# ------------------------------------------------------------------------------------------------
# Blocks sliced from the tangent in a fixed pattern

"A block `A[rows, cols]` (times `sign`) with the positions of its entries in `A.nzval`, refilled without allocation."
struct BlockSlice
  M::SparseMatrixCSC{Float64,Int}
  map::Vector{Int}
  sign::Float64
end

function BlockSlice(A::SparseMatrixCSC, rows::UnitRange{Int}, cols::UnitRange{Int}, sign::Float64)
  colptr = Vector{Int}(undef, length(cols) + 1)
  rowval = Int[]; map = Int[]
  colptr[1] = 1
  lo, hi = first(rows), last(rows)
  @inbounds for (jj, j) in enumerate(cols)
    for k in A.colptr[j]:A.colptr[j+1]-1
      i = A.rowval[k]
      if lo ≤ i ≤ hi
        push!(rowval, i - lo + 1); push!(map, k)
      end
    end
    colptr[jj+1] = length(rowval) + 1
  end
  M = SparseMatrixCSC(length(rows), length(cols), colptr, rowval, zeros(length(rowval)))
  s = BlockSlice(M, map, sign)
  _refill!(s, A)
end

function _refill!(s::BlockSlice, A::SparseMatrixCSC)
  nz, map, sign = s.M.nzval, s.map, s.sign
  Threads.@threads :static for k in eachindex(map)
    @inbounds nz[k] = sign * A.nzval[map[k]]
  end
  s
end

"Sign of the block (i, j): −1 on the negated rows."
_block_sign(s::FieldSplitSolver, i::Int) = s.negate !== nothing && s.fields[i] == s.negate ? -1.0 : 1.0

"Whether the (lower / upper) off-diagonal block (i, j) is applied by the split."
_block_needed(s::FieldSplitSolver, i::Int, j::Int) =
  s.split === :multiplicative ? j < i : s.split === :schur ? (i, j) == (1, 2) : false

# ------------------------------------------------------------------------------------------------
# The negated tangent as the Krylov operator (no copy: the product, then the sign of the negated rows)

struct NegatedOperator{TA<:SparseMatrixCSC}
  A::TA
  negate::Union{Nothing,UnitRange{Int}}
  symmetric::Bool
end
Base.size(op::NegatedOperator) = size(op.A)
Base.size(op::NegatedOperator, i::Int) = size(op.A, i)
Base.eltype(::NegatedOperator) = Float64
function LinearAlgebra.mul!(y::AbstractVector, op::NegatedOperator, x::AbstractVector)
  _mul!(y, op.A, x, op.symmetric)
  _negate!(y, op.negate)
end
Base.:*(op::NegatedOperator, x::AbstractVector) = mul!(similar(x, size(op.A, 1)), op, x)

function _negate!(y::AbstractVector, r::Union{Nothing,UnitRange{Int}})
  r === nothing && return y
  Threads.@threads :static for i in r
    @inbounds y[i] = -y[i]
  end
  y
end

# ------------------------------------------------------------------------------------------------
# One application of a block solver

"Whether the block's numerical setup can be applied (the multigrid reports indefinite or non-finite blocks)."
_block_usable(ns::HMultigridNS) = ns.ok
_block_usable(::Algebra.NumericalSetup) = true

"`z = P⁻¹ r` with one application of the block solver: a V-cycle of the multigrid, a solve from zero otherwise."
_block_apply!(z::AbstractVector, ns::HMultigridNS, r::AbstractVector) = _vcycle!(z, ns, length(ns.solver.spaces), r)
_block_apply!(z::AbstractVector, ns::Algebra.NumericalSetup, r::AbstractVector) = (fill!(z, 0.0); solve!(z, ns, r))

_block_builds(ns::Algebra.NumericalSetup) = preconditioner_builds(ns)

# ------------------------------------------------------------------------------------------------
# Numerical setup and the preconditioner

mutable struct FieldSplitNS{S<:FieldSplitSolver,TA<:AbstractMatrix} <: Algebra.NumericalSetup
  const solver::S
  A::TA                                      # the matrix of the last setup (the Newton cache's)
  Af::SparseMatrixCSC{Float64,Int}           # the same as CSC
  diag::Vector{BlockSlice}                   # diagonal blocks (negated rows with their sign)
  off::Matrix{Union{Nothing,BlockSlice}}     # off-diagonal blocks the split applies
  block_ns::Vector{Any}                      # numerical setups of the block solvers
  r::Vector{Vector{Float64}}                 # work per field: right-hand side of the block application
  z::Vector{Vector{Float64}}                 # and its result
  w::Vector{Float64}                         # copy of the input of the in-place application
  ok::Bool
  failure::Int                               # 0, or the block whose solver is not usable
  nsetup::Int
  nbuild::Int
  iters::Int
  rnorm::Float64
  converged::Bool
  rtol::Float64
end

struct FieldSplitSS{S<:FieldSplitSolver} <: Algebra.SymbolicSetup
  solver::S
end
Algebra.symbolic_setup(s::FieldSplitSolver, ::AbstractMatrix) = FieldSplitSS(s)

function _slices!(ns::FieldSplitNS)
  s = ns.solver
  f = s.fields
  n = length(f)
  ns.diag = [BlockSlice(ns.Af, f[i], f[i], _block_sign(s, i)) for i in 1:n]
  ns.off = [_block_needed(s, i, j) ? BlockSlice(ns.Af, f[i], f[j], _block_sign(s, i)) : nothing for i in 1:n, j in 1:n]
  ns
end

function _refill_slices!(ns::FieldSplitNS)
  for d in ns.diag
    _refill!(d, ns.Af)
  end
  for o in ns.off
    o === nothing || _refill!(o, ns.Af)
  end
  ns
end

_same_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC) = size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval

"Sets up the block solvers on the current diagonal blocks (first time: creates the setups)."
function _build_blocks!(ns::FieldSplitNS)
  s = ns.solver
  ns.ok = true
  ns.failure = 0
  for (i, b) in enumerate(s.blocks)
    M = ns.diag[i].M
    if ns.block_ns[i] === nothing
      ns.block_ns[i] = numerical_setup(symbolic_setup(b, M), M)
    else
      numerical_setup!(ns.block_ns[i], M)
    end
    if ns.ok && !_block_usable(ns.block_ns[i])
      ns.ok = false; ns.failure = i
    end
  end
  ns.nbuild += 1
  ns
end

function numerical_setup(ss::FieldSplitSS, A::AbstractMatrix)
  s = ss.solver
  last(s.fields[end]) == size(A, 1) ||
    error("HyperFEM: FieldSplitSolver: the field ranges end at $(last(s.fields[end])) but the matrix has $(size(A, 1)) rows")
  n = length(s.fields)
  Af = _sparse_fine(A)
  ns = FieldSplitNS{typeof(s),typeof(A)}(s, A, Af, BlockSlice[], Matrix{Union{Nothing,BlockSlice}}(nothing, 0, 0), Any[nothing for _ in 1:n],
                                        [zeros(length(r)) for r in s.fields], [zeros(length(r)) for r in s.fields], zeros(size(A, 1)),
                                        false, 0, 1, 0, 0, 0.0, true, s.rtol)
  _slices!(ns)
  _build_blocks!(ns)
  ns
end

function numerical_setup!(ns::FieldSplitNS, A::AbstractMatrix)
  s = ns.solver
  ns.nsetup += 1
  if A !== ns.A                                # a new matrix object: new slices (new block objects), the block solvers rebuild
    ns.A = A
    ns.Af = _sparse_fine(A)
    old = ns.diag
    _slices!(ns)
    for i in eachindex(old)                    # a block with a new pattern (a different mesh, dropped zeros) needs a new setup
      _same_pattern(old[i].M, ns.diag[i].M) || (ns.block_ns[i] = nothing)
    end
    _build_blocks!(ns)
  else
    ns.Af = _sparse_fine(A)
    _refill_slices!(ns)                        # the block objects keep their identity: the block solvers apply their own lag
    ((ns.nsetup - 1) % s.reuse == 0 || !ns.ok) && _build_blocks!(ns)
  end
  ns
end

"The block preconditioner object IterativeSolvers applies (right preconditioning)."
struct FieldSplitPrec{NS<:FieldSplitNS}
  ns::NS
end

function _block_solve!(y, ns::FieldSplitNS, i)
  _block_apply!(ns.z[i], ns.block_ns[i], ns.r[i])
  copyto!(view(y, ns.solver.fields[i]), ns.z[i])
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::FieldSplitPrec, x::AbstractVector)
  ns = P.ns
  s = ns.solver
  f, r, z = s.fields, ns.r, ns.z
  n = length(f)
  for i in 1:n
    copyto!(r[i], view(x, f[i]))
  end
  if s.split === :additive
    for i in 1:n
      _block_solve!(y, ns, i)
    end
  elseif s.split === :multiplicative           # lower block triangular: y_i = P_i⁻¹ (x_i − Σ_{j<i} A_ij y_j)
    for i in 1:n
      for j in 1:i-1
        mul!(r[i], ns.off[i, j].M, z[j], -1.0, 1.0)
      end
      _block_solve!(y, ns, i)
    end
  else                                         # :schur, upper, S ≈ A₂₂: y_2 = P_2⁻¹ x_2, y_1 = P_1⁻¹ (x_1 − A_12 y_2)
    _block_solve!(y, ns, 2)
    mul!(r[1], ns.off[1, 2].M, z[2], -1.0, 1.0)
    _block_solve!(y, ns, 1)
  end
  y
end
LinearAlgebra.ldiv!(P::FieldSplitPrec, x::AbstractVector) = (copyto!(P.ns.w, x); ldiv!(x, P, P.ns.w))

function Algebra.solve!(x::AbstractVector, ns::FieldSplitNS, b::AbstractVector)
  s = ns.solver
  fill!(x, zero(eltype(x)))
  if !ns.ok
    ns.iters, ns.converged, ns.rnorm = 0, false, NaN
    return x
  end
  op = NegatedOperator(ns.Af, s.negate, s.symmetric)
  _negate!(b, s.negate)
  # right preconditioning: the residual the method monitors is the true one, ‖b − A x‖ ≤ rtol ‖b‖, as the PETSc solver
  h = _with_serial_blas() do                 # OpenBLAS's spinning threads would starve the threaded V-cycles (see HMultigrid.jl)
    IterativeSolvers.gmres!(x, op, b; Pr=FieldSplitPrec(ns), reltol=ns.rtol, abstol=s.atol, maxiter=s.maxiter,
                            restart=min(s.restart, length(b)), initially_zero=true, log=true)[2]
  end
  _negate!(b, s.negate)
  ns.iters = h.iters
  ns.converged = h.isconverged
  res = h[:resnorm]
  ns.rnorm = isempty(res) ? 0.0 : res[end]
  x
end

linear_converged(ns::FieldSplitNS) = ns.converged
linear_iterations(ns::FieldSplitNS) = ns.iters
linear_indefinite(ns::FieldSplitNS) = !ns.ok && ns.failure > 0 && linear_indefinite(ns.block_ns[ns.failure])
preconditioner_builds(ns::FieldSplitNS) = ns.nbuild
linear_info(ns::FieldSplitNS) = ns.ok ? " $(ns.iters)" : " failed (block $(ns.failure))"
set_tolerance!(ns::FieldSplitNS, rtol) = (ns.rtol = rtol === nothing ? ns.solver.rtol : _check_rtol(rtol); ns)
tolerance(ns::FieldSplitNS) = ns.rtol
