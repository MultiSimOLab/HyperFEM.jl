module HyperFEMPETScExt

# PETSc Krylov + AMG / direct solvers for HyperFEM's Newton (design/solvers.md, S1): the methods of
# `PETScSolver` as a Gridap `LinearSolver` on GridapPETSc's `KSP`. Activated by `using GridapPETSc`
# alone, so that PETSc can be used in serial (HyperFEMDistributed needs GridapDistributed too).
#
# Design:
# - The CSC tangent of the Newton cache is mirrored once into a 0-based CSR (`CSRMirror`): index
#   arrays and a gather permutation computed at the first setup, PETSc's `Mat` aliasing the CSR
#   arrays (`MatCreateSeqAIJWithArrays`). Every later numerical setup gathers the values (threaded),
#   bumps the object state and lets the KSP rebuild its preconditioner, or not (`reuse`, lagged AMG).
# - The near-null space (`near_nullspace(U)`) is attached to the matrix with `MatSetNearNullSpace`;
#   GAMG and BoomerAMG read it.
# - Options go through PETSc's options database under a prefix unique to the solver instance, so
#   several solvers (e.g. the u and φ blocks of S2) coexist and `options` can carry any PETSc flag.
# - `linear_converged`, `linear_iterations` and `linear_info` report the last solve to the Newton.
# Vectors of other types (e.g. PETSc's own, PartitionedArrays') go through GridapPETSc's `convert`.

using HyperFEM
using HyperFEM.Solvers: PETScSolver, _linear_name, split_nullspace
import HyperFEM.Solvers: petsc_symbolic_setup, linear_converged, linear_iterations, linear_info, preconditioner_builds
using Gridap.Algebra
import Gridap.Algebra: numerical_setup, numerical_setup!, solve!
using GridapPETSc
using GridapPETSc: PETSC, PETScMatrix, PETScVector, PETScIndexSet, PetscScalar, PetscInt, PetscReal, @check_error_code, libpetsc_handle
using GridapPETSc.PETSC: Mat, Vec, KSP, PC, IS, MatNullSpace, PETSC_TRUE, PETSC_FALSE, PetscBool
using GridapPETSc: SparseMatrixCSR
using SparseArrays
using LinearAlgebra
const Libdl = GridapPETSc.Libdl
const MPI = GridapPETSc.MPI

# ------------------------------------------------------------------------------------------------
# PETSc functions GridapPETSc does not wrap (looked up lazily in the PETSc library)

const _SYMBOLS = Dict{Symbol,Ptr{Cvoid}}()
function _sym(name::Symbol)
  get!(_SYMBOLS, name) do
    p = Libdl.dlsym(libpetsc_handle[], name; throw_error=false)
    p === nothing && error("HyperFEM: symbol $name not found in the PETSc library")
    p
  end
end

PetscOptionsSetValue(name::String, value::String) =
  ccall(_sym(:PetscOptionsSetValue), Cint, (Ptr{Cvoid}, Cstring, Cstring), C_NULL, name, value)
KSPGetConvergedReason(ksp::KSP, reason::Ref{Cint}) =
  ccall(_sym(:KSPGetConvergedReason), Cint, (KSP, Ptr{Cint}), ksp, reason)
KSPSetReusePreconditioner(ksp::KSP, flag::PetscBool) =
  ccall(_sym(:KSPSetReusePreconditioner), Cint, (KSP, PetscBool), ksp, flag)
PCFieldSplitSetIS(pc::PC, name::String, is::IS) =
  ccall(_sym(:PCFieldSplitSetIS), Cint, (PC, Cstring, IS), pc, name, is)
PetscObjectCompose(obj::Ptr{Cvoid}, name::String, other::Ptr{Cvoid}) =
  ccall(_sym(:PetscObjectCompose), Cint, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), obj, name, other)
PetscOptionsClearValue(name::String) =
  ccall(_sym(:PetscOptionsClearValue), Cint, (Ptr{Cvoid}, Cstring), C_NULL, name)
# "The values of the matrix changed behind PETSc's back": the Get/Restore pair bumps the object state and
# invalidates the caches of the AIJ matrix (inverse diagonal of the SOR smoothers), which a bare state bump
# would not do
function MatValuesChanged(mat::Mat)
  arr = Ref{Ptr{PetscScalar}}()
  err = ccall(_sym(:MatSeqAIJGetArray), Cint, (Mat, Ptr{Ptr{PetscScalar}}), mat, arr)
  err == 0 || return err
  ccall(_sym(:MatSeqAIJRestoreArray), Cint, (Mat, Ptr{Ptr{PetscScalar}}), mat, arr)
end

# ------------------------------------------------------------------------------------------------
# CSR mirror of a CSC matrix (PETSc's AIJ is CSR, 0-based)

"""
CSR copy of a `SparseMatrixCSC` with the gather `perm` such that `csr.nzval[k] = A.nzval[perm[k]]`:
the index arrays are built once, the values are gathered at every update. The mirror is valid while
`A` keeps its index arrays (checked by identity).
"""
struct CSRMirror{Ti<:Integer}
  csr::SparseMatrixCSR{0,PetscScalar,PetscInt}
  perm::Vector{Ti}
  colptr::Vector                              # the index arrays of the source matrix (identity check, no copy)
  rowval::Vector
end

function CSRMirror(A::SparseMatrixCSC{<:Real,<:Integer})
  m, n = size(A)
  nz = nnz(A)
  Ti = nz < typemax(Int32) ? Int32 : Int
  colptr, rowval = A.colptr, A.rowval
  rowptr = zeros(PetscInt, m + 1)             # 0-based: rowptr[i] = nonzeros before row i, rowptr[m+1] = nnz
  @inbounds for k in 1:nz
    rowptr[rowval[k]+1] += 1                 # count row i at rowptr[i+1]
  end
  @inbounds for i in 1:m
    rowptr[i+1] += rowptr[i]
  end
  next = rowptr[1:m] .+ 1                     # next free slot (1-based) of every row
  colval = Vector{PetscInt}(undef, nz)
  perm = Vector{Ti}(undef, nz)
  @inbounds for j in 1:n, k in colptr[j]:colptr[j+1]-1
    i = rowval[k]
    p = next[i]
    colval[p] = j - 1
    perm[p] = k
    next[i] = p + 1
  end
  nzval = Vector{PetscScalar}(undef, nz)
  csr = SparseMatrixCSR{0}(m, n, rowptr, colval, nzval)
  mirror = CSRMirror{Ti}(csr, perm, colptr, rowval)
  gather!(mirror, A)
  mirror
end

"Whether `A` still has the index arrays the mirror was built from."
same_pattern(mirror::CSRMirror, A::SparseMatrixCSC) = size(A, 1) == mirror.csr.m && size(A, 2) == mirror.csr.n &&
  nnz(A) == length(mirror.perm) && A.colptr === mirror.colptr && A.rowval === mirror.rowval

# Gathers the values of `A` into the CSR mirror (threaded, no allocation); the rows in `negate` change sign.
# `:static` scheduling: with the dynamic scheduler a garbage collection triggered from a worker thread runs
# GridapPETSc's finalizers there, and they assert the main thread.
function gather!(mirror::CSRMirror, A::SparseMatrixCSC, negate=nothing)
  v, src, perm = mirror.csr.nzval, A.nzval, mirror.perm
  if negate === nothing
    Threads.@threads :static for k in eachindex(perm)
      @inbounds v[k] = src[perm[k]]
    end
  else
    rowptr = mirror.csr.rowptr
    Threads.@threads :static for i in 1:mirror.csr.m
      s = i in negate ? -1.0 : 1.0
      @inbounds for k in rowptr[i]+1:rowptr[i+1]
        v[k] = s * src[perm[k]]
      end
    end
  end
  mirror
end

"Changes the sign of the entries `negate` of `b` (in place; applied before and after the solve)."
function negate!(b::AbstractVector, negate)
  negate === nothing && return b
  @inbounds for i in negate
    b[i] = -b[i]
  end
  b
end

# ------------------------------------------------------------------------------------------------
# Symbolic and numerical setups

struct PETScSolverSS <: SymbolicSetup
  solver::PETScSolver
end

petsc_symbolic_setup(s::PETScSolver, ::AbstractMatrix) = PETScSolverSS(s)

mutable struct PETScSolverNS{M} <: NumericalSetup
  const solver::PETScSolver
  const prefix::String
  mirror::M                                    # CSRMirror (CSC fast path) or nothing
  A::AbstractMatrix                            # the Julia matrix of the last setup
  B::PETScMatrix
  const ksp::Ref{KSP}
  nullsp::Ref{MatNullSpace}
  nullvecs::Vector{PETScVector}
  has_nullspace::Bool
  fieldsets::Vector{PETScIndexSet}            # fieldsplit: the IS of every field, with its near-null space composed
  fieldnull::Vector{Ref{MatNullSpace}}
  fieldnullvecs::Vector{PETScVector}
  nsetup::Int                                  # numerical setups (initial one included)
  nbuild::Int                                  # preconditioner builds
  iters::Int
  rnorm::Float64
  reason::Cint
  initialized::Bool
end

const _NSOLVERS = Ref(0)

function _ensure_init()
  GridapPETSc.Initialized() && return
  GridapPETSc.Init()
end

"Options of the solver as PETSc flags (without prefix)."
function _options(s::PETScSolver)
  opts = Pair{String,String}[]
  push!(opts, "ksp_type" => String(s.ksp))
  if s.pc === :gamg
    push!(opts, "pc_type" => "gamg", "pc_gamg_type" => "agg")
  elseif s.pc === :boomeramg
    push!(opts, "pc_type" => "hypre", "pc_hypre_type" => "boomeramg")
  elseif s.pc === :mumps
    push!(opts, "pc_type" => "lu", "pc_factor_mat_solver_type" => "mumps")
  elseif s.pc === :mumps_cholesky
    push!(opts, "pc_type" => "cholesky", "pc_factor_mat_solver_type" => "mumps")
  elseif s.pc === :lu
    push!(opts, "pc_type" => "lu")
  elseif s.pc === :fieldsplit
    push!(opts, "pc_type" => "fieldsplit", "pc_fieldsplit_type" => String(s.split))
    if s.split === :schur
      push!(opts, "pc_fieldsplit_schur_fact_type" => String(s.schur.fact), "pc_fieldsplit_schur_precondition" => String(s.schur.precondition))
    end
    for (name, b) in zip(s.fieldnames, s.blocks)   # one application of the block preconditioner per outer iteration
      pre = "fieldsplit_" * name * "_"
      push!(opts, pre * "ksp_type" => "preonly")
      if b === :gamg
        push!(opts, pre * "pc_type" => "gamg", pre * "pc_gamg_type" => "agg")
      elseif b === :boomeramg
        push!(opts, pre * "pc_type" => "hypre", pre * "pc_hypre_type" => "boomeramg")
      elseif b === :mumps
        push!(opts, pre * "pc_type" => "lu", pre * "pc_factor_mat_solver_type" => "mumps")
      else
        push!(opts, pre * "pc_type" => String(b))
      end
    end
  else
    push!(opts, "pc_type" => String(s.pc))
  end
  if s.ksp !== :preonly
    push!(opts, "ksp_norm_type" => "unpreconditioned")   # ‖b - A x‖ ≤ rtol ‖b‖: the tolerance the Newton reasons about
    s.ksp in (:gmres, :fgmres) && push!(opts, "ksp_pc_side" => "right")
  end
  s.monitor && push!(opts, "ksp_monitor" => "")
  # user options: "-a b -c -d 1.5" → a=b, c="", d=1.5
  words = split(s.options)
  i = 1
  while i ≤ length(words)
    w = words[i]
    startswith(w, "-") && tryparse(Float64, w) === nothing ||
      error("HyperFEM: PETScSolver options: expected a flag starting with '-', got '$w'")
    name = String(w[2:end])
    if i < length(words) && (!startswith(words[i+1], "-") || tryparse(Float64, words[i+1]) !== nothing)   # a value, e.g. -0.01
      push!(opts, name => String(words[i+1])); i += 2
    else
      push!(opts, name => ""); i += 1
    end
  end
  opts
end

function _set_options!(prefix::String, s::PETScSolver)
  for (name, value) in _options(s)
    @check_error_code PetscOptionsSetValue("-" * prefix * name, value)
  end
end

"The PETSc matrix of `A`: a CSR mirror aliased by PETSc for CSC matrices, GridapPETSc's conversion otherwise."
function _petsc_matrix(A::SparseMatrixCSC, negate)
  mirror = CSRMirror(A)
  negate === nothing || gather!(mirror, A, negate)
  mirror, PETScMatrix(mirror.csr)
end
function _petsc_matrix(A::AbstractMatrix, negate)
  negate === nothing || error("HyperFEM: PETScSolver: `negate` needs a SparseMatrixCSC tangent (got $(typeof(A)))")
  nothing, convert(PETScMatrix, A)
end

function _set_nullspace!(ns::PETScSolverNS, N::Matrix{Float64})
  ns.solver.pc === :fieldsplit && return nothing   # the near-null spaces go to the blocks (see _setup_fieldsplit!)
  ns.B.comm == MPI.COMM_SELF || error("HyperFEM: PETScSolver: the near-null space is only supported on serial matrices")
  n, k = size(N)
  k == 0 && return nothing                       # nothing to tell PETSc (every mode vanished on the free dofs)
  size(ns.B, 1) == n || error("HyperFEM: PETScSolver: the near-null space has $n rows but the matrix has $(size(ns.B, 1))")
  ns.nullvecs = [PETScVector(N[:, j]) for j in 1:k]      # own copies of the columns, aliased by PETSc
  vecs = [v.vec[] for v in ns.nullvecs]
  @check_error_code PETSC.MatNullSpaceCreate(MPI.COMM_SELF, PETSC_FALSE, k, vecs, ns.nullsp)
  @check_error_code PETSC.MatSetNearNullSpace(ns.B.mat[], ns.nullsp[])
  ns.has_nullspace = true
  nothing
end

# Fieldsplit: the IS of every field on the KSP's PC, with the field's near-null space composed on it under
# "nearnullspace" (PCFIELDSPLIT hands it to the sub-matrix of the block, where GAMG reads it).
function _setup_fieldsplit!(ns::PETScSolverNS)
  s = ns.solver
  ns.B.comm == MPI.COMM_SELF || error("HyperFEM: PETScSolver: fieldsplit is only supported on serial matrices")
  n = size(ns.B, 1)
  last(s.fields[end]) == n || error("HyperFEM: PETScSolver: the field ranges end at $(last(s.fields[end])) but the matrix has $n rows")
  pc = Ref{PC}()
  @check_error_code PETSC.KSPGetPC(ns.ksp[], pc)
  @check_error_code PETSC.PCSetType(pc[], "fieldsplit")
  Ns = s.nullspace === nothing ? nothing : split_nullspace(s.nullspace, s.fields)
  for (i, r) in enumerate(s.fields)
    is = PETScIndexSet(r)                        # 1-based indices, converted inside
    push!(ns.fieldsets, is)
    if Ns !== nothing && size(Ns[i], 2) > 0
      Ni = Ns[i]
      vecs = [PETScVector(Ni[:, j]) for j in 1:size(Ni, 2)]
      append!(ns.fieldnullvecs, vecs)
      sp = Ref{MatNullSpace}()
      @check_error_code PETSC.MatNullSpaceCreate(MPI.COMM_SELF, PETSC_FALSE, size(Ni, 2), [v.vec[] for v in vecs], sp)
      push!(ns.fieldnull, sp)
      @check_error_code PetscObjectCompose(is.is[].ptr, "nearnullspace", sp[].ptr)
    end
    @check_error_code PCFieldSplitSetIS(pc[], s.fieldnames[i], is.is[])
  end
  nothing
end

function _destroy_fieldsplit!(ns::PETScSolverNS)
  for sp in ns.fieldnull
    @check_error_code PETSC.MatNullSpaceDestroy(sp)
  end
  empty!(ns.fieldnull)
  foreach(GridapPETSc.destroy, ns.fieldnullvecs); empty!(ns.fieldnullvecs)
  foreach(GridapPETSc.destroy, ns.fieldsets); empty!(ns.fieldsets)
  nothing
end

function _set_blocksize!(B::PETScMatrix, bs::Int)
  bs == 1 && return
  n = size(B, 1)
  n % bs == 0 || error("HyperFEM: PETScSolver: blocksize $bs does not divide the number of equations $n")
  @check_error_code PETSC.MatSetBlockSize(B.mat[], PetscInt(bs))
end

function numerical_setup(ss::PETScSolverSS, A::AbstractMatrix)
  _ensure_init()
  s = ss.solver
  _NSOLVERS[] += 1
  prefix = "hyperfem$(_NSOLVERS[])_"
  mirror, B = _petsc_matrix(A, s.negate)
  _set_blocksize!(B, s.blocksize)
  ns = PETScSolverNS{typeof(mirror)}(s, prefix, mirror, A, B, Ref{KSP}(), Ref{MatNullSpace}(), PETScVector[], false,
                                      PETScIndexSet[], Ref{MatNullSpace}[], PETScVector[], 1, 1, 0, 0.0, Cint(0), false)
  s.nullspace === nothing || _set_nullspace!(ns, s.nullspace)
  @check_error_code PETSC.KSPCreate(B.comm, ns.ksp)
  @check_error_code PETSC.KSPSetOptionsPrefix(ns.ksp[], prefix)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[], B.mat[], B.mat[])
  s.pc === :fieldsplit && _setup_fieldsplit!(ns)
  @check_error_code PETSC.KSPSetTolerances(ns.ksp[], s.rtol, s.atol, PetscReal(1e10), PetscInt(s.maxiter))
  _set_options!(prefix, s)
  @check_error_code PETSC.KSPSetFromOptions(ns.ksp[])   # after the tolerances, so that -ksp_rtol etc. in `options` win; the options
                                                        # stay in the database until destroy: the multigrid levels read theirs at KSPSetUp
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  _init!(ns)
end

function numerical_setup!(ns::PETScSolverNS, A::AbstractMatrix)
  s = ns.solver
  ns.nsetup += 1
  rebuilt = false
  if ns.mirror !== nothing && A isa SparseMatrixCSC && same_pattern(ns.mirror, A)
    gather!(ns.mirror, A, s.negate)
    @check_error_code MatValuesChanged(ns.B.mat[])
  else
    # new matrix object or pattern: rebuild the PETSc matrix (the near-null space goes with it) and the
    # preconditioner, whatever `reuse` says: a lagged AMG keeps references to the old matrix (its smoothers),
    # whose arrays are owned by the Julia mirror that goes away here
    ns.has_nullspace && @check_error_code PETSC.MatNullSpaceDestroy(ns.nullsp)
    ns.has_nullspace = false
    foreach(GridapPETSc.destroy, ns.nullvecs); empty!(ns.nullvecs)
    GridapPETSc.destroy(ns.B)
    ns.mirror, ns.B = _petsc_matrix(A, s.negate)
    _set_blocksize!(ns.B, s.blocksize)
    s.nullspace === nothing || _set_nullspace!(ns, s.nullspace)
    rebuilt = true
  end
  ns.A = A
  reuse = (ns.nsetup - 1) % s.reuse != 0 && !rebuilt
  @check_error_code KSPSetReusePreconditioner(ns.ksp[], reuse ? PETSC_TRUE : PETSC_FALSE)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[], ns.B.mat[], ns.B.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  reuse || (ns.nbuild += 1)
  ns
end

# ------------------------------------------------------------------------------------------------
# Solve

function _solve_vec!(ns::PETScSolverNS, X::PETScVector, Bv::PETScVector)
  @check_error_code PETSC.KSPSolve(ns.ksp[], Bv.vec[], X.vec[])   # the caller negates the rows of b if asked
  its = Ref{PetscInt}(); rn = Ref{PetscReal}(); reason = Ref{Cint}()
  @check_error_code PETSC.KSPGetIterationNumber(ns.ksp[], its)
  @check_error_code PETSC.KSPGetResidualNorm(ns.ksp[], rn)
  @check_error_code KSPGetConvergedReason(ns.ksp[], reason)
  ns.iters, ns.rnorm, ns.reason = Int(its[]), Float64(rn[]), reason[]
  nothing
end

function solve!(x::PETScVector, ns::PETScSolverNS, b::PETScVector)
  _solve_vec!(ns, x, b)
  x
end

# Julia vectors of the PETSc scalar type are aliased (no copy); others are converted by GridapPETSc.
function solve!(x::Vector{PetscScalar}, ns::PETScSolverNS, b::Vector{PetscScalar})
  x === b && error("HyperFEM: PETScSolver: x and b must be different vectors")
  negate!(b, ns.solver.negate)                 # (PETSc starts from x = 0: the initial guess is zero unless told otherwise)
  X = PETScVector(x); Bv = PETScVector(b)
  _solve_vec!(ns, X, Bv)
  GridapPETSc.destroy(X); GridapPETSc.destroy(Bv)
  negate!(b, ns.solver.negate)
  x
end

function solve!(x::AbstractVector, ns::PETScSolverNS, b::AbstractVector)
  (x isa PETScVector || b isa PETScVector) &&
    error("HyperFEM: PETScSolver: mixed PETSc / Julia vectors are not supported (pass two PETScVectors or two Julia vectors)")
  negate!(b, ns.solver.negate)
  X = convert(PETScVector, x); Bv = convert(PETScVector, b)
  _solve_vec!(ns, X, Bv)
  copy!(x, X)
  GridapPETSc.destroy(X); GridapPETSc.destroy(Bv)
  negate!(b, ns.solver.negate)
  x
end

# ------------------------------------------------------------------------------------------------
# Report to the Newton

linear_converged(ns::PETScSolverNS) = ns.reason > 0
linear_iterations(ns::PETScSolverNS) = ns.iters
preconditioner_builds(ns::PETScSolverNS) = ns.nbuild
linear_info(ns::PETScSolverNS) = ns.solver.ksp === :preonly ? "" : " $(ns.iters)"

"PETSc's name of the last convergence reason (e.g. `KSP_CONVERGED_RTOL`, `KSP_DIVERGED_ITS`)."
function converged_reason(ns::PETScSolverNS)
  names = Dict{Cint,String}(1 => "KSP_CONVERGED_RTOL_NORMAL", 9 => "KSP_CONVERGED_ATOL_NORMAL", 2 => "KSP_CONVERGED_RTOL",
    3 => "KSP_CONVERGED_ATOL", 4 => "KSP_CONVERGED_ITS", 5 => "KSP_CONVERGED_NEG_CURVE", 7 => "KSP_CONVERGED_STEP_LENGTH",
    8 => "KSP_CONVERGED_HAPPY_BREAKDOWN", -2 => "KSP_DIVERGED_NULL", -3 => "KSP_DIVERGED_ITS", -4 => "KSP_DIVERGED_DTOL",
    -5 => "KSP_DIVERGED_BREAKDOWN", -6 => "KSP_DIVERGED_BREAKDOWN_BICG", -7 => "KSP_DIVERGED_NONSYMMETRIC",
    -8 => "KSP_DIVERGED_INDEFINITE_PC", -9 => "KSP_DIVERGED_NANORINF", -10 => "KSP_DIVERGED_INDEFINITE_MAT",
    -11 => "KSP_DIVERGED_PC_FAILED", 0 => "KSP_CONVERGED_ITERATING")
  get(names, ns.reason, "KSP reason $(ns.reason)")
end

# ------------------------------------------------------------------------------------------------
# Lifetime of the PETSc objects (as GridapPETSc's PETScLinearSolverNS)

function _init!(ns::PETScSolverNS)
  @assert Threads.threadid() == 1
  GridapPETSc._NREFS[] += 1
  ns.initialized = true
  finalizer(_finalize, ns)
end

function GridapPETSc.destroy(ns::PETScSolverNS)
  if ns.initialized && GridapPETSc.Initialized()
    for (name, _) in _options(ns.solver)
      PetscOptionsClearValue("-" * ns.prefix * name)
    end
    @check_error_code PETSC.KSPDestroy(ns.ksp)
    ns.has_nullspace && @check_error_code PETSC.MatNullSpaceDestroy(ns.nullsp)
    ns.has_nullspace = false
    foreach(GridapPETSc.destroy, ns.nullvecs)
    _destroy_fieldsplit!(ns)
    GridapPETSc.destroy(ns.B)
    ns.initialized = false
    @assert Threads.threadid() == 1
    GridapPETSc._NREFS[] -= 1
  end
  nothing
end

function _finalize(ns::PETScSolverNS)
  if ns.initialized && GridapPETSc.Initialized()
    if ns.B.comm == MPI.COMM_SELF
      @check_error_code PETSC.KSPDestroy(ns.ksp)
      ns.has_nullspace && @check_error_code PETSC.MatNullSpaceDestroy(ns.nullsp)
      foreach(sp -> @check_error_code(PETSC.MatNullSpaceDestroy(sp)), ns.fieldnull)   # the ISs and vectors have their own finalizers
    else
      @check_error_code PETSC.PetscObjectRegisterDestroy(ns.ksp[].ptr)
    end
    ns.initialized = false
    GridapPETSc._NREFS[] -= 1
  end
  nothing
end

end # module
