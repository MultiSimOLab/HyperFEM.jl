# h-multigrid on a refined mesh hierarchy, as a Gridap `LinearSolver` for the assembled tangent (design/solvers-s4b-plan.md,
# S4-B step 2, option J of DS4B-3). Pure Julia: Galerkin coarse operators PᵀAP with SparseArrays, Chebyshev smoothing
# preconditioned by the inverted nodal blocks of each level, an injected solver on the coarsest level, IterativeSolvers'
# CG/GMRES outside. The same object serves CI (direct coarse solver) and production (PETSc GAMG on the coarsest level).

"""
    HMultigridSolver(spaces, Ps; krylov = :cg, rtol = 1e-6, atol = 0.0, maxiter = 500, restart = 30, reuse = 1,
                     degree = 2, range = 20.0, inflate = 1.1, eig_iters = 10, coarse = LUSolver())
    HMultigridSolver(h::MeshHierarchy, space::Function; kwargs...)

Krylov method (`:cg` for symmetric positive definite tangents, `:gmres` otherwise) preconditioned by one V-cycle of
geometric h-multigrid on a hierarchy of nested meshes: `spaces` are the test spaces of the levels, coarsest first (the
finest is the Newton's), and `Ps[l]` the prolongation from level `l` to level `l + 1` ([`prolongation`](@ref), free dofs,
homogeneous Dirichlet). The coarse operators are the Galerkin products `A_l = P_lᵀ A_{l+1} P_l` of the assembled
tangent, recomputed at every numerical setup (every `reuse` setups: lagged, as `PETScSolver`'s `reuse`); the
smoother on every level but the coarsest is a Chebyshev iteration of `degree` preconditioned by the inverted nodal
blocks of the level (the dofs of one node, 3 × 3 for 3D elasticity; scalar Jacobi for scalar fields), on the interval
`[λ/range, λ]` with `λ = inflate · λmax`, `λmax` a Ritz estimate of the block-preconditioned level operator from
`eig_iters` Lanczos steps in the block inner product from a fixed start vector (reproducible; real Ritz values inside the spectrum); the coarsest level is solved by `coarse`, any Gridap
`LinearSolver` applied once per cycle (a direct solver, or one GAMG V-cycle: `PETScSolver(:preonly, :gamg; nullspace =
near_nullspace(U_coarse))` with the rigid modes of the coarse space). Pre- and post-smoothing use the same polynomial,
so the cycle is symmetric and CG is valid on SPD tangents.

Chebyshev needs a positive real spectrum: a negative Ritz value (an indefinite tangent, past a limit point), a
non-finite tangent or a singular nodal block make the solve fail at once (`linear_converged == false`, zero iterations),
so the Newton cuts back instead of iterating on a diverging smoother. Saddle-point (u, φ) tangents are not for this
solver as a whole: use it on the blocks of a field-split preconditioner.

The Newton hooks: [`linear_converged`](@ref), [`linear_iterations`](@ref), [`preconditioner_builds`](@ref) (Galerkin
and smoother rebuilds), [`set_tolerance!`](@ref)/[`tolerance`](@ref) (Eisenstat–Walker forcing term). Serial sparse
products (threads only in the assembly); the PETSc PCMG route is the MPI one (DS4B-3).
"""
struct HMultigridSolver{S<:Tuple,P<:Tuple,C<:Algebra.LinearSolver} <: Algebra.LinearSolver
  spaces::S                                  # test spaces per level, coarsest first
  Ps::P                                      # Ps[l]: level l → l + 1
  Pts::P                                     # their transposes as CSC (the prolongation as a threaded transposed product)
  krylov::Symbol
  rtol::Float64
  atol::Float64
  maxiter::Int
  restart::Int
  reuse::Int
  degree::Int
  range::Float64
  inflate::Float64
  eig_iters::Int
  coarse::C
  function HMultigridSolver(spaces, Ps; krylov::Symbol=:cg, rtol::Real=1e-6, atol::Real=0.0, maxiter::Integer=500,
                            restart::Integer=30, reuse::Integer=1, degree::Integer=2, range::Real=20.0, inflate::Real=1.1,
                            eig_iters::Integer=10, coarse::Algebra.LinearSolver=LUSolver())
    krylov in (:cg, :gmres) || error("HyperFEM: HMultigridSolver: unknown Krylov method $krylov (use :cg or :gmres)")
    L = length(spaces)
    L ≥ 2 || error("HyperFEM: HMultigridSolver needs at least two levels")
    length(Ps) == L - 1 || error("HyperFEM: HMultigridSolver: $(L) levels need $(L - 1) prolongations (got $(length(Ps)))")
    for l in 1:L-1
      size(Ps[l]) == (num_free_dofs(spaces[l+1]), num_free_dofs(spaces[l])) ||
        error("HyperFEM: HMultigridSolver: prolongation $l is $(size(Ps[l])), expected $((num_free_dofs(spaces[l+1]), num_free_dofs(spaces[l])))")
    end
    reuse ≥ 1 || error("HyperFEM: HMultigridSolver: reuse must be ≥ 1")
    degree ≥ 1 || error("HyperFEM: HMultigridSolver: degree must be ≥ 1")
    range > 1 || error("HyperFEM: HMultigridSolver: range must be > 1")
    inflate ≥ 1 || error("HyperFEM: HMultigridSolver: inflate must be ≥ 1")
    eig_iters ≥ 2 || error("HyperFEM: HMultigridSolver: eig_iters must be ≥ 2")
    ps = Tuple(SparseMatrixCSC{Float64,Int}(P) for P in Ps)
    pts = Tuple(SparseMatrixCSC{Float64,Int}(sparse(transpose(P))) for P in ps)
    sp = Tuple(spaces)
    new{typeof(sp),typeof(ps),typeof(coarse)}(sp, ps, pts, krylov, rtol, atol, maxiter, restart, reuse, degree, range, inflate,
                                               eig_iters, coarse)
  end
end

_linear_name(s::HMultigridSolver) = _KSP_NAMES[s.krylov] * "+hMG($(length(s.spaces)))"

# ------------------------------------------------------------------------------------------------
# Threaded products with a symmetric sparse matrix

"""
`y = A x` for a **symmetric** `A` in CSC storage, threaded over columns: `A x = Aᵀ x`, and the transposed product of a CSC
matrix is a dot product per column, independent across threads (Julia's `mul!` with a CSC matrix scatters by columns and
runs serially). The tangents this solver accepts are symmetric (CG) — for `:gmres` the serial `mul!` is used.
"""
function _tmul!(y::AbstractVector, M::SparseMatrixCSC, x::AbstractVector)
  colptr, rowval, nzval = M.colptr, M.rowval, M.nzval
  Threads.@threads :static for j in 1:size(M, 2)
    s = 0.0
    @inbounds for k in colptr[j]:colptr[j+1]-1
      s += nzval[k] * x[rowval[k]]
    end
    @inbounds y[j] = s
  end
  y
end
"`y = Mᵀ x` for any CSC matrix `M`, threaded over its columns (the transfers: `Pᵀ r` with `P`, `P xc` with `Pᵀ` stored)."
_symmul!(y, A::SparseMatrixCSC, x) = _tmul!(y, A, x)

"Product `y = A x` of a level operator: threaded for symmetric tangents, Julia's `mul!` otherwise."
_mul!(y, A::SparseMatrixCSC, x, symmetric::Bool) = symmetric ? _symmul!(y, A, x) : mul!(y, A, x)

"The fine tangent as the operator the Krylov method applies (threaded product when symmetric)."
struct LevelOperator{TA<:SparseMatrixCSC}
  A::TA
  symmetric::Bool
end
Base.size(op::LevelOperator) = size(op.A)
Base.size(op::LevelOperator, i::Int) = size(op.A, i)
Base.eltype(::LevelOperator) = Float64
LinearAlgebra.mul!(y::AbstractVector, op::LevelOperator, x::AbstractVector) = _mul!(y, op.A, x, op.symmetric)
Base.:*(op::LevelOperator, x::AbstractVector) = _mul!(similar(x, size(op.A, 1)), op.A, x, op.symmetric)

# ------------------------------------------------------------------------------------------------
# Nodal blocks of a level: the free dofs of every node, from the cell dof ids and the Lagrangian layout

"Groups of free dofs that share a node (one group per node, components present as free dofs), coarsest-level numbering."
function nodal_blocks(V::SingleFieldFESpace)
  db = Gridap.Arrays.testitem(Gridap.CellData.get_data(get_fe_dof_basis(V)))
  db isa Gridap.ReferenceFEs.LagrangianDofBasis ||
    error("HyperFEM: HMultigridSolver: nodal blocks need Lagrangian (nodal) spaces, got $(nameof(typeof(db)))")
  n2d = db.node_and_comp_to_dof
  ncomp = n2d[1] isa Integer ? 1 : length(n2d[1])
  ldof(n, c) = ncomp == 1 ? Int(n2d[n]) : Int(n2d[n][c])
  nfree = num_free_dofs(V)
  block_of = zeros(Int, nfree)
  blocks = Vector{Int}[]
  ids = get_cell_dof_ids(V)
  cache = Gridap.Arrays.array_cache(ids)
  for cell in 1:length(ids)
    d = Gridap.Arrays.getindex!(cache, ids, cell)
    for n in eachindex(n2d)
      first_free = 0
      for c in 1:ncomp
        g = d[ldof(n, c)]
        g > 0 || continue
        if block_of[g] == 0
          if first_free == 0
            push!(blocks, Int[]); first_free = length(blocks)
          end
          block_of[g] = first_free
          push!(blocks[first_free], g)
        else
          first_free == 0 && (first_free = block_of[g])
        end
      end
    end
  end
  all(>(0), block_of) || error("HyperFEM: HMultigridSolver: a free dof belongs to no cell")
  blocks
end

function nodal_blocks(V::MultiFieldFESpace)
  MultiFieldStyle(V) isa ConsecutiveMultiFieldStyle || error("HyperFEM: HMultigridSolver: only ConsecutiveMultiFieldStyle multi-field spaces")
  blocks = Vector{Int}[]
  off = 0
  for s in V.spaces
    for b in nodal_blocks(s)
      push!(blocks, b .+ off)
    end
    off += num_free_dofs(s)
  end
  blocks
end
nodal_blocks(U::TrialFESpace) = nodal_blocks(U.space)

"Inverted nodal blocks of `A` (block Jacobi); `ok` false if a block is singular or non-finite."
struct BlockJacobiPrec
  blocks::Vector{Vector{Int}}
  blk::Vector{Matrix{Float64}}               # the nodal blocks of A (the D-inner product of the Lanczos estimate)
  inv::Vector{Matrix{Float64}}               # their inverses
  block_of::Vector{Int}                      # dof → block
  pos_in::Vector{Int}                        # dof → position inside its block
  ok::Base.RefValue{Bool}
end

function BlockJacobiPrec(blocks)
  n = sum(length, blocks; init=0)
  block_of = zeros(Int, n); pos_in = zeros(Int, n)
  for (k, b) in enumerate(blocks), (i, d) in enumerate(b)
    block_of[d] = k; pos_in[d] = i
  end
  BlockJacobiPrec(blocks, [zeros(length(b), length(b)) for b in blocks], [zeros(length(b), length(b)) for b in blocks],
                  block_of, pos_in, Ref(true))
end

function _update_blocks!(D::BlockJacobiPrec, A::SparseMatrixCSC)
  colptr, rowval, nzval = A.colptr, A.rowval, A.nzval
  block_of, pos_in = D.block_of, D.pos_in
  for M in D.blk
    fill!(M, 0.0)
  end
  @inbounds for j in 1:size(A, 2)                         # one sweep over the nonzeros: the entries inside a block
    kj = block_of[j]; pj = pos_in[j]
    for k in colptr[j]:colptr[j+1]-1
      i = rowval[k]
      block_of[i] == kj && (D.blk[kj][pos_in[i], pj] = nzval[k])
    end
  end
  ok = true
  for (M, Mi) in zip(D.blk, D.inv)
    if all(isfinite, M) && (size(M, 1) == 1 ? M[1, 1] != 0 : abs(det(M)) > 0)
      Mi .= inv(M)
    else
      ok = false
    end
  end
  D.ok[] = ok
  D
end

"`y = D x` with the nodal blocks themselves."
function _dmul!(y::AbstractVector, D::BlockJacobiPrec, x::AbstractVector)
  Threads.@threads :static for k in eachindex(D.blocks)
    b = D.blocks[k]
    M = D.blk[k]
    for (i, ri) in enumerate(b)
      s = 0.0
      for (j, cj) in enumerate(b)
        s += M[i, j] * x[cj]
      end
      y[ri] = s
    end
  end
  y
end

function LinearAlgebra.ldiv!(y::AbstractVector, D::BlockJacobiPrec, x::AbstractVector)
  Threads.@threads :static for k in eachindex(D.blocks)
    b = D.blocks[k]
    M = D.inv[k]
    for (i, ri) in enumerate(b)
      s = 0.0
      for (j, cj) in enumerate(b)
        s += M[i, j] * x[cj]
      end
      y[ri] = s
    end
  end
  y
end

# ------------------------------------------------------------------------------------------------
# Chebyshev smoother of one level

mutable struct ChebyshevLevel
  A::SparseMatrixCSC{Float64,Int}            # the level operator (Galerkin product, or the fine tangent)
  D::BlockJacobiPrec
  degree::Int
  λmin::Float64                              # smallest Ritz value of D⁻¹A (negative: indefinite)
  λmax::Float64                              # largest Ritz value
  lo::Float64                                # interval [lo, hi] of the polynomial
  hi::Float64
  ok::Bool
  r::Vector{Float64}                         # work: residual
  z::Vector{Float64}                         # work: preconditioned residual
  d::Vector{Float64}                         # work: direction
  start::Vector{Float64}                     # fixed Lanczos start vector
  symmetric::Bool                            # threaded products (CG: symmetric tangents)
end

function ChebyshevLevel(A::SparseMatrixCSC, blocks, degree, symmetric::Bool)
  n = size(A, 1)
  start = [sin(0.37 * i + 0.11) for i in 1:n]             # fixed, no random state: reproducible estimates
  ChebyshevLevel(A, BlockJacobiPrec(blocks), degree, 0.0, 0.0, 0.0, 0.0, false, zeros(n), zeros(n), zeros(n), start, symmetric)
end

"""
Ritz values (min, max) of `D⁻¹A` after `k` Lanczos steps from the fixed start vector, in the `D`-inner product, in which
`D⁻¹A` is self-adjoint when `A` is symmetric: the Ritz values are real and lie inside the spectrum of the pencil `(A, D)`,
so a negative one means an indefinite `A` (an Arnoldi estimate in the Euclidean inner product can return negative Ritz
values for a positive definite pencil when `D⁻¹A` is far from normal — badly shaped cells).
"""
function _ritz_bounds(S::ChebyshevLevel, k::Int)
  n = size(S.A, 1)
  k = min(k, n)
  q = copy(S.start); qp = zeros(n); w = S.z; r = S.r; t = S.d
  _dmul!(t, S.D, q)
  nq = dot(q, t)
  nq > 0 || return (-1.0, 1.0)                           # the blocks are not positive definite: an indefinite tangent
  q ./= sqrt(nq)
  α = zeros(k); β = zeros(k)
  m = k
  for j in 1:k
    _mul!(r, S.A, q, S.symmetric)                         # r = A q
    ldiv!(w, S.D, r)                                      # w = D⁻¹ A q
    α[j] = dot(q, r)                                      # ⟨w, q⟩_D = qᵀ A q
    w .-= α[j] .* q
    j > 1 && (w .-= β[j-1] .* qp)
    _dmul!(t, S.D, w)
    b2 = dot(w, t)
    b2 ≥ -1e-12 * max(1.0, abs(α[j])) || return (-1.0, 1.0)    # a negative D-norm: the blocks are indefinite
    b = sqrt(max(b2, 0.0))                                # ‖w‖_D
    if j == k || b ≤ 1e-14 * max(1.0, abs(α[j]))
      m = j; break
    end
    β[j] = b
    copyto!(qp, q); q .= w ./ b
  end
  all(isfinite, view(α, 1:m)) && all(isfinite, view(β, 1:m-1)) || return (NaN, NaN)
  ev = eigvals(SymTridiagonal(α[1:m], β[1:m-1]))
  (minimum(ev), maximum(ev))
end

"Updates blocks and interval of a level; returns `:none` or the reason it is not usable."
function _update_level!(S::ChebyshevLevel, range::Float64, inflate::Float64, eig_iters::Int)
  if !all(isfinite, nonzeros(S.A))
    S.ok = false; S.λmin = S.λmax = NaN
    return :nonfinite
  end
  _update_blocks!(S.D, S.A)
  if !S.D.ok[]
    S.ok = false; S.λmin = S.λmax = NaN
    return :blocks
  end
  S.λmin, S.λmax = _ritz_bounds(S, eig_iters)
  S.ok = isfinite(S.λmax) && S.λmax > 0 && S.λmin > 0
  S.hi = inflate * S.λmax
  S.lo = S.hi / range
  S.ok ? :none : :indefinite
end

"""
Chebyshev iteration (Saad, Iterative Methods, alg. 12.1) of `S.degree` steps on `A x = b` from `x` (or from zero with
`zero_guess`), preconditioned by the nodal blocks, on the interval `[lo, hi]`. With `residual` the final residual
`b − A x` is written there (one more product).
"""
function smooth!(x::AbstractVector, S::ChebyshevLevel, b::AbstractVector; zero_guess::Bool=false, residual=nothing)
  A, D, r, z, d = S.A, S.D, S.r, S.z, S.d
  θ = (S.hi + S.lo) / 2
  δ = (S.hi - S.lo) / 2
  σ1 = θ / δ
  ρ = 1 / σ1
  if zero_guess
    fill!(x, 0.0)
    copyto!(r, b)
  else
    _mul!(r, A, x, S.symmetric); r .= b .- r
  end
  ldiv!(z, D, r)
  d .= z ./ θ
  x .+= d
  for k in 2:S.degree
    _mul!(r, A, x, S.symmetric); r .= b .- r
    ldiv!(z, D, r)
    ρk = 1 / (2σ1 - ρ)
    d .= (ρk * ρ) .* d .+ (2ρk / δ) .* z
    x .+= d
    ρ = ρk
  end
  if residual !== nothing
    _mul!(residual, A, x, S.symmetric); residual .= b .- residual
  end
  x
end

# ------------------------------------------------------------------------------------------------
# The V-cycle as a preconditioner, and the numerical setup

mutable struct HMultigridNS{S<:HMultigridSolver,TA<:AbstractMatrix,C} <: Algebra.NumericalSetup
  const solver::S
  A::TA                                      # the fine matrix of the last setup (the Newton cache's)
  levels::Vector{ChebyshevLevel}             # levels 2 … L (smoothed), index l − 1
  Acoarse::SparseMatrixCSC{Float64,Int}
  coarse_ns::C
  bc::Vector{Vector{Float64}}                # work per smoothed level: restricted right-hand side (size of level l − 1)
  xc::Vector{Vector{Float64}}                # and the coarser solution
  e::Vector{Vector{Float64}}                 # prolongated correction (size of level l)
  res::Vector{Vector{Float64}}               # residual after pre-smoothing (size of level l)
  w::Vector{Float64}                         # in-place application copy
  ok::Bool
  coarse_ok::Bool                            # the lagged coarse levels (Galerkin operators, smoothers, coarse setup) are usable
  failure::Symbol                            # :none, or why the last setup is not usable: :nonfinite, :blocks, :indefinite, :coarse
  nsetup::Int
  nbuild::Int
  iters::Int
  rnorm::Float64
  converged::Bool
  rtol::Float64
end

struct HMultigridSS{S<:HMultigridSolver} <: Algebra.SymbolicSetup
  solver::S
end
Algebra.symbolic_setup(s::HMultigridSolver, ::AbstractMatrix) = HMultigridSS(s)

_sparse_fine(A::SparseMatrixCSC{Float64,Int}) = A
_sparse_fine(A::AbstractMatrix) = SparseMatrixCSC{Float64,Int}(sparse(A))

"""
Galerkin products down the hierarchy from the fine matrix; writes the coarse level operators. The fine level always
aliases the Newton's current matrix (its smoother is refreshed at every setup, see `numerical_setup!`); with `reuse > 1`
only the coarse operators and their smoothers lag — the design of S4-A's `coarse_every` (a fully lagged preconditioner,
fine smoother included, made CG stall at 41 iterations instead of 17 on the 800k cube; a fine smoother with a stale
interval on the current matrix made it fail).
"""
function _galerkin!(ns::HMultigridNS, A::SparseMatrixCSC)
  s = ns.solver
  L = length(s.spaces)
  ns.levels[L-1].A = A
  for l in L-1:-1:1
    Al = l == L - 1 ? A : ns.levels[l].A
    Ac = s.Pts[l] * (Al * s.Ps[l])
    if l == 1
      ns.Acoarse = Ac
    else
      ns.levels[l-1].A = Ac
    end
  end
  ns
end

"Refreshes the smoothers (`all` levels, or the fine one only) and the coarse setup."
function _build!(ns::HMultigridNS; all_levels::Bool=true)
  s = ns.solver
  ok = true
  ns.failure = :none
  for (i, S) in enumerate(ns.levels)
    if all_levels || i == length(ns.levels)
      why = _update_level!(S, s.range, s.inflate, s.eig_iters)
      (why !== :none && ns.failure === :none) && (ns.failure = why)
    end
    ok &= S.ok
  end
  if !all_levels
    ns.ok = ok && ns.coarse_ok
    ok && !ns.coarse_ok && (ns.failure = :coarse)
    return ns
  end
  if ok && all(isfinite, nonzeros(ns.Acoarse))
    if ns.coarse_ns === nothing
      ns.coarse_ns = numerical_setup(symbolic_setup(s.coarse, ns.Acoarse), ns.Acoarse)
    else
      numerical_setup!(ns.coarse_ns, ns.Acoarse)
    end
  else
    ok && (ns.failure = :coarse)
    ok = false
  end
  ns.coarse_ok = ok
  ns.ok = ok
  ns.nbuild += 1
  ns
end

function numerical_setup(ss::HMultigridSS, A::AbstractMatrix)
  s = ss.solver
  L = length(s.spaces)
  size(A, 1) == num_free_dofs(s.spaces[end]) ||
    error("HyperFEM: HMultigridSolver: the matrix has $(size(A, 1)) rows but the finest space has $(num_free_dofs(s.spaces[end])) free dofs")
  Af = _sparse_fine(A)
  levels = [ChebyshevLevel(spzeros(num_free_dofs(s.spaces[l]), num_free_dofs(s.spaces[l])), nodal_blocks(s.spaces[l]), s.degree, s.krylov === :cg) for l in 2:L]
  sizes = [num_free_dofs(sp) for sp in s.spaces]
  ns = HMultigridNS{typeof(s),typeof(A),Any}(s, A, levels, spzeros(sizes[1], sizes[1]), nothing,
                                             [zeros(sizes[l-1]) for l in 2:L], [zeros(sizes[l-1]) for l in 2:L],
                                             [zeros(sizes[l]) for l in 2:L], [zeros(sizes[l]) for l in 2:L], zeros(sizes[L]),
                                             false, false, :none, 1, 0, 0, 0.0, true, s.rtol)
  _galerkin!(ns, Af)
  _build!(ns)
  ns
end

function numerical_setup!(ns::HMultigridNS, A::AbstractMatrix)
  s = ns.solver
  ns.nsetup += 1
  # a new matrix object always rebuilds; so does the setup after a failed one (a lagged failure must not be reused —
  # the cut-back of the Newton would fail again on the stale levels, S4-A's lesson)
  rebuild = A !== ns.A || (ns.nsetup - 1) % s.reuse == 0 || !ns.coarse_ok
  ns.A = A
  Af = _sparse_fine(A)
  if rebuild
    _galerkin!(ns, Af)
    _build!(ns)
  else                                        # lagged coarse levels; the fine smoother follows the current tangent
    ns.levels[end].A = Af
    _build!(ns; all_levels=false)
  end
  ns
end

"One V-cycle on level `l` (2 … L): x ← M⁻¹ b, x overwritten."
function _vcycle!(x::AbstractVector, ns::HMultigridNS, l::Int, b::AbstractVector)
  s = ns.solver
  S = ns.levels[l-1]
  bc, xc, e, res = ns.bc[l-1], ns.xc[l-1], ns.e[l-1], ns.res[l-1]
  smooth!(x, S, b; zero_guess=true, residual=res)
  _tmul!(bc, s.Ps[l-1], res)                              # restriction Pᵀ res, threaded over the coarse dofs
  if l == 2
    solve!(xc, ns.coarse_ns, bc)
  else
    _vcycle!(xc, ns, l - 1, bc)
  end
  _tmul!(e, s.Pts[l-1], xc)                               # prolongation P xc = (Pᵀ)ᵀ xc, threaded over the fine dofs
  x .+= e
  smooth!(x, S, b)
  x
end

"The V-cycle as the preconditioner object IterativeSolvers applies."
struct HMultigridPrec{NS<:HMultigridNS}
  ns::NS
end
LinearAlgebra.ldiv!(y::AbstractVector, P::HMultigridPrec, x::AbstractVector) = _vcycle!(y, P.ns, length(P.ns.solver.spaces), x)
LinearAlgebra.ldiv!(P::HMultigridPrec, x::AbstractVector) = (copyto!(P.ns.w, x); ldiv!(x, P, P.ns.w))

function Algebra.solve!(x::AbstractVector, ns::HMultigridNS, b::AbstractVector)
  s = ns.solver
  fill!(x, zero(eltype(x)))
  if !ns.ok                                   # non-finite tangent, singular block or indefinite level: a failed solve
    ns.iters, ns.converged, ns.rnorm = 0, false, NaN
    return x
  end
  P = HMultigridPrec(ns)
  A = LevelOperator(ns.levels[end].A, s.krylov === :cg)   # the current tangent (the fine level aliases it)
  if s.krylov === :cg
    _, h = IterativeSolvers.cg!(x, A, b; Pl=P, reltol=ns.rtol, abstol=s.atol, maxiter=s.maxiter, initially_zero=true, log=true)
  else
    _, h = IterativeSolvers.gmres!(x, A, b; Pr=P, reltol=ns.rtol, abstol=s.atol, maxiter=s.maxiter, restart=min(s.restart, length(b)),
                                   initially_zero=true, log=true)
  end
  ns.iters = h.iters
  ns.converged = h.isconverged
  r = h[:resnorm]
  ns.rnorm = isempty(r) ? 0.0 : r[end]
  x
end

linear_converged(ns::HMultigridNS) = ns.converged
linear_iterations(ns::HMultigridNS) = ns.iters
preconditioner_builds(ns::HMultigridNS) = ns.nbuild
linear_info(ns::HMultigridNS) = ns.ok ? " $(ns.iters)" : " failed ($(ns.failure))"
set_tolerance!(ns::HMultigridNS, rtol) = (ns.rtol = rtol === nothing ? ns.solver.rtol : _check_rtol(rtol); ns)
tolerance(ns::HMultigridNS) = ns.rtol

"Smallest and largest Ritz values of the block-preconditioned operator of every smoothed level (coarse → fine)."
level_spectra(ns::HMultigridNS) = [(S.λmin, S.λmax) for S in ns.levels]
