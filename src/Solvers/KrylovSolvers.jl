"""
Krylov + AMG linear solvers on the assembled tangent (design/solvers.md, stage S1).

The two solver types below hold options only (pure Julia). Their Gridap `LinearSolver` methods live
in package extensions: [`PETScSolver`](@ref) in `HyperFEMPETScExt` (loaded with GridapPETSc) and
[`AMGSolver`](@ref) in `HyperFEMAMGExt` (loaded with AlgebraicMultigrid). Both read the near-null
space of the tangent built by [`near_nullspace`](@ref) from the FE space, which is what makes
algebraic multigrid work on elasticity.
"""

# ------------------------------------------------------------------------------------------------
# PETSc Krylov + preconditioner (production, D1)

"""
    PETScSolver(ksp=:cg, pc=:gamg; rtol=1e-6, atol=0.0, maxiter=500, reuse=1, nullspace=nothing,
                negate=nothing, blocksize=1, options="", monitor=false)
    PETScSolver(ksp, :fieldsplit; fields, blocks=fill(:gamg, length(fields)), split=:schur,
                schur=(fact=:upper, precondition=:a11), fieldnames=["f1", …], kwargs...)
    PETScSolver(U::MultiFieldFESpace; ksp=:gmres, blocks, split, schur, negate=:last, kwargs...)

Linear solver of the assembled tangent through a PETSc `KSP` (needs `using GridapPETSc`, which
activates the extension). The PETSc binary of GridapPETSc provides everything the solver uses.

- `ksp`: Krylov method, `:cg` (symmetric positive definite tangents, e.g. the projected tangent of
  [`HybridTangent`](@ref)), `:gmres` (general, the consistent tangent near instabilities) or
  `:preonly` (apply the preconditioner once: direct solvers).
- `pc`: preconditioner, `:gamg` (PETSc's smoothed-aggregation AMG), `:boomeramg` (hypre),
  `:mumps` (LU of MUMPS, the direct reference), `:mumps_cholesky`, `:lu` (PETSc's own LU),
  `:jacobi`, `:none`, or `:fieldsplit` (block preconditioner by fields, stage S2, below).
- `rtol`, `atol`, `maxiter`: Krylov tolerances (relative to the right-hand side, as PETSc's
  `KSPSetTolerances` with `-ksp_norm_type unpreconditioned`) and maximum number of iterations. Keep
  `atol = 0` inside a Newton: the right-hand side is the residual, which becomes tiny near convergence,
  and an absolute tolerance would accept an unsolved increment there.
- `reuse`: the preconditioner is rebuilt every `reuse` numerical setups (lagged AMG: `reuse = 3`
  keeps the hierarchy for three Newton iterations; `1` rebuilds it at every tangent). With a direct
  solver (`:preonly`) a lagged factorisation is a modified Newton: the increments solve an old tangent
  and the quadratic rate is lost, with no failure reported. A new matrix object always rebuilds.
- `nullspace`: near-null space of the tangent (free dofs × modes, from [`near_nullspace`](@ref)),
  passed to the matrix with `MatSetNearNullSpace`; the rigid-body modes are essential for AMG on
  elasticity. GAMG reads it by default; BoomerAMG only with its interpolation-vector options
  (`-pc_hypre_boomeramg_interp_vec_variant`). `nothing` leaves PETSc's default (the constant vector).
- `negate`: range of equations multiplied by −1 (tangent rows and right-hand side) before PETSc sees
  the system; the solution is unchanged. A symmetric saddle-point tangent, concave in the electric
  or magnetic potential (the monolithic EM system: `negate = nu+1:n` for the φ block), becomes a
  positive-real matrix on which GMRES + AMG converge, while AMG on the indefinite form fails.
- `blocksize`: block size of the matrix (`MatSetBlockSize`), e.g. 3 for a single vector field in 3D whose
  free dofs come in complete nodal blocks (no component-wise Dirichlet masks): GAMG then aggregates
  nodes instead of scalar dofs, which is what it expects on elasticity. `1` (default) sets nothing.
- `options`: extra PETSc options for this solver, e.g. `"-pc_gamg_threshold 0.01 -ksp_gmres_restart 50"`
  (the prefix of the solver is added automatically).
- `monitor`: print PETSc's residual history (`-ksp_monitor`).

**Block preconditioners (`pc = :fieldsplit`, PETSc's `PCFIELDSPLIT`).** `fields` are the dof ranges of
the fields ([`field_ranges`](@ref) of a consecutive multi-field space); `blocks` the preconditioner of
each diagonal block (`:gamg`, `:boomeramg`, `:mumps`, `:lu`, `:jacobi`), applied once per outer
iteration (`preonly`); `split` is `:additive` (block Jacobi), `:multiplicative` (block Gauss–Seidel),
`:symmetric_multiplicative` or `:schur` (two fields: block triangular with the Schur complement, `schur
= (fact = :upper | :lower | :full | :diag, precondition = :a11 | :selfp)`); `fieldnames` give the option
prefixes `fieldsplit_<name>_…`. The near-null space of the whole space is split by field (a mode of u
goes to the u block, the constant to φ). On the symmetric saddle point of the monolithic
electro-mechanical tangent (K_φφ negative definite) negate the φ equations: with `negate = range_φ`
both diagonal blocks are SPD and the Schur complement too, and GMRES converges with any of the four
splits; without it the AMG of the φ block sees a negative definite matrix. The constructor from the
space, `PETScSolver(U; …)`, fills `fields`, `nullspace = near_nullspace(U)` and `negate` (`:last`, the
last field, the potential by HyperFEM's convention; a range or `nothing` otherwise):
`NewtonRaphson(PETScSolver(U))` is the production path of the multi-field drivers.

Three configurations of the plan: `PETScSolver(:cg, :gamg; nullspace)`, `PETScSolver(:gmres, :boomeramg;
nullspace)` and `PETScSolver(:preonly, :mumps)`. The solver reports [`linear_converged`](@ref),
[`linear_iterations`](@ref) and [`linear_info`](@ref) to the Newton: a Krylov solve that does not
converge fails the Newton iteration like a failed line search.

The matrix of the Newton cache (CSC) is mirrored once into a CSR matrix that PETSc aliases without
copying; at every numerical setup only the values are gathered (threaded), so the conversion costs
one pass over the nonzeros and allocates nothing after the first setup. Works with any assembling
backend (`:lazy`, `:compiled`).
"""
struct PETScSolver <: Algebra.LinearSolver
  ksp::Symbol
  pc::Symbol
  rtol::Float64
  atol::Float64
  maxiter::Int
  reuse::Int
  nullspace::Union{Nothing,Matrix{Float64}}
  negate::Union{Nothing,UnitRange{Int}}
  blocksize::Int
  options::String
  monitor::Bool
  fields::Vector{UnitRange{Int}}             # fieldsplit: dof range of every field (empty otherwise)
  fieldnames::Vector{String}
  blocks::Vector{Symbol}
  split::Symbol
  schur::NamedTuple{(:fact, :precondition),Tuple{Symbol,Symbol}}
  function PETScSolver(ksp::Symbol=:cg, pc::Symbol=:gamg; rtol::Real=1e-6, atol::Real=0.0, maxiter::Integer=500,
                       reuse::Integer=1, nullspace::Union{Nothing,AbstractMatrix}=nothing,
                       negate::Union{Nothing,AbstractUnitRange}=nothing, blocksize::Integer=1, options::AbstractString="",
                       monitor::Bool=false, fields::AbstractVector{<:AbstractUnitRange}=UnitRange{Int}[],
                       fieldnames::AbstractVector{<:AbstractString}=["f$i" for i in 1:length(fields)],
                       blocks::AbstractVector{Symbol}=fill(:gamg, length(fields)), split::Symbol=:schur,
                       schur::NamedTuple=(fact=:upper, precondition=:a11))
    ksp in (:cg, :gmres, :fgmres, :preonly) || error("HyperFEM: PETScSolver: unknown Krylov method $ksp (use :cg, :gmres, :fgmres or :preonly)")
    pc in (:gamg, :boomeramg, :mumps, :mumps_cholesky, :lu, :jacobi, :none, :fieldsplit) ||
      error("HyperFEM: PETScSolver: unknown preconditioner $pc (use :gamg, :boomeramg, :mumps, :mumps_cholesky, :lu, :jacobi, :none or :fieldsplit)")
    reuse ≥ 1 || error("HyperFEM: PETScSolver: reuse must be ≥ 1")
    blocksize ≥ 1 || error("HyperFEM: PETScSolver: blocksize must be ≥ 1")
    fr = [UnitRange{Int}(r) for r in fields]
    if pc === :fieldsplit
      length(fr) ≥ 2 || error("HyperFEM: PETScSolver: :fieldsplit needs the dof ranges of at least two fields (`fields = field_ranges(U)`)")
      length(fieldnames) == length(fr) == length(blocks) || error("HyperFEM: PETScSolver: fields, fieldnames and blocks must have the same length")
      all(b -> b in (:gamg, :boomeramg, :mumps, :lu, :jacobi, :none), blocks) ||
        error("HyperFEM: PETScSolver: block preconditioners are :gamg, :boomeramg, :mumps, :lu, :jacobi or :none")
      split in (:additive, :multiplicative, :symmetric_multiplicative, :schur) ||
        error("HyperFEM: PETScSolver: split must be :additive, :multiplicative, :symmetric_multiplicative or :schur")
      split === :schur && length(fr) != 2 && error("HyperFEM: PETScSolver: the Schur split needs exactly two fields")
      schur.fact in (:upper, :lower, :full, :diag) || error("HyperFEM: PETScSolver: schur.fact must be :upper, :lower, :full or :diag")
      schur.precondition in (:a11, :selfp) || error("HyperFEM: PETScSolver: schur.precondition must be :a11 or :selfp")
      for i in 2:length(fr)
        first(fr[i]) == last(fr[i-1]) + 1 || error("HyperFEM: PETScSolver: the field ranges must be consecutive")
      end
    elseif !isempty(fr)
      error("HyperFEM: PETScSolver: `fields` is only used with pc = :fieldsplit")
    end
    ns = nullspace === nothing ? nothing : Matrix{Float64}(nullspace)
    neg = negate === nothing ? nothing : UnitRange{Int}(negate)
    new(ksp, pc, rtol, atol, maxiter, reuse, ns, neg, Int(blocksize), String(options), monitor,
        fr, String.(fieldnames), Symbol.(blocks), split, (fact=Symbol(schur.fact), precondition=Symbol(schur.precondition)))
  end
end

"""
    PETScSolver(U::MultiFieldFESpace; ksp=:gmres, blocks=fill(:gamg, num_fields(U)), split=:schur,
                schur=(fact=:upper, precondition=:a11), negate=:last, fieldnames, kwargs...)

Block preconditioner (`pc = :fieldsplit`) for the tangent assembled on the consecutive multi-field
space `U`: `fields = field_ranges(U)`, `nullspace = near_nullspace(U)` (the rigid modes of the vector
fields, the constants of the scalar ones, split by block), and `negate = :last` (the equations of
the last field, the potential of the electro- and magneto-mechanical drivers, so that the saddle point
becomes positive-real; a range or `nothing` otherwise). `fieldnames` default to `"u"` for a vector
field and `"phi"` for a scalar one when there is one of each, `"f1", "f2", …` otherwise.
"""
function PETScSolver(U::MultiFieldFESpace; ksp::Symbol=:gmres, blocks=fill(:gamg, num_fields(U)), split::Symbol=:schur,
                     schur::NamedTuple=(fact=:upper, precondition=:a11), negate=:last, fieldnames=_default_fieldnames(U), kwargs...)
  fields = field_ranges(U)
  neg = negate === :last ? fields[end] : negate
  PETScSolver(ksp, :fieldsplit; fields, blocks, split, schur, nullspace=near_nullspace(U), negate=neg, fieldnames, kwargs...)
end

function _default_fieldnames(U::MultiFieldFESpace)
  nc = [_num_components(U[i]) for i in 1:num_fields(U)]
  num_fields(U) == 2 && nc[1] > 1 && nc[2] == 1 ? ["u", "phi"] : ["f$i" for i in 1:num_fields(U)]
end

"""
    field_ranges(U::MultiFieldFESpace) -> Vector{UnitRange{Int}}

Free-dof range of every field of a consecutive multi-field space (`ConsecutiveMultiFieldStyle`, the
layout of HyperFEM's multi-field drivers): the blocks of the assembled tangent. Block-style spaces
(`BlockMultiFieldStyle`, where the matrix is a `BlockMatrix`) are not supported.
"""
function field_ranges(U::MultiFieldFESpace)
  MultiFieldStyle(U) isa ConsecutiveMultiFieldStyle ||
    error("HyperFEM: field_ranges needs a consecutive multi-field space (ConsecutiveMultiFieldStyle)")
  n = [num_free_dofs(U[i]) for i in 1:num_fields(U)]
  offsets = cumsum([0; n])
  [offsets[i]+1:offsets[i+1] for i in 1:length(n)]
end

"""
    split_nullspace(N, fields) -> Vector{Matrix{Float64}}

The near-null space `N` (dofs × modes) split by field: the block of every field with the modes that
live in it (a mode belongs to the field where it is nonzero; the modes of [`near_nullspace`](@ref) are
block-structured). Fields without modes get a `n_i × 0` matrix.
"""
function split_nullspace(N::Matrix{Float64}, fields::Vector{UnitRange{Int}})
  [begin
    cols = [j for j in 1:size(N, 2) if any(!iszero, view(N, r, j))]
    N[r, cols]
  end for r in fields]
end

Algebra.symbolic_setup(s::PETScSolver, A::AbstractMatrix) = petsc_symbolic_setup(s, A)
petsc_symbolic_setup(::PETScSolver, A) =
  error("HyperFEM: `PETScSolver` needs GridapPETSc: add `using GridapPETSc` (it activates the extension HyperFEMPETScExt)")

const _PC_NAMES = Dict(:gamg => "GAMG", :boomeramg => "BoomerAMG", :mumps => "MUMPS", :mumps_cholesky => "MUMPS-Chol",
                       :lu => "PETSc-LU", :jacobi => "Jacobi", :none => "none", :fieldsplit => "FS")
const _KSP_NAMES = Dict(:cg => "CG", :gmres => "GMRES", :fgmres => "FGMRES", :preonly => "")
const _SPLIT_NAMES = Dict(:additive => "add", :multiplicative => "mult", :symmetric_multiplicative => "smult", :schur => "schur")

function _linear_name(s::PETScSolver)
  k, p = _KSP_NAMES[s.ksp], _PC_NAMES[s.pc]
  s.pc === :fieldsplit && (p = "FS-" * _SPLIT_NAMES[s.split] * "(" * join((_PC_NAMES[b] for b in s.blocks), ",") * ")")
  isempty(k) ? p : k * "+" * p
end

# ------------------------------------------------------------------------------------------------
# Pure-Julia Krylov + AMG (prototyping and CI, D1)

"""
    AMGSolver(krylov=:cg; rtol=1e-6, atol=0.0, maxiter=500, restart=30, reuse=1, nullspace=nothing,
              fields=UnitRange{Int}[], split=:schur, negate=nothing)
    AMGSolver(U::MultiFieldFESpace; krylov=:gmres, split=:schur, negate=:last, kwargs...)

Pure-Julia counterpart of [`PETScSolver`](@ref) (needs `import AlgebraicMultigrid`, which activates the
extension; imported rather than used because the package exports its own `solve!`): IterativeSolvers' `cg` or `gmres` preconditioned by one V-cycle of AlgebraicMultigrid's
smoothed aggregation, built with the near-null space `nullspace` (free dofs × modes, see
[`near_nullspace`](@ref)). `reuse` rebuilds the hierarchy every `reuse` numerical setups (lagged
AMG); `atol = 0` for the same reason as in [`PETScSolver`](@ref). With `fields` (the dof ranges of
[`field_ranges`](@ref)) the preconditioner is a block one with one hierarchy per field: `split =
:additive` (block Jacobi), `:multiplicative` (block Gauss–Seidel, lower triangular) or `:schur` (two
fields, upper block triangular with the Schur complement approximated by its diagonal block, PETSc's
`upper, a11`); `negate` as in `PETScSolver` (the φ equations of a saddle point). The constructor from
the space fills `fields`, `nullspace` and `negate = :last`. Block preconditioners need `:gmres`. Serial; it is the variant used in CI, where installing PETSc is slow, and the reference for
the Newton integration (`linear_converged`, lagged setups).
"""
struct AMGSolver <: Algebra.LinearSolver
  krylov::Symbol
  rtol::Float64
  atol::Float64
  maxiter::Int
  restart::Int
  reuse::Int
  nullspace::Union{Nothing,Matrix{Float64}}
  fields::Vector{UnitRange{Int}}
  split::Symbol
  negate::Union{Nothing,UnitRange{Int}}
  function AMGSolver(krylov::Symbol=:cg; rtol::Real=1e-6, atol::Real=0.0, maxiter::Integer=500, restart::Integer=30,
                     reuse::Integer=1, nullspace::Union{Nothing,AbstractMatrix}=nothing,
                     fields::AbstractVector{<:AbstractUnitRange}=UnitRange{Int}[], split::Symbol=:schur,
                     negate::Union{Nothing,AbstractUnitRange}=nothing)
    krylov in (:cg, :gmres) || error("HyperFEM: AMGSolver: unknown Krylov method $krylov (use :cg or :gmres)")
    reuse ≥ 1 || error("HyperFEM: AMGSolver: reuse must be ≥ 1")
    fr = [UnitRange{Int}(r) for r in fields]
    if !isempty(fr)
      length(fr) ≥ 2 || error("HyperFEM: AMGSolver: a block preconditioner needs at least two fields")
      split in (:additive, :multiplicative, :schur) || error("HyperFEM: AMGSolver: split must be :additive, :multiplicative or :schur")
      split === :schur && length(fr) != 2 && error("HyperFEM: AMGSolver: the Schur split needs exactly two fields")
      krylov === :gmres || error("HyperFEM: AMGSolver: block preconditioners are not symmetric: use krylov = :gmres")
      for i in 2:length(fr)
        first(fr[i]) == last(fr[i-1]) + 1 || error("HyperFEM: AMGSolver: the field ranges must be consecutive")
      end
    end
    ns = nullspace === nothing ? nothing : Matrix{Float64}(nullspace)
    neg = negate === nothing ? nothing : UnitRange{Int}(negate)
    new(krylov, rtol, atol, maxiter, restart, reuse, ns, fr, split, neg)
  end
end

function AMGSolver(U::MultiFieldFESpace; krylov::Symbol=:gmres, split::Symbol=:schur, negate=:last, kwargs...)
  fields = field_ranges(U)
  neg = negate === :last ? fields[end] : negate
  AMGSolver(krylov; fields, split, nullspace=near_nullspace(U), negate=neg, kwargs...)
end

Algebra.symbolic_setup(s::AMGSolver, A::AbstractMatrix) = amg_symbolic_setup(s, A)
amg_symbolic_setup(::AMGSolver, A) =
  error("HyperFEM: `AMGSolver` needs AlgebraicMultigrid: add `import AlgebraicMultigrid` (it activates the extension HyperFEMAMGExt)")

_linear_name(s::AMGSolver) = _KSP_NAMES[s.krylov] * (isempty(s.fields) ? "+AMG" : "+FS-" * _SPLIT_NAMES[s.split] * "(AMG)")

"""
    linear_iterations(ns) -> Int

Iterations of the last solve of an iterative linear solver (`0` for direct solvers).
"""
linear_iterations(::Algebra.NumericalSetup) = 0

"""
    preconditioner_builds(ns) -> Int

Number of times the numerical setup `ns` has built its preconditioner (factorisation or AMG
hierarchy); with `reuse > 1` it lags behind the number of setups. `0` when not tracked.
"""
preconditioner_builds(::Algebra.NumericalSetup) = 0

# ------------------------------------------------------------------------------------------------
# Near-null space of the tangent

_value_components(::Type{<:Integer}) = 1
_value_components(::Type{T}) where {T} = length(T)

function _num_components(U::SingleFieldFESpace)
  dofbasis = first(get_data(get_fe_dof_basis(U)))
  dofbasis isa Gridap.ReferenceFEs.LagrangianDofBasis ||
    error("HyperFEM: near_nullspace needs Lagrangian spaces (got $(typeof(dofbasis)))")
  _value_components(eltype(dofbasis.node_and_comp_to_dof))
end

"Rigid-body modes of a vector field with `nc` components in `dim` space dimensions, as functions of x."
function _rigid_modes(nc::Int, dim::Int)
  e(c) = VectorValue(ntuple(i -> i == c ? 1.0 : 0.0, nc))
  modes = Any[x -> e(c) for c in 1:nc]                     # translations
  if nc == dim == 2
    push!(modes, x -> VectorValue(-x[2], x[1]))             # rotation
  elseif nc == dim == 3
    for (a, b) in ((1, 2), (1, 3), (2, 3))                  # rotations in the a–b planes
      push!(modes, x -> x[a] * e(b) - x[b] * e(a))
    end
  end
  modes
end

_modes(U::SingleFieldFESpace) = (nc = _num_components(U); nc == 1 ? Any[x -> 1.0] : _rigid_modes(nc, num_point_dims(get_triangulation(U))))

"Zero of the value type of the space, for the fields a mode does not touch."
_zero_mode(U::SingleFieldFESpace) = (nc = _num_components(U); nc == 1 ? (x -> 0.0) : (x -> zero(VectorValue{nc,Float64})))

"""
    near_nullspace(U::FESpace) -> Matrix{Float64}

Near-null space of a tangent assembled on the trial space `U`, as an orthonormal matrix of free dofs
× modes: the rigid-body modes (translations and rotations, 3 in 2D and 6 in 3D) of every
vector-valued Lagrangian field and the constant mode of every scalar field, interpolated on the
space, so that the Dirichlet dofs are left out and the modes of a multi-field space live in their
own block (a mode of u is zero on φ). Modes that vanish on the free dofs (e.g. a translation fully
constrained by Dirichlet conditions) are dropped; an error if none is left. A vector field with fewer
or more components than the space dimension gets its translations only. Independent of the Dirichlet values, so it is
computed once per space and passed to the solver: `PETScSolver(:cg, :gamg; nullspace = near_nullspace(U))`.
"""
function near_nullspace(U::SingleFieldFESpace)
  cols = [get_free_dof_values(interpolate(f, U)) for f in _modes(U)]
  _orthonormal(reduce(hcat, cols))
end

function near_nullspace(U::MultiFieldFESpace)
  nf = num_fields(U)
  cols = Vector{Float64}[]
  for i in 1:nf
    zeros_i = [_zero_mode(U[j]) for j in 1:nf]
    for f in _modes(U[i])
      fs = copy(zeros_i); fs[i] = f
      push!(cols, get_free_dof_values(interpolate(fs, U)))
    end
  end
  _orthonormal(reduce(hcat, cols))
end

"""
Orthonormal basis of the columns of `B` (thin QR without pivoting, so that the first columns keep spanning the
first modes: the modes of a field stay in its block), dropping the numerically dependent columns.
"""
function _orthonormal(B::Matrix{Float64})
  size(B, 2) == 0 && error("HyperFEM: near_nullspace: the space has no modes")
  while true
    F = qr(B)
    d = abs.(diag(F.R))
    keep = findall(>(1e-10 * max(maximum(d), eps())), d)
    isempty(keep) && error("HyperFEM: near_nullspace: every mode vanishes on the free dofs (fully constrained space)")
    length(keep) == size(B, 2) && return Matrix(F.Q)[:, 1:length(keep)]
    B = B[:, keep]                              # dropping a column can expose another dependency: check again
  end
end
