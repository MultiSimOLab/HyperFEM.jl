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

Linear solver of the assembled tangent through a PETSc `KSP` (needs `using GridapPETSc`, which
activates the extension). The PETSc binary of GridapPETSc provides everything the solver uses.

- `ksp`: Krylov method, `:cg` (symmetric positive definite tangents, e.g. the projected tangent of
  [`HybridTangent`](@ref)), `:gmres` (general, the consistent tangent near instabilities) or
  `:preonly` (apply the preconditioner once: direct solvers).
- `pc`: preconditioner, `:gamg` (PETSc's smoothed-aggregation AMG), `:boomeramg` (hypre),
  `:mumps` (LU of MUMPS, the direct reference), `:mumps_cholesky`, `:lu` (PETSc's own LU),
  `:jacobi`, `:none`.
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
  function PETScSolver(ksp::Symbol=:cg, pc::Symbol=:gamg; rtol::Real=1e-6, atol::Real=0.0, maxiter::Integer=500,
                       reuse::Integer=1, nullspace::Union{Nothing,AbstractMatrix}=nothing,
                       negate::Union{Nothing,AbstractUnitRange}=nothing, blocksize::Integer=1, options::AbstractString="",
                       monitor::Bool=false)
    ksp in (:cg, :gmres, :preonly) || error("HyperFEM: PETScSolver: unknown Krylov method $ksp (use :cg, :gmres or :preonly)")
    pc in (:gamg, :boomeramg, :mumps, :mumps_cholesky, :lu, :jacobi, :none) ||
      error("HyperFEM: PETScSolver: unknown preconditioner $pc (use :gamg, :boomeramg, :mumps, :mumps_cholesky, :lu, :jacobi or :none)")
    reuse ≥ 1 || error("HyperFEM: PETScSolver: reuse must be ≥ 1")
    blocksize ≥ 1 || error("HyperFEM: PETScSolver: blocksize must be ≥ 1")
    ns = nullspace === nothing ? nothing : Matrix{Float64}(nullspace)
    neg = negate === nothing ? nothing : UnitRange{Int}(negate)
    new(ksp, pc, rtol, atol, maxiter, reuse, ns, neg, Int(blocksize), String(options), monitor)
  end
end

Algebra.symbolic_setup(s::PETScSolver, A::AbstractMatrix) = petsc_symbolic_setup(s, A)
petsc_symbolic_setup(::PETScSolver, A) =
  error("HyperFEM: `PETScSolver` needs GridapPETSc: add `using GridapPETSc` (it activates the extension HyperFEMPETScExt)")

const _PC_NAMES = Dict(:gamg => "GAMG", :boomeramg => "BoomerAMG", :mumps => "MUMPS", :mumps_cholesky => "MUMPS-Chol",
                       :lu => "PETSc-LU", :jacobi => "Jacobi", :none => "none")
const _KSP_NAMES = Dict(:cg => "CG", :gmres => "GMRES", :preonly => "")

function _linear_name(s::PETScSolver)
  k, p = _KSP_NAMES[s.ksp], _PC_NAMES[s.pc]
  isempty(k) ? p : k * "+" * p
end

# ------------------------------------------------------------------------------------------------
# Pure-Julia Krylov + AMG (prototyping and CI, D1)

"""
    AMGSolver(krylov=:cg; rtol=1e-6, atol=0.0, maxiter=500, restart=30, reuse=1, nullspace=nothing)

Pure-Julia counterpart of [`PETScSolver`](@ref) (needs `import AlgebraicMultigrid`, which activates the
extension; imported rather than used because the package exports its own `solve!`): IterativeSolvers' `cg` or `gmres` preconditioned by one V-cycle of AlgebraicMultigrid's
smoothed aggregation, built with the near-null space `nullspace` (free dofs × modes, see
[`near_nullspace`](@ref)). `reuse` rebuilds the hierarchy every `reuse` numerical setups (lagged
AMG); `atol = 0` for the same reason as in [`PETScSolver`](@ref). Serial; it is the variant used in CI, where installing PETSc is slow, and the reference for
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
  function AMGSolver(krylov::Symbol=:cg; rtol::Real=1e-6, atol::Real=0.0, maxiter::Integer=500, restart::Integer=30,
                     reuse::Integer=1, nullspace::Union{Nothing,AbstractMatrix}=nothing)
    krylov in (:cg, :gmres) || error("HyperFEM: AMGSolver: unknown Krylov method $krylov (use :cg or :gmres)")
    reuse ≥ 1 || error("HyperFEM: AMGSolver: reuse must be ≥ 1")
    ns = nullspace === nothing ? nothing : Matrix{Float64}(nullspace)
    new(krylov, rtol, atol, maxiter, restart, reuse, ns)
  end
end

Algebra.symbolic_setup(s::AMGSolver, A::AbstractMatrix) = amg_symbolic_setup(s, A)
amg_symbolic_setup(::AMGSolver, A) =
  error("HyperFEM: `AMGSolver` needs AlgebraicMultigrid: add `import AlgebraicMultigrid` (it activates the extension HyperFEMAMGExt)")

_linear_name(s::AMGSolver) = _KSP_NAMES[s.krylov] * "+AMG"

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
