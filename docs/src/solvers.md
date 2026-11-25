# Load stepping and solvers

## Load stepping

The keyword `stepping` of `solve!` (e.g. `solve!(comp_model; stepping=(nsteps=5, maxbisec=5))`) controls the
load increments: the stage Λ goes from 0 to 1 starting with `ΔΛ = 1/nsteps`; a failed
increment is bisected (`maxbisec` consecutive bisections at most). With `LoadStepping` the increment can also grow
after easy increments: `LoadStepping(nsteps=1000, maxbisec=5, grow=1.5, easy=4, ΔΛmax=0.016, neasy=2)` starts with
ΔΛ = 0.001 and multiplies it by 1.5 after every two consecutive increments converged in at most 4 Newton iterations,
up to ΔΛ = 0.016. By default (`grow = 1`, `ΔΛmax = 1/nsteps`, `neasy = 1`) the increment never exceeds the initial one.
A small initial increment with a larger `ΔΛmax` gets through a hard start and speeds up where the problem allows it;
`neasy = 2` (Abaqus' rule) makes a large `ΔΛmax` safer.

## Solvers for large problems

`NewtonRaphson` accepts any Gridap `LinearSolver`. Beyond ~100k degrees of freedom in 3D the LU factorisation
dominates the time and the memory of every Newton iteration; HyperFEM then provides, with `using GridapPETSc`
(PETSc through GridapPETSc, serial), Krylov solvers preconditioned by algebraic multigrid on the assembled
tangent, and the direct solver MUMPS:

```julia
using GridapPETSc

# Multi-field problems (u, φ): GMRES with a block (fieldsplit) preconditioner, one AMG per field with
# rigid-body modes on the displacement block, rebuilt every 3 Newton iterations (reuse), and an inexact
# Newton (forcing): every linear solve only as tight as the Newton needs. The production configuration for
# electro- and magneto-mechanics.
nls = NewtonRaphson(PETScSolver(U; reuse=3); maxiter=20, rtol=1.e-8, forcing=EisenstatWalker())

# Mechanics (single vector field, P1/Q1): CG + GAMG with the rigid-body modes of the space.
nls = NewtonRaphson(PETScSolver(:cg, :gamg; nullspace=near_nullspace(U)); rtol=1.e-8)

# Direct solver with a fraction of the memory of LU: the reference, and the solver for thin structures
# (few elements across the thickness), where algebraic multigrid does not converge.
nls = NewtonRaphson(PETScSolver(:preonly, :mumps); rtol=1.e-8)
```

Krylov tolerances (`rtol`, `atol = 0`) are relative to the right-hand side; `reuse = n` keeps the
preconditioner for `n` Newton iterations; `forcing = EisenstatWalker()` lets the Newton choose the tolerance of
every linear solve (loose far from the solution, tight near it). Neither is the default, because both change
the Newton trajectory (a few more Newton iterations, far fewer Krylov iterations and preconditioner builds):
on electro- and magneto-mechanical benchmarks of 100k–240k dofs the configuration above solved ×1.9–2.2
faster than `PETScSolver(U)` with exact linear solves. The pure-Julia counterpart `AMGSolver(U)` (with
`using AlgebraicMultigrid`) runs the same block preconditioners without PETSc and is what the test suite
uses. Assembly can be delegated to compiled, threaded kernels with `backend=:compiled` in the drivers
(`using HyperFEMKernels`); with an iterative solver this is what makes the assembly negligible.

With `backend = CompiledBackend(jacobian = :matrixfree)` the nonlinear drivers never assemble the tangent
for the Krylov solver: HyperFEMKernels applies it matrix-free and `MatrixFreeSolver` (GMRES, FGMRES or CG
from Krylov.jl) solves the Newton systems on that operator. The preconditioner is a slot;
`LaggedAssembled(solver; every = k)` assembles and factorises the tangent only every `k` Newton iterations
and preconditions with it in between, so the Newton keeps the exact tangent (same trajectory as a direct
solver) with a fraction of the factorisations:

```julia
using HyperFEMKernels, GridapPETSc
using HyperFEMKernels: MatrixFreeSolver, LaggedAssembled

ls  = MatrixFreeSolver(; krylov = :gmres, rtol = 1e-10,
                       preconditioner = LaggedAssembled(PETScSolver(:preonly, :mumps); every = 3))
nls = NewtonRaphson(ls; maxiter = 20, rtol = 1e-8)
m   = StaticNonlinearModel(res, jac, U, V, D; nls, backend = CompiledBackend(jacobian = :matrixfree))
```

On a thin electro-mechanical Q2 hexahedral benchmark, where algebraic multigrid does not converge and MUMPS
is the reference, this configuration took the same 20 Newton iterations as MUMPS and solved ×2.3 (57k dofs)
and ×5.0 (218k dofs) faster (our measurement: Mac, 6 Julia threads, PETSc serial). On simplices the
matrix-free product is slower than the assembled one; there the assembled configurations above remain the
choice.

### Geometric multigrid on a refined mesh hierarchy

For Q1/P1 models whose geometry a coarse mesh can describe, `MeshHierarchy` refines it uniformly (Gridap's
`refine`, tetrahedra and hexahedra; the Dirichlet tags must be closed sets — face, edges and vertices — so
that every level has the same Dirichlet boundary) and `HMultigridSolver` preconditions CG with one geometric
V-cycle (Galerkin coarse operators, Chebyshev smoothing on the nodal blocks, any linear solver on the coarsest
level). `FieldSplitSolver` applies one V-cycle per block of a multi-field saddle point. Both are pure Julia
(threads); the coarsest level is the only place PETSc enters:

```julia
using GridapPETSc

h  = MeshHierarchy(coarse_model, 4)                      # 3 uniform refinements: the Newton sees fine_model(h)
Vh(m) = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D, conformity=:H1)
Uc = TrialFESpace(Vh(level_model(h, 1)))
# mechanics: CG + one V-cycle, the coarse levels rebuilt every 3 Newton iterations (the fine smoother always
# follows the current tangent), the coarsest level one GAMG cycle with the rigid-body modes
ls  = HMultigridSolver(h, Vh; krylov=:cg, reuse=3,
                       coarse=PETScSolver(:preonly, :gamg; nullspace=near_nullspace(Uc), blocksize=3))
nls = NewtonRaphson(ls; maxiter=20, rtol=1e-8, forcing=EisenstatWalker())

# electro-mechanics (u, φ): GMRES with one V-cycle per block (Schur split), the same recommendation
ls  = FieldSplitSolver(U, (HMultigridSolver(h, Vu; krylov=:cg, reuse=3, coarse=…),
                           HMultigridSolver(h, Vφ; krylov=:cg, reuse=3, coarse=…)); split=:schur)
nls = NewtonRaphson(ls; maxiter=20, rtol=1e-8, forcing=EisenstatWalker())
```

The CG iterations are independent of the mesh (17 on Q1 hexahedra, 32 on P1 tetrahedra, from 100k to 900k
dofs). Against CG + GAMG on the same assembled tangent the Newton iteration is ×5 faster at 800k–900k dofs,
and ×3 against the PETSc fieldsplit on the electro-mechanical plate at 555k; with `forcing = EisenstatWalker()`
(the recommended configuration, as above; the defaults keep the exact Newton) the Krylov iterations drop by
×4–5 and the whole simulation of the 893k-dof tetrahedral cube takes 86 s against 815 s with CG + GAMG (our
measurement: Mac, 6 Julia threads, PETSc serial). With `tangent = HybridTangent(…)` an indefinite tangent
reported by these solvers (a limit point, pull-in, buckling) switches the Newton to the projected tangent at
the same state instead of bisecting the load step.
