
# Explicit keyword arguments (e.g. `dirichlet_masks`) take precedence over the boundary conditions.
function Gridap.FESpaces.TestFESpace(model, reffe, bc::BoundaryConditions; kwargs...)
  bc_kwargs = (dirichlet_tags=get_tags(bc), dirichlet_masks=get_masks(bc))
  TestFESpace(model, reffe; merge(bc_kwargs, kwargs)...)
end

"""
    TrialFESpace(space, bc, Λ=0; stage=Λ)

Trial space with the Dirichlet values of `bc` at the load `Λ` and the stage `τ = stage` (see
[`Prescription`](@ref)). The values are set through a [`dirichlet_applicator`](@ref) built for the
occasion; a driver keeps its own applicators (one per space) and uses the `TrialFESpace!` methods below.
"""
function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real=0.0; stage::Real=Λ)
  TrialFESpace!(TrialFESpace(zero_dirichlet_values(space), space), bc, Λ; stage)
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, ::NothingBC, Λ::Real=0.0; stage::Real=Λ)
  space
end

function Gridap.FESpaces.TrialFESpace(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Real=0.0; stage::Real=Λ)
  @assert length(space) == length(bc)
  U = map(space.spaces, bc) do s, b
    TrialFESpace(s, b, Λ; stage)
  end
  return MultiFieldFESpace(U)
end

"""
    TrialFESpace!(space, app, Λ; stage=Λ)
    TrialFESpace!(space, app, Λ, ΔΛ; stage=(Λ, ΔΛ))

Set the Dirichlet values of `space` at the load `Λ` and the stage `τ` (see [`Prescription`](@ref)), or
to their increment between `(Λ - ΔΛ, τ - Δτ)` and `(Λ, τ)` with `stage = (τ, Δτ)`. `app` is the
[`dirichlet_applicator`](@ref) of the space (`nothing` for `NothingBC`, a tuple for a multi-field space)
or the `BoundaryConditions` themselves, in which case an applicator is built for the call.
"""
function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, app::DirichletUpdater, Λ::Real; stage::Real=Λ)
  apply!(space, app, Λ, stage)
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, app::DirichletUpdater, Λ::Real, ΔΛ::Real; stage::Tuple{Real,Real}=(Λ, ΔΛ))
  apply!(space, app, Λ, ΔΛ, stage[1], stage[2])
end

Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::Nothing, Λ::Real; stage::Real=Λ) = space
Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::Nothing, Λ::Real, ΔΛ::Real; stage=(Λ, ΔΛ)) = space

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real; stage::Real=Λ)
  TrialFESpace!(space, dirichlet_applicator(space, bc), Λ; stage)
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real, ΔΛ::Real; stage::Tuple{Real,Real}=(Λ, ΔΛ))
  TrialFESpace!(space, dirichlet_applicator(space, bc), Λ, ΔΛ; stage)
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, apps::Union{Tuple,MultiFieldBC}, Λ::Real; stage::Real=Λ)
  @inbounds for (i, s) in enumerate(space.spaces)
    TrialFESpace!(s, apps[i], Λ; stage)
  end
  return space
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, apps::Union{Tuple,MultiFieldBC}, Λ::Real, ΔΛ::Real; stage::Tuple{Real,Real}=(Λ, ΔΛ))
  @inbounds for (i, s) in enumerate(space.spaces)
    TrialFESpace!(s, apps[i], Λ, ΔΛ; stage)
  end
  return space
end


function instantiate_caches(x, nls::NLSolver, op::NonlinearOperator)
  Gridap.Algebra._new_nlsolve_cache(x, nls, op)
end

function instantiate_caches(x, nls::NewtonRaphsonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = similar(b)
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A)
  return Gridap.Algebra.NewtonRaphsonCache(A, b, dx, ns)
end

instantiate_caches(x, nls::NewtonRaphson, op::NonlinearOperator) = NewtonCache(x, nls, op)


"""
    level_spaces(U, V, bc, coarse_orders = (1,)) -> Vector

Spaces of a p-multigrid hierarchy for HyperFEMKernels' `PMultigrid`, finest first: `(U, V)` (the working spaces of
the driver) followed by the Lagrangian spaces of `coarse_orders` (decreasing; `(1,)` for Q2 → Q1, `(2, 1)` for
Q3 → Q2 → Q1) on the same triangulation and with the same boundary conditions `bc`, so every level has the same
closed Dirichlet boundary. The coarse trial spaces are `TrialFESpace`s whose Dirichlet values `PMultigrid` overwrites
with the ones injected from the working state. Single-field spaces only (p-multigrid on the (u, φ) saddle point needs
block smoothers, not yet available).

    nls = NewtonRaphson(MatrixFreeSolver(; preconditioner = PMultigrid(level_spaces(U, V, D);
                                          coarse = PETScSolver(:preonly, :gamg; nullspace = N1))))
"""
function level_spaces(U::SingleFieldFESpace, V::SingleFieldFESpace, bc::BoundaryConditions, coarse_orders=(1,))
  trian = get_triangulation(V)
  T = _shape_valtype(V)
  spaces = Any[(U, V)]
  for p in coarse_orders
    p ≥ 1 || error("HyperFEM: level_spaces: orders must be ≥ 1 (got $p)")
    Vl = TestFESpace(trian, ReferenceFE(lagrangian, T, p), bc; conformity=:H1)
    push!(spaces, (TrialFESpace(Vl, bc, 0.0), Vl))
  end
  spaces
end
level_spaces(::MultiFieldFESpace, ::MultiFieldFESpace, bc, coarse_orders=(1,)) =
  error("HyperFEM: level_spaces: p-multigrid levels of multi-field spaces are not available yet (block smoothers)")

"Value type of the shape functions of a single-field space."
function _shape_valtype(s::SingleFieldFESpace)
  fields = Gridap.Arrays.testitem(Gridap.CellData.get_data(get_fe_basis(s)))
  x = Gridap.Arrays.testitem(Gridap.CellData.get_data(get_cell_points(get_triangulation(s))))
  eltype(evaluate(fields, x))
end
