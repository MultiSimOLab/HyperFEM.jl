
# Explicit keyword arguments (e.g. `dirichlet_masks`) take precedence over the boundary conditions.
function Gridap.FESpaces.TestFESpace(model, reffe, bc::BoundaryConditions; kwargs...)
  bc_kwargs = (dirichlet_tags=get_tags(bc), dirichlet_masks=get_masks(bc))
  TestFESpace(model, reffe; merge(bc_kwargs, kwargs)...)
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real=0.0)
  TrialFESpace(space, get_space_functions(bc, Λ))
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, ::NothingBC, Λ::Real=0.0)
  space
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real)
  TrialFESpace!(space, get_space_functions(bc, Λ))
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Real)
  space
end

"""
    TrialFESpace!(space, bc, Λ, ΔΛ)

Set the Dirichlet values to the increment between the load parameters `Λ - ΔΛ` and `Λ`.
"""
function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::BoundaryConditions, Λ::Real, ΔΛ::Real)
  TrialFESpace!(space, bc, Λ - ΔΛ)
  values⁻ = copy(get_dirichlet_dof_values(space))
  TrialFESpace!(space, bc, Λ)
  get_dirichlet_dof_values(space) .-= values⁻
  space
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Real, ΔΛ::Real)
  space
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Real)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc[i], Λ)
  end
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Real, ΔΛ::Real)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc[i], Λ, ΔΛ)
  end
end

function Gridap.FESpaces.TrialFESpace(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Real=0.0)
  U_ = Vector{Union{TrialFESpace,UnconstrainedFESpace,ConstantFESpace}}(undef, length(space))
  @inbounds for (i, space) in enumerate(space.spaces)
    U_[i] = TrialFESpace(space, bc[i], Λ)
  end
  return MultiFieldFESpace(U_)
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

function instantiate_caches(x, nls::NewtonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A, x)
  return GridapSolvers.NonlinearSolvers.NewtonCache(A, b, dx, ns)
end

function instantiate_caches(x, nls::Newton_RaphsonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A, x)
  return Newton_RaphsonCache(A, b, dx, ns)
end
