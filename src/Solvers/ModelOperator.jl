"""
    ModelOperator(op; energy=nothing, max_step=nothing)

Algebraic operator handed by the drivers to [`NewtonRaphson`](@ref): the operator `op` of the
assembling backend plus the optional functions of the model that the line searches and tangent
strategies may need, implemented by the driver for every backend:

- `energy(x)`: the incremental energy whose gradient is the residual (Armijo line search);
- `max_step(x, dx, β, jmin)`: the exact injectivity step along `dx` from `x` ([`InjectivityLS`](@ref)).

Residual and Jacobian calls are forwarded to `op`. [`energy`](@ref) and [`max_step`](@ref) raise a
clear error when the model does not provide them.
"""
struct ModelOperator{O<:NonlinearOperator,E,M} <: NonlinearOperator
  op::O
  energy::E
  max_step::M
end

ModelOperator(op::NonlinearOperator; energy=nothing, max_step=nothing) = ModelOperator(op, energy, max_step)

Algebra.residual!(b::AbstractVector, o::ModelOperator, x::AbstractVector) = residual!(b, o.op, x)
Algebra.jacobian!(A, o::ModelOperator, x::AbstractVector) = jacobian!(A, o.op, x)
Algebra.jacobian!(A::AbstractMatrix, o::ModelOperator, x::AbstractVector) = jacobian!(A, o.op, x)
Algebra.residual_and_jacobian!(b, A, o::ModelOperator, x::AbstractVector) = residual_and_jacobian!(b, A, o.op, x)
Algebra.residual_and_jacobian!(b::AbstractVector, A::AbstractMatrix, o::ModelOperator, x::AbstractVector) = residual_and_jacobian!(b, A, o.op, x)
Algebra.allocate_residual(o::ModelOperator, x::AbstractVector) = allocate_residual(o.op, x)
Algebra.allocate_jacobian(o::ModelOperator, x::AbstractVector) = allocate_jacobian(o.op, x)
Algebra.zero_initial_guess(o::ModelOperator) = zero_initial_guess(o.op)
Algebra.residual(o::ModelOperator, x::AbstractVector) = residual(o.op, x)
Algebra.jacobian(o::ModelOperator, x::AbstractVector) = jacobian(o.op, x)

"Operator of the backend wrapped by a [`ModelOperator`](@ref) (itself otherwise)."
backend_operator(o::ModelOperator) = o.op
backend_operator(o::NonlinearOperator) = o

"""
    energy(op, x) -> Float64

Incremental energy of the operator at `x` (the residual is its gradient). Provided by the drivers
through [`ModelOperator`](@ref) when the model is given an `energy`.
"""
energy(o::ModelOperator, x) = _energy(o.energy, x)
_energy(::Nothing, x) = error("HyperFEM: the model provides no energy; the Armijo line search needs `energy = …` in the driver")
_energy(f, x) = f(x)
energy(::NonlinearOperator, x) = error("HyperFEM: the operator provides no energy; the Armijo line search needs a driver with `energy = …`")

"""
    max_step(op, x, dx, β, jmin) -> α

Largest injectivity-preserving step along `dx` from `x` (see [`injectivity_step`](@ref)). Provided
by the drivers through [`ModelOperator`](@ref) when the model is given `injectivity = …`.
"""
max_step(o::ModelOperator, x, dx, β, jmin) = _max_step(o.max_step, x, dx, β, jmin)
_max_step(::Nothing, x, dx, β, jmin) = error("HyperFEM: the model provides no injectivity step; `InjectivityLS` needs `injectivity = …` in the driver")
_max_step(f, x, dx, β, jmin) = f(x, dx, β, jmin)
max_step(::NonlinearOperator, x, dx, β, jmin) = error("HyperFEM: the operator provides no injectivity step; `InjectivityLS` needs a driver with `injectivity = …`")
