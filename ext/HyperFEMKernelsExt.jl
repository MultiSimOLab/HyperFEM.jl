module HyperFEMKernelsExt

# Kernel backend for HyperFEM's drivers: `StaticNonlinearModel(...; backend = :kernel)` and
# `StaticLinearModel(...; backend = :kernel)`.
# The first load step builds a `KernelOperator` (tables, sparsity pattern, colouring); later
# steps only re-capture the weak form for the new load factor with `update`.

using HyperFEM
using HyperFEMKernels: KernelOperator, update, KernelState, KernelUpdater
using Gridap.FESpaces: FEFunction
using Gridap.Algebra: AffineOperator, residual, jacobian, residual!, jacobian!, zero_initial_guess, solve
using Gridap.Algebra: allocate_residual, allocate_jacobian

function HyperFEM.ComputationalModels.build_operator(::Val{:kernel}, res, jac, U, V, assem, previous)
    previous isa KernelOperator ? update(previous, res, jac) : KernelOperator(res, jac, U, V)
end

# Dirichlet projection (`solve!(...; ProjectDirichlet=true)`): the linearised problem
# jac(uh, du, v) = -res(uh, v) on ∆U is the operator r(du, v) = jac(uh, du, v) + res(uh, v),
# j(du, ddu, v) = jac(uh, ddu, v), evaluated at du = 0: the residual there is res(uh, v) plus the
# contribution of the Dirichlet increment stored in ∆U, and the Jacobian is the tangent. The
# operator of the previous step is reused (`update`): same spaces, pattern and colouring.
function HyperFEM.ComputationalModels.dirichlet_projection(::Val{:kernel}, res, jac, uh, ∆U, V, ls, previous)
    r(du, v) = jac(uh, du, v) + res(uh, v)
    j(du, ddu, v) = jac(uh, ddu, v)
    op = previous isa KernelOperator ? update(previous, r, j) : KernelOperator(r, j, ∆U, V)
    x0 = zero_initial_guess(op)
    b = residual(op, x0)
    A = jacobian(op, x0)
    b .*= -1.0
    dx = solve(ls, AffineOperator(A, b))
    FEFunction(∆U, dx), op
end

# Linear problems (`StaticLinearModel(...; backend = :kernel)`): the affine problem K x = b of
# bf(u, v) = lf(v) is the operator r(u, v) = bf(u, v) - lf(v), j(u, du, v) = bf(du, v) at u = 0, so
# K = j and b = -r(0); r(0) carries the Dirichlet lift (the FE function of u = 0 holds the Dirichlet
# values of U, read by the kernels at every assembly). Re-assembly re-captures the forms with
# `update` (same tables, pattern and colouring), as the nonlinear driver does per load step. A
# linear form returning a number (`l(v) = 0.0`, no load) is accepted as in Gridap.
_linear_res(bf, lf) = (u, v) -> _minus(bf(u, v), lf(v))
_linear_res(bf, ::Nothing) = bf
_linear_jac(bf) = (u, du, v) -> bf(du, v)
_load_res(lf) = (u, v) -> -1.0 * _contribution(lf(v))
_minus(a, b) = a - b
_minus(a, b::Number) = iszero(b) ? a : error("HyperFEM: a linear form must return integrals (got $b)")
_contribution(c) = c
_contribution(c::Number) = iszero(c) ? error("HyperFEM: a linear form without integrals (got $c) cannot be assembled by the kernel backend") :
    error("HyperFEM: a linear form must return integrals (got $c)")

_kernel_operator(op, r, j, U, V) = op isa KernelOperator ? update(op, r, j) : KernelOperator(r, j, U, V)

function _linear_assemble!(K, b, op, x0)
    jacobian!(K, op, x0)
    if b !== nothing
        residual!(b, op, x0)
        b .*= -1.0
    end
end

# `K` was allocated from the pattern of a previous operator: `update` keeps it unless fallback terms
# (assembled by Gridap) changed the pattern, which the LU setup of the model could not follow.
function _check_pattern(K, op::KernelOperator, previous)
    previous isa KernelOperator && op.A0 === previous.A0 && return
    (size(K) == size(op.A0) && K.colptr == op.A0.colptr && K.rowval == op.A0.rowval) ||
        error("HyperFEM: the sparsity pattern of the kernel operator changed on re-assembly; build a new StaticLinearModel")
end

function HyperFEM.ComputationalModels.linear_system(::Val{:kernel}, bf, lf, U, V, assem)
    op = KernelOperator(_linear_res(bf, lf), _linear_jac(bf), U, V)
    x0 = zero_initial_guess(op)
    K, b = allocate_jacobian(op, x0), allocate_residual(op, x0)
    _linear_assemble!(K, lf === nothing ? nothing : b, op, x0)
    K, b, op
end

function HyperFEM.ComputationalModels.linear_system!(::Val{:kernel}, K, b, bf, lf, U, V, assem, previous)
    op = _kernel_operator(previous, _linear_res(bf, lf), _linear_jac(bf), U, V)
    _check_pattern(K, op, previous)
    _linear_assemble!(K, lf === nothing ? nothing : b, op, zero_initial_guess(op))
    op
end

function HyperFEM.ComputationalModels.load_vector!(::Val{:kernel}, b, lf, U, V, assem, previous)
    previous isa KernelOperator || error("HyperFEM: load_vector! with the kernel backend needs the operator of linear_system")
    op = update(previous, _load_res(lf), previous.jac)
    residual!(b, op, zero_initial_guess(op))
    b .*= -1.0
    op
end

# State variables: `CellState(model, dΩ; backend = :kernel)` stores them as dense `KernelState`s, whose
# updates run with the engine automatically; `update_state!(model, ...; backend = :kernel)` runs the
# update with the engine for Gridap's `CellState`s too.
HyperFEM.PhysicalModels._new_state(::Val{:kernel}, v, points) = KernelState(v, points)
HyperFEM.PhysicalModels._state_updater(::Val{:kernel}, f) = KernelUpdater(f)

end # module
