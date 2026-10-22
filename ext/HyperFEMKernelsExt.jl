module HyperFEMKernelsExt

# Kernel backend for HyperFEM's drivers: `StaticNonlinearModel(...; backend = :kernel)`.
# The first load step builds a `KernelOperator` (tables, sparsity pattern, colouring); later
# steps only re-capture the weak form for the new load factor with `update`.

using HyperFEM
using HyperFEMKernels: KernelOperator, update
using Gridap.FESpaces: FEFunction
using Gridap.Algebra: AffineOperator, residual, jacobian, zero_initial_guess, solve

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

end # module
