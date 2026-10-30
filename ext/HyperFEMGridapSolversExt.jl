module HyperFEMGridapSolversExt

# Compatibility with GridapSolvers' `NewtonSolver` in HyperFEM's drivers (deprecated: HyperFEM's own
# `NewtonRaphson` replaces it; this extension is removed in the next release).

using HyperFEM
using HyperFEM.ComputationalModels: instantiate_caches, nls_converged, nls_message
using GridapSolvers
using GridapSolvers.NonlinearSolvers: NewtonSolver
using GridapSolvers.SolverInterfaces: converged, print_message
using Gridap.Algebra: NonlinearOperator, residual, jacobian, allocate_in_domain, symbolic_setup, numerical_setup

function HyperFEM.ComputationalModels.instantiate_caches(x, nls::NewtonSolver, op::NonlinearOperator)
    Base.depwarn("GridapSolvers' `NewtonSolver` in HyperFEM's drivers is deprecated; use `NewtonRaphson` " *
                 "(same iterations and results with the default line search)", :instantiate_caches)
    b = residual(op, x)
    A = jacobian(op, x)
    dx = allocate_in_domain(A)
    fill!(dx, zero(eltype(dx)))
    ss = symbolic_setup(nls.ls, A)
    ns = numerical_setup(ss, A, x)
    return GridapSolvers.NonlinearSolvers.NewtonCache(A, b, dx, ns)
end

function HyperFEM.ComputationalModels.nls_converged(nls::NewtonSolver)
    log = nls.log
    r_abs = log.residuals[log.num_iters+1]
    converged(log.tols, log.num_iters, r_abs, r_abs / log.residuals[1])
end

HyperFEM.ComputationalModels.nls_message(nls::NewtonSolver, msg) = print_message(nls.log, msg)

end # module
