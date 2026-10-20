module HyperFEMKernelsExt

# Kernel backend for HyperFEM's drivers: `StaticNonlinearModel(...; backend = :kernel)`.
# The first load step builds a `KernelOperator` (tables, sparsity pattern, colouring); later
# steps only re-capture the weak form for the new load factor with `update`.

using HyperFEM
using HyperFEMKernels: KernelOperator, update

function HyperFEM.ComputationalModels.build_operator(::Val{:kernel}, res, jac, U, V, assem, previous)
    previous isa KernelOperator ? update(previous, res, jac) : KernelOperator(res, jac, U, V)
end

end # module
