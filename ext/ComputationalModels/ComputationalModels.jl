module ComputationalModels

using HyperFEM
using Gridap, Gridap.Algebra, GridapDistributed, GridapPETSc
using PartitionedArrays
using PartitionedArrays: getany


function HyperFEM.ComputationalModels.instantiate_caches(x, nls::PETScNonlinearSolver, op::NonlinearOperator)
  return GridapPETSc._setup_cache(x, nls, op)
end

function HyperFEM.ComputationalModels.get_local_matrix_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_matrix_type,a.assems))
end

function HyperFEM.ComputationalModels.get_local_vector_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_vector_type,a.assems))
end

function HyperFEM.ComputationalModels.get_local_assembly_strategy(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return get_assembly_strategy(a)
end

end
