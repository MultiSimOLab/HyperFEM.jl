using Gridap.CellData


function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractMatrix)
  mortar(map(i -> allocate_in_domain(M), 1:nblocks))
end


get_local_matrix_type(a::Assembler) = get_matrix_type(a)
get_local_vector_type(a::Assembler) = get_vector_type(a)
get_local_assembly_strategy(a::Assembler) = get_assembly_strategy(a)

function get_local_matrix_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_matrix_type(first(a.block_assemblers))
end
function get_local_vector_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_vector_type(first(a.block_assemblers))
end
function get_local_assembly_strategy(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_assembly_strategy(first(a.block_assemblers))
end

