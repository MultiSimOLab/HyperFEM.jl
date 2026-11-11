# Code generation of the products in index notation.
#
# The tensors are stored with full indices: a tensor of order `o` in dimension `D` is a
# `VectorValue{D}` (o = 1), a `TensorValue{D,D}` (o = 2) or a `HighOrderTensorValue{Tuple{D,…,D}}`
# (o ≥ 3), and its components are read as `A[i,j,…]`.

"""Expression of the type of a tensor of order `o` in dimension `D`, with elements of type `T`."""
function tensor_type(o::Int, D::Int, T)
  o == 1 && return :(VectorValue{$D,$T})
  o == 2 && return :(TensorValue{$D,$D,$T})
  :(HighOrderTensorValue{Tuple{$(ntuple(_ -> D, o)...)},$T,$o,$(D^o)})
end

"""
    einstein_sum(Val(a), Val(b), A, B)

Product of `A` and `B` in Einstein notation. `a` and `b` are the index labels of `A` and `B`:
a label present in both is summed, and the other labels, in alphabetical order, are the indices
of the result.

    einstein_sum(Val((:i, :m)), Val((:j, :m, :k, :l)), A, H)   # R[i,j,k,l] = Σm A[i,m] H[j,m,k,l]

The components of the result and the summed indices run in column-major order (first index fastest).
"""
@inline @generated function einstein_sum(::Val{a}, ::Val{b}, A::MultiValue, B::MultiValue) where {a,b}
  D = size(A)[1]
  if any(!=(D), (size(A)..., size(B)...))
    return :(throw(DimensionMismatch("einstein_sum expects tensors with all dimensions equal")))
  end
  labels = (a..., b...)
  free = sort([l for l in labels if count(==(l), labels) == 1])
  summed = sort([l for l in a if l in b])
  components = map(CartesianIndices(ntuple(_ -> D, length(free)))) do I
    terms = map(CartesianIndices(ntuple(_ -> D, length(summed)))) do J
      index = Dict(zip([free; summed], [Tuple(I)...; Tuple(J)...]))
      :(A[$(map(l -> index[l], a)...)] * B[$(map(l -> index[l], b)...)])
    end
    Expr(:call, :+, terms...)
  end
  T = promote_type(eltype(A), eltype(B))
  :($(tensor_type(length(free), D, T))(($(components...),)))
end
