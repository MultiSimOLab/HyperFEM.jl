# Code generation of the products in index notation.
#
# The tensors are stored with full indices: a tensor of order `o` in dimension `D` is a
# `VectorValue{D}`, a `TensorValue{D,D}`, a `ThirdOrderTensorValue{D,D,D}` or a `FourthOrderTensorValue{D}`
# (o = 1 to 4), and its components are read as `A[i,j,…]`.

"""Expression of the type of a tensor of order `o` in dimension `D`, with elements of type `T`."""
function tensor_type(o::Int, D::Int, T)
  o == 1 && return :(VectorValue{$D,$T})
  o == 2 && return :(TensorValue{$D,$D,$T})
  o == 3 && return :(ThirdOrderTensorValue{$D,$D,$D,$T})
  o == 4 && return :(FourthOrderTensorValue{$D,$T})
  throw(ArgumentError("Trying to get the type of a tensor of order $o"))
end

"""
Expression of the component `I` (a tuple of indices) of a tensor in dimension `D`, read from the tuple
`t` of its components with the linear index (column-major).

The generated functions read the components from the tuple, `t = Tuple(A)`. A tuple index which is a
literal number is resolved at once, while `A[i,j,…]` adds a bounds check per access which the compiler has
to remove: same runtime, but a much longer compilation (e.g. 20 s instead of 0.3 s for push_forward_C_to_F in 3D).
"""
component(t::Symbol, I::Tuple, D::Int) = :($t[$(LinearIndices(ntuple(_ -> D, length(I)))[I...])])

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
      :($(component(:tA, map(l -> index[l], a), D)) * $(component(:tB, map(l -> index[l], b), D)))
    end
    Expr(:call, :+, terms...)
  end
  T = promote_type(eltype(A), eltype(B))
  quote
    tA, tB = Tuple(A), Tuple(B)
    $(tensor_type(length(free), D, T))(($(components...),))
  end
end
