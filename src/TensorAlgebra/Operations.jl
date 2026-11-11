# Tensor operations.
#
# HyperFEM extends the Gridap operators (type piracy) to tensors of order 3 and 4 in the flattened
# convention (see Kernel.jl), and redefines some of them for performance. All the operations are
# `@generated` functions built by the kernel of Kernel.jl: fully unrolled and allocation-free.
#
# Public operators:
#   a ⊗ b    outer product
#   a ⋅ b    single contraction: last index of `a` with the first index of `b`
#   a ⋅² b   double contraction: last two indices of `a` with the first two indices of `b`
#   a ⊙ b    full contraction: all the indices of the operand of lower order, i.e. the last indices
#            of `a` or the first indices of `b`
#   a × b    cross product, on the last two indices of `a` and the first two indices of `b`
#   a * b    same as `a ⋅ b` (to be removed)
#
# Products in index notation, e.g. `A ⊗ᵢⱼₘₙᵐⁿᵏˡ B`, see `_index_labels`.


# =====================
# Arithmetic
# =====================

function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end

# The element type is explicit so that empty tensors (e.g. `TensorValue{0,3}`,
# built by Gridap for vertex grids) are also supported.
@inline @generated function (+)(A::TensorValue{D1,D2}, B::TensorValue{D1,D2}) where {D1,D2}
  T = promote_type(eltype(A), eltype(B))
  _map_expr(:+, :(TensorValue{$D1,$D2,$T}), D1 * D2, (:A, :B))
end

@inline @generated function (-)(A::TensorValue{D1,D2}, B::TensorValue{D1,D2}) where {D1,D2}
  T = promote_type(eltype(A), eltype(B))
  _map_expr(:-, :(TensorValue{$D1,$D2,$T}), D1 * D2, (:A, :B))
end

@inline @generated function (-)(A::TensorValue{D1,D2}) where {D1,D2}
  _map_expr(:-, :(TensorValue{$D1,$D2,$(eltype(A))}), D1 * D2, (:A,))
end


# =====================
# Products in index notation
# =====================

const _INDEX_PRODUCTS = (
  # outer products
  :⊗ᵢʲ, :⊗ᵢⱼᵏ, :⊗ᵢₖʲ, :⊗ᵢʲᵏ, :⊗ᵢⱼₖˡ, :⊗ᵢⱼᵏˡ, :⊗ᵢₖʲˡ, :⊗ᵢₗʲᵏ,
  # single contractions
  :⊗ᵢⱼₘᵐ, :⊗ₘᵐⁱʲ, :⊗ᵢⱼₘᵐᵏ, :⊗ᵢₘᵐʲᵏˡ, :⊗ᵢₘʲᵐᵏˡ, :⊗ᵢⱼₘᵐᵏˡ,
  # double contractions
  :⊗ᵢₘₙᵐⁿ, :⊗ᵢⱼₘₙᵐⁿ, :⊗ᵢₘⱼₙᵐⁿ, :⊗ₘₙᵐⁿⁱʲ, :⊗ᵢⱼₘₙᵐⁿᵏˡ,
)

function _index_product_doc(op::Symbol)
  la, lb = _index_labels(op)
  labels = (la..., lb...)
  out = sort(filter(l -> count(==(l), labels) == 1, collect(labels)))
  summed = sort(unique(filter(l -> count(==(l), labels) == 2, collect(labels))))
  sum_str = isempty(summed) ? "" : "Σ" * join(summed) * " "
  """
      A $op B

  Product in index notation of tensors of order $(length(la)) and $(length(lb)), returning a tensor of order $(length(out)):

      R[$(join(out, ","))] = $(sum_str)A[$(join(la, ","))] * B[$(join(lb, ","))]

  The subscripts are the indices of `A` and the superscripts the indices of `B`. The free indices
  `i, j, k, l` give the indices of the result, in this order, and the indices `m, n, p` are summed.
  """
end

for op in _INDEX_PRODUCTS
  la, lb = _index_labels(op)
  TA = length(la) == 1 ? :VectorValue : :TensorValue
  TB = length(lb) == 1 ? :VectorValue : :TensorValue
  @eval begin
    @inline ($op)(A::$TA, B::$TB) = _einsum($(Val(la)), $(Val(lb)), A, B)
    @doc $(_index_product_doc(op)) $op
  end
end


# =====================
# Gridap operators
# =====================

Gridap.TensorValues.outer(A::VectorValue{D}, B::VectorValue{D}) where {D} = A ⊗ᵢʲ B
Gridap.TensorValues.outer(A::TensorValue{D,D}, B::TensorValue{D,D}) where {D} = A ⊗ᵢⱼᵏˡ B
Gridap.TensorValues.outer(A::TensorValue{2,4}, B::VectorValue{2}) = A ⊗ᵢⱼₖˡ B
Gridap.TensorValues.outer(A::TensorValue{3,9}, B::VectorValue{3}) = A ⊗ᵢⱼₖˡ B
# `outer(::VectorValue, ::TensorValue)` and `outer(::TensorValue, ::VectorValue)` are left to Gridap,
# which returns a `ThirdOrderTensorValue`. Redefining them to return the flattened `TensorValue{D,D²}`
# changes the gradient type Gridap derives for tensor fields and gives a wrong `∇∇` for vector fields
# (cross terms lost). The flattened products are still available as `⊗ᵢʲᵏ` and `⊗ᵢⱼᵏ`.

Gridap.TensorValues.inner(H::TensorValue{4,4}, A::TensorValue{2,2}) = H ⊗ᵢⱼₘₙᵐⁿ A
Gridap.TensorValues.inner(H::TensorValue{9,9}, A::TensorValue{3,3}) = H ⊗ᵢⱼₘₙᵐⁿ A
Gridap.TensorValues.inner(W::TensorValue{2,4}, A::TensorValue{2,2}) = W ⊗ᵢₘₙᵐⁿ A
Gridap.TensorValues.inner(W::TensorValue{3,9}, A::TensorValue{3,3}) = W ⊗ᵢₘₙᵐⁿ A
Gridap.TensorValues.inner(W::TensorValue{2,4}, v::VectorValue{2}) = W ⊗ᵢⱼₘᵐ v
Gridap.TensorValues.inner(W::TensorValue{3,9}, v::VectorValue{3}) = W ⊗ᵢⱼₘᵐ v
Gridap.TensorValues.inner(v::VectorValue{2}, W::TensorValue{2,4}) = v ⊗ₘᵐⁱʲ W
Gridap.TensorValues.inner(v::VectorValue{3}, W::TensorValue{3,9}) = v ⊗ₘᵐⁱʲ W

Gridap.TensorValues.dot(A::TensorValue{2,2}, H::TensorValue{4,4}) = A ⊗ᵢₘᵐʲᵏˡ H
Gridap.TensorValues.dot(A::TensorValue{3,3}, H::TensorValue{9,9}) = A ⊗ᵢₘᵐʲᵏˡ H
Gridap.TensorValues.dot(A::TensorValue{2,4}, B::TensorValue{2,4}) = A ⊗ᵢⱼₘᵐᵏˡ B
Gridap.TensorValues.dot(A::TensorValue{3,9}, B::TensorValue{3,9}) = A ⊗ᵢⱼₘᵐᵏˡ B
Gridap.TensorValues.dot(W::TensorValue{2,4}, A::TensorValue{2,2}) = W ⊗ᵢⱼₘᵐᵏ A
Gridap.TensorValues.dot(W::TensorValue{3,9}, A::TensorValue{3,3}) = W ⊗ᵢⱼₘᵐᵏ A
Gridap.TensorValues.dot(W::TensorValue{2,4}, v::VectorValue{2}) = W ⊗ᵢⱼₘᵐ v
Gridap.TensorValues.dot(W::TensorValue{3,9}, v::VectorValue{3}) = W ⊗ᵢⱼₘᵐ v
Gridap.TensorValues.dot(v::VectorValue{2}, W::TensorValue{2,4}) = v ⊗ₘᵐⁱʲ W
Gridap.TensorValues.dot(v::VectorValue{3}, W::TensorValue{3,9}) = v ⊗ₘᵐⁱʲ W

# `⋅²` is Gridap's alias of `double_contraction`
Gridap.TensorValues.double_contraction(H::TensorValue{4,4}, G::TensorValue{4,4}) = H ⊗ᵢⱼₘₙᵐⁿᵏˡ G
Gridap.TensorValues.double_contraction(H::TensorValue{9,9}, G::TensorValue{9,9}) = H ⊗ᵢⱼₘₙᵐⁿᵏˡ G
Gridap.TensorValues.double_contraction(H::TensorValue{4,4}, A::TensorValue{2,2}) = H ⊗ᵢⱼₘₙᵐⁿ A
Gridap.TensorValues.double_contraction(H::TensorValue{9,9}, A::TensorValue{3,3}) = H ⊗ᵢⱼₘₙᵐⁿ A
Gridap.TensorValues.double_contraction(A::TensorValue{2,2}, H::TensorValue{4,4}) = A ⊗ₘₙᵐⁿⁱʲ H
Gridap.TensorValues.double_contraction(A::TensorValue{3,3}, H::TensorValue{9,9}) = A ⊗ₘₙᵐⁿⁱʲ H
Gridap.TensorValues.double_contraction(W::TensorValue{2,4}, A::TensorValue{2,2}) = W ⊗ᵢₘₙᵐⁿ A
Gridap.TensorValues.double_contraction(W::TensorValue{3,9}, A::TensorValue{3,3}) = W ⊗ᵢₘₙᵐⁿ A


"""
    transpose(A::ThirdOrderTensorValue{D,D,D})::ThirdOrderTensorValue{D,D,D}

Transpose of a third-order tensor in the sense `x·A·y·z = y·Aᵀ·z·x` for all `x,y,z`,
i.e. the cyclic index shift

    Aᵀ[i,j,k] = A[k,i,j]

Applied twice, it is not the identity.
"""
@inline @generated function Base.transpose(A::ThirdOrderTensorValue{D,D,D}) where {D}
  _einsum_expr(('i', 'j', 'k'), (:A => ('k', 'i', 'j'),), D, eltype(A);
    constructor=:(ThirdOrderTensorValue{$D,$D,$D}))
end

# `A'`, as Gridap does for the real second-order tensors
@inline Base.adjoint(A::ThirdOrderTensorValue{D,D,D,<:Real}) where {D} = transpose(A)


# =====================
# Cross products (3D)
# =====================

"""
Cross product on the last two indices of a tensor of order `oa` and the first two indices of a tensor
of order `ob`, in 3D:

    (A × B)[α..., i, j, β...] = ε[i,m,n] ε[j,p,q] A[α..., m, p] B[n, q, β...]
"""
function _cross_expr(oa::Int, ob::Int, T; a::Symbol=:A, b::Symbol=:B)
  α = Tuple(('a', 'b')[1:oa-2])
  β = Tuple(('c', 'd')[1:ob-2])
  operands = (a => (α..., 'm', 'p'), b => ('n', 'q', β...))
  coefficient(v) = _levi_civita(v['i'], v['m'], v['n']) * _levi_civita(v['j'], v['p'], v['q'])
  _einsum_expr((α..., 'i', 'j', β...), operands, 3, T; coefficient=coefficient, reorder=_cross_terms_order)
end

@inline @generated function Gridap.TensorValues.cross(A::TensorValue{3,3}, B::TensorValue{3,3})
  _cross_expr(2, 2, promote_type(eltype(A), eltype(B)))
end

@inline @generated function Gridap.TensorValues.cross(H::TensorValue{9,9}, A::TensorValue{3,3})
  _cross_expr(4, 2, promote_type(eltype(H), eltype(A)); a=:H, b=:A)
end

@inline @generated function Gridap.TensorValues.cross(A::TensorValue{3,3}, H::TensorValue{9,9})
  _cross_expr(2, 4, promote_type(eltype(A), eltype(H)); a=:A, b=:H)
end

@inline @generated function Gridap.TensorValues.cross(A::TensorValue{3,9}, B::TensorValue{3,3})
  _cross_expr(3, 2, promote_type(eltype(A), eltype(B)))
end

@inline function Gridap.TensorValues.cross(A::SMatrix, B::SMatrix)
  return get_array(TensorValue(A) × TensorValue(B))
end

@inline function Gridap.TensorValues.outer(A::SVector, B::SVector)
  return get_array(VectorValue(A) ⊗ VectorValue(B))
end

"""
    ×ᵢ⁴(A::TensorValue{3,3})::TensorValue{9,9}

Fourth-order tensor such that `(×ᵢ⁴(A)) ⊙ B = A × B` for every second-order tensor `B`,
i.e. the derivative of the cofactor, `∂cof(A)/∂A = ×ᵢ⁴(A)`:

    ×ᵢ⁴(A)[i,j,k,l] = ε[i,m,k] ε[j,p,l] A[m,p]
"""
@inline @generated function (×ᵢ⁴)(A::TensorValue{3,3})
  coefficient(v) = _levi_civita(v['i'], v['m'], v['k']) * _levi_civita(v['j'], v['p'], v['l'])
  _einsum_expr(('i', 'j', 'k', 'l'), (:A => ('m', 'p'),), 3, eltype(A); coefficient=coefficient)
end


# =====================
# Other products
# =====================

"""
    IIsym(A::TensorValue{D,D})::TensorValue{D²,D²}

Symmetrised outer product `IIsym(A)[i,j,k,l] = ½ (A[i,k] A[j,l] + A[i,l] A[j,k])`.
"""
@inline @generated function IIsym(A::TensorValue{D,D}) where {D}
  a(I...) = _component(:A, I, D)
  _tensor_expr(4, D, ((i, j, k, l),) -> :(0.5 * ($(a(i, k)) * $(a(j, l)) + $(a(i, l)) * $(a(j, k)))))
end

"""
    push_forward_C_to_F(F::TensorValue{D,D}, H::TensorValue{D²,D²})::TensorValue{D²,D²}

Assumming `C` is symmetric, compute directly `0.5 * DCDF' · H · DCDF` without
computing the 4th order tensor `DCDF`:

    R[i,j,k,l] = Σmn ½ F[i,m] F[k,n] (H[m,j,n,l] + H[m,j,l,n] + H[j,m,n,l] + H[j,m,l,n])
"""
@inline @generated function push_forward_C_to_F(F::TensorValue{D,D}, H::TensorValue{D²,D²}) where {D,D²}
  D * D == D² || return :(throw(DimensionMismatch("Mismatch dimensions of F ($D) and H ($D²).")))
  T = promote_type(eltype(F), eltype(H))
  h(I...) = _component(:H, I, D)
  _tensor_expr(4, D, ((i, j, k, l),) -> _sum_expr([
      Expr(:call, :*, 0.5, _component(:F, (i, m), D), _component(:F, (k, n), D),
        Expr(:call, :+, h(m, j, n, l), h(m, j, l, n), h(j, m, n, l), h(j, m, l, n)))
      for n in 1:D for m in 1:D], T))
end
