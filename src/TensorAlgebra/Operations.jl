# Tensor operations.
#
# The tensors are stored with full indices (Gridap ≥ 0.20). Gridap provides the products
# (`⊗`, `⋅`, `⋅²`, `⊙` between tensors of the same order) for all the orders. HyperFEM adds:
#   - `⊙` between tensors of different orders: all the indices of the operand of lower order;
#   - the products in index notation that the operators above cannot write;
#   - `+`, `-` and the scaling of high-order tensors, and `A ⊗ B`, which are slower in Gridap;
#   - the transpose of third- and fourth-order tensors, and the cross products.
#
# Public operators:
#   a ⊗ b    outer product
#   a ⋅ b    single contraction: last index of `a` with the first index of `b`
#   a ⋅² b   double contraction: last two indices of `a` with the first two indices of `b`
#   a ⊙ b    full contraction: all the indices of the operand of lower order
#   a × b    cross product (3D)
#   a * b    same as `a ⋅ b` for second-order tensors (to be removed)


# =====================
# Arithmetic
# =====================

function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end

@inline @generated function (+)(A::HighOrderTensorValue{S,TA,N,L}, B::HighOrderTensorValue{S,TB,N,L}) where {S,TA,TB,N,L}
  components = [:(A.data[$i] + B.data[$i]) for i in 1:L]
  :(HighOrderTensorValue{S,$(promote_type(TA, TB)),N,L}(($(components...),)))
end

@inline @generated function (-)(A::HighOrderTensorValue{S,TA,N,L}, B::HighOrderTensorValue{S,TB,N,L}) where {S,TA,TB,N,L}
  components = [:(A.data[$i] - B.data[$i]) for i in 1:L]
  :(HighOrderTensorValue{S,$(promote_type(TA, TB)),N,L}(($(components...),)))
end

@inline @generated function (-)(A::HighOrderTensorValue{S,T,N,L}) where {S,T,N,L}
  components = [:(-A.data[$i]) for i in 1:L]
  :(HighOrderTensorValue{S,T,N,L}(($(components...),)))
end

@inline (+)(A::HighOrderTensorValue) = A

# Scaling and outer product of second-order tensors: Gridap's are 2× slower for 81 components.

@inline @generated function (*)(α::Real, A::HighOrderTensorValue{S,T,N,L}) where {S,T,N,L}
  components = [:(α * A.data[$i]) for i in 1:L]
  :(HighOrderTensorValue{S,$(promote_type(α, T)),N,L}(($(components...),)))
end

@inline (*)(A::HighOrderTensorValue, α::Real) = α * A

# Multiplication by the reciprocal: one division instead of 81, but not bitwise identical to dividing
# each component (the reciprocal is rounded).
@inline Base.:(/)(A::HighOrderTensorValue, α::Real) = inv(α) * A

@inline Gridap.TensorValues.outer(A::TensorValue{D,D}, B::TensorValue{D,D}) where {D} =
  einstein_sum(Val((:i, :j)), Val((:k, :l)), A, B)

# Double contraction of fourth-order tensors: Gridap's is 1.6× slower and its first call takes 4 s in 3D (0.4 s here).
@inline Gridap.TensorValues.double_contraction(H::FourthOrderTensorValue{D}, G::FourthOrderTensorValue{D}) where {D} =
  einstein_sum(Val((:i, :j, :m, :n)), Val((:m, :n, :k, :l)), H, G)


# =====================
# Products in index notation
# =====================
# Subscripts: indices of the first operand. Superscripts: indices of the second operand.
# The free indices i, j, k, l are the indices of the result, the indices m, n are summed.

"""
    A ⊗ᵢₖʲ v

Outer product with the indices of the result reordered: `R[i,j,k] = A[i,k] v[j]`.
"""
@inline (⊗ᵢₖʲ)(A::TensorValue, v::VectorValue) = einstein_sum(Val((:i, :k)), Val((:j,)), A, v)

"""
    A ⊗ᵢₖʲˡ B

Outer product with the indices of the result reordered: `R[i,j,k,l] = A[i,k] B[j,l]`.
"""
@inline (⊗ᵢₖʲˡ)(A::TensorValue, B::TensorValue) = einstein_sum(Val((:i, :k)), Val((:j, :l)), A, B)

"""
    A ⊗ᵢₗʲᵏ B

Outer product with the indices of the result reordered: `R[i,j,k,l] = A[i,l] B[j,k]`.
"""
@inline (⊗ᵢₗʲᵏ)(A::TensorValue, B::TensorValue) = einstein_sum(Val((:i, :l)), Val((:j, :k)), A, B)

"""
    A ⊗ᵢₘʲᵐᵏˡ H

Contraction of the second index of `A` with the second index of `H`: `R[i,j,k,l] = Σm A[i,m] H[j,m,k,l]`.
"""
@inline (⊗ᵢₘʲᵐᵏˡ)(A::TensorValue, H::FourthOrderTensorValue) = einstein_sum(Val((:i, :m)), Val((:j, :m, :k, :l)), A, H)


# =====================
# Full contraction between tensors of different orders
# =====================
# Gridap 0.20 defines the first two, but marks them to be removed.

Gridap.TensorValues.inner(H::FourthOrderTensorValue{D}, A::TensorValue{D,D}) where {D} = H ⋅² A
Gridap.TensorValues.inner(A::TensorValue{D,D}, H::FourthOrderTensorValue{D}) where {D} = A ⋅² H
Gridap.TensorValues.inner(W::ThirdOrderTensorValue{D,D,D}, A::TensorValue{D,D}) where {D} = W ⋅² A
Gridap.TensorValues.inner(W::ThirdOrderTensorValue{D,D,D}, v::VectorValue{D}) where {D} = W ⋅ v
Gridap.TensorValues.inner(v::VectorValue{D}, W::ThirdOrderTensorValue{D,D,D}) where {D} = v ⋅ W


# =====================
# Transpose
# =====================
# Gridap returns the tensor unchanged.

"""
    transpose(W::ThirdOrderTensorValue{D,D,D})

Transpose of a third-order tensor in the sense `x·W·y·z = y·Wᵀ·z·x` for all `x,y,z`,
i.e. the cyclic index shift `Wᵀ[i,j,k] = W[k,i,j]`. Applied twice, it is not the identity.
"""
@inline Base.transpose(W::ThirdOrderTensorValue{D,D,D}) where {D} = permutedims(W, Val((2, 3, 1)))

"""
    transpose(H::FourthOrderTensorValue)

Major transpose of a fourth-order tensor, `Hᵀ[i,j,k,l] = H[k,l,i,j]`, such that `A ⊙ Hᵀ ⊙ B = B ⊙ H ⊙ A`.
"""
@inline @generated function Base.transpose(H::FourthOrderTensorValue{D}) where {D}
  components = [component(:tH, (k, l, i, j), D) for l in 1:D for k in 1:D for j in 1:D for i in 1:D]
  quote
    tH = Tuple(H)
    $(tensor_type(4, D, eltype(H)))(($(components...),))
  end
end

@inline Base.adjoint(W::ThirdOrderTensorValue{D,D,D,<:Real}) where {D} = transpose(W)
@inline Base.adjoint(H::FourthOrderTensorValue{D,<:Real}) where {D} = transpose(H)


# =====================
# Norm
# =====================
# Gridap defines the (Frobenius) norm for the tensors of order 1 and 2 only. It is also used by `isapprox`.
# (Gridap's `inner` of two fourth-order tensors allocates: a single sum of 81 terms.)

LinearAlgebra.norm(A::HighOrderTensorValue{S,<:Real}) where {S} = sqrt(sum(abs2, Tuple(A)))


# =====================
# muladd (temporary, until a Gridap release > 0.20.10)
# =====================
# Julia 1.14 computes the generic matrix-vector product with `muladd(a, b, c)`. Gridap 0.20.10 has no
# `muladd` for its tensors, and the fallback promotes a scalar and a tensor, which fails: e.g. when Gridap
# builds the nodes of the Lagrangian elements of order ≥ 2 (`Matrix{Float64} * Vector{VectorValue}`).
# Gridap defines it after 0.20.10 (gridap/Gridap.jl#1347). The methods are only defined while Gridap does
# not have them: redefining Gridap's methods would break the precompilation. Remove when Gridap > 0.20.10
# is required.

if which(muladd, Tuple{Float64,VectorValue{2,Float64},VectorValue{2,Float64}}).module === Base
  Base.muladd(a::Number, b::MultiValue, c::MultiValue) = a * b + c
  Base.muladd(a::MultiValue, b::Number, c::MultiValue) = a * b + c
end


# =====================
# Other products
# =====================

"""
    IIsym(A::TensorValue{D,D})::FourthOrderTensorValue{D}

Symmetrised outer product `IIsym(A)[i,j,k,l] = ½ (A[i,k] A[j,l] + A[i,l] A[j,k])`.
"""
@inline @generated function IIsym(A::TensorValue{D,D}) where {D}
  a(I...) = component(:tA, I, D)
  components = [:(0.5 * ($(a(i, k)) * $(a(j, l)) + $(a(i, l)) * $(a(j, k)))) for l in 1:D for k in 1:D for j in 1:D for i in 1:D]
  quote
    tA = Tuple(A)
    $(tensor_type(4, D, eltype(A)))(($(components...),))
  end
end

"""
    push_forward_C_to_F(F::TensorValue{D,D}, H::FourthOrderTensorValue{D})::FourthOrderTensorValue{D}

Assumming `C` is symmetric, compute directly `0.5 * DCDF' · H · DCDF` without
computing the 4th order tensor `DCDF`:

    R[i,j,k,l] = Σmn ½ F[i,m] F[k,n] (H[m,j,n,l] + H[m,j,l,n] + H[j,m,n,l] + H[j,m,l,n])
"""
@inline @generated function push_forward_C_to_F(F::TensorValue{D,D}, H::FourthOrderTensorValue{D}) where {D}
  f(I...) = component(:tF, I, D)
  h(I...) = component(:tH, I, D)
  components = map(Iterators.product(1:D, 1:D, 1:D, 1:D)) do (i, j, k, l)
    terms = [:(0.5 * $(f(i, m)) * $(f(k, n)) * ($(h(m, j, n, l)) + $(h(m, j, l, n)) + $(h(j, m, n, l)) + $(h(j, m, l, n))))
             for n in 1:D for m in 1:D]
    Expr(:call, :+, terms...)
  end
  quote
    tF, tH = Tuple(F), Tuple(H)
    $(tensor_type(4, D, promote_type(eltype(F), eltype(H))))(($(components...),))
  end
end

# Former names and methods (temporary). The benchmark job of a pull request runs the benchmark scripts
# of `main` against the pull request (AirspeedVelocity freezes the script at the default branch); these
# keep them running. Remove them, with I3, I4, I9 (TensorsDefinitions.jl) and `I3` in Exports.jl, once
# `main` has the new benchmark scripts.
Base.@deprecate_binding δᵢₖδⱼₗ2D (I2D ⊗ᵢₖʲˡ I2D)
Base.@deprecate_binding δᵢₗδⱼₖ2D (I2D ⊗ᵢₗʲᵏ I2D)
Base.@deprecate_binding δᵢⱼδₖₗ2D (I2D ⊗ I2D)
@deprecate push_forward_C_to_F(F::TensorValue{D,D}, H::TensorValue{N,N}) where {D,N} push_forward_C_to_F(F, FourthOrderTensorValue(Tuple(H)))


# =====================
# Cross products (3D)
# =====================

@inline function (×ᵢ⁴)(A::TensorValue{3,3})
  FourthOrderTensorValue{3}(0.0, 0.0, 0.0, 0.0, A[9], -A[8], 0.0, -A[6], A[5], 0.0, 0.0, 0.0, -A[9],
    0.0, A[7], A[6], 0.0, -A[4], 0.0, 0.0, 0.0, A[8], -A[7], 0.0, -A[5], A[4], 0.0, 0.0, -A[9],
    A[8], 0.0, 0.0, 0.0, 0.0, A[3], -A[2], A[9], 0.0, -A[7], 0.0, 0.0, 0.0, -A[3], 0.0,
    A[1], -A[8], A[7], 0.0, 0.0, 0.0, 0.0, A[2], -A[1], 0.0, 0.0, A[6], -A[5], 0.0,
    -A[3], A[2], 0.0, 0.0, 0.0, -A[6], 0.0, A[4], A[3], 0.0, -A[1],
    0.0, 0.0, 0.0, A[5], -A[4], 0.0, -A[2], A[1], 0.0, 0.0, 0.0, 0.0)
end


@inline function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue{3,3}(A[5] * B[9] - A[6] * B[8] - A[8] * B[6] + A[9] * B[5],
    A[6] * B[7] - A[4] * B[9] + A[7] * B[6] - A[9] * B[4],
    A[4] * B[8] - A[5] * B[7] - A[7] * B[5] + A[8] * B[4],
    A[3] * B[8] - A[2] * B[9] + A[8] * B[3] - A[9] * B[2],
    A[1] * B[9] - A[3] * B[7] - A[7] * B[3] + A[9] * B[1],
    A[2] * B[7] - A[1] * B[8] + A[7] * B[2] - A[8] * B[1],
    A[2] * B[6] - A[3] * B[5] - A[5] * B[3] + A[6] * B[2],
    A[3] * B[4] - A[1] * B[6] + A[4] * B[3] - A[6] * B[1],
    A[1] * B[5] - A[2] * B[4] - A[4] * B[2] + A[5] * B[1])
end


@inline function Gridap.TensorValues.cross(H::FourthOrderTensorValue{3,T1}, A::TensorValue{3,3,T2}) where {T1,T2}

  FourthOrderTensorValue{3}(A[9] * H[37] - A[8] * H[46] - A[6] * H[64] + A[5] * H[73],
    A[9] * H[38] - A[8] * H[47] - A[6] * H[65] + A[5] * H[74],
    A[9] * H[39] - A[8] * H[48] - A[6] * H[66] + A[5] * H[75],
    A[9] * H[40] - A[8] * H[49] - A[6] * H[67] + A[5] * H[76],
    A[9] * H[41] - A[8] * H[50] - A[6] * H[68] + A[5] * H[77],
    A[9] * H[42] - A[8] * H[51] - A[6] * H[69] + A[5] * H[78],
    A[9] * H[43] - A[8] * H[52] - A[6] * H[70] + A[5] * H[79],
    A[9] * H[44] - A[8] * H[53] - A[6] * H[71] + A[5] * H[80],
    A[9] * H[45] - A[8] * H[54] - A[6] * H[72] + A[5] * H[81],
    A[7] * H[46] - A[9] * H[28] + A[6] * H[55] - A[4] * H[73],
    A[7] * H[47] - A[9] * H[29] + A[6] * H[56] - A[4] * H[74],
    A[7] * H[48] - A[9] * H[30] + A[6] * H[57] - A[4] * H[75],
    A[7] * H[49] - A[9] * H[31] + A[6] * H[58] - A[4] * H[76],
    A[7] * H[50] - A[9] * H[32] + A[6] * H[59] - A[4] * H[77],
    A[7] * H[51] - A[9] * H[33] + A[6] * H[60] - A[4] * H[78],
    A[7] * H[52] - A[9] * H[34] + A[6] * H[61] - A[4] * H[79],
    A[7] * H[53] - A[9] * H[35] + A[6] * H[62] - A[4] * H[80],
    A[7] * H[54] - A[9] * H[36] + A[6] * H[63] - A[4] * H[81],
    A[8] * H[28] - A[7] * H[37] - A[5] * H[55] + A[4] * H[64],
    A[8] * H[29] - A[7] * H[38] - A[5] * H[56] + A[4] * H[65],
    A[8] * H[30] - A[7] * H[39] - A[5] * H[57] + A[4] * H[66],
    A[8] * H[31] - A[7] * H[40] - A[5] * H[58] + A[4] * H[67],
    A[8] * H[32] - A[7] * H[41] - A[5] * H[59] + A[4] * H[68],
    A[8] * H[33] - A[7] * H[42] - A[5] * H[60] + A[4] * H[69],
    A[8] * H[34] - A[7] * H[43] - A[5] * H[61] + A[4] * H[70],
    A[8] * H[35] - A[7] * H[44] - A[5] * H[62] + A[4] * H[71],
    A[8] * H[36] - A[7] * H[45] - A[5] * H[63] + A[4] * H[72],
    A[8] * H[19] - A[9] * H[10] + A[3] * H[64] - A[2] * H[73],
    A[8] * H[20] - A[9] * H[11] + A[3] * H[65] - A[2] * H[74],
    A[8] * H[21] - A[9] * H[12] + A[3] * H[66] - A[2] * H[75],
    A[8] * H[22] - A[9] * H[13] + A[3] * H[67] - A[2] * H[76],
    A[8] * H[23] - A[9] * H[14] + A[3] * H[68] - A[2] * H[77],
    A[8] * H[24] - A[9] * H[15] + A[3] * H[69] - A[2] * H[78],
    A[8] * H[25] - A[9] * H[16] + A[3] * H[70] - A[2] * H[79],
    A[8] * H[26] - A[9] * H[17] + A[3] * H[71] - A[2] * H[80],
    A[8] * H[27] - A[9] * H[18] + A[3] * H[72] - A[2] * H[81],
    A[9] * H[1] - A[7] * H[19] - A[3] * H[55] + A[1] * H[73],
    A[9] * H[2] - A[7] * H[20] - A[3] * H[56] + A[1] * H[74],
    A[9] * H[3] - A[7] * H[21] - A[3] * H[57] + A[1] * H[75],
    A[9] * H[4] - A[7] * H[22] - A[3] * H[58] + A[1] * H[76],
    A[9] * H[5] - A[7] * H[23] - A[3] * H[59] + A[1] * H[77],
    A[9] * H[6] - A[7] * H[24] - A[3] * H[60] + A[1] * H[78],
    A[9] * H[7] - A[7] * H[25] - A[3] * H[61] + A[1] * H[79],
    A[9] * H[8] - A[7] * H[26] - A[3] * H[62] + A[1] * H[80],
    A[9] * H[9] - A[7] * H[27] - A[3] * H[63] + A[1] * H[81],
    A[7] * H[10] - A[8] * H[1] + A[2] * H[55] - A[1] * H[64],
    A[7] * H[11] - A[8] * H[2] + A[2] * H[56] - A[1] * H[65],
    A[7] * H[12] - A[8] * H[3] + A[2] * H[57] - A[1] * H[66],
    A[7] * H[13] - A[8] * H[4] + A[2] * H[58] - A[1] * H[67],
    A[7] * H[14] - A[8] * H[5] + A[2] * H[59] - A[1] * H[68],
    A[7] * H[15] - A[8] * H[6] + A[2] * H[60] - A[1] * H[69],
    A[7] * H[16] - A[8] * H[7] + A[2] * H[61] - A[1] * H[70],
    A[7] * H[17] - A[8] * H[8] + A[2] * H[62] - A[1] * H[71],
    A[7] * H[18] - A[8] * H[9] + A[2] * H[63] - A[1] * H[72],
    A[6] * H[10] - A[5] * H[19] - A[3] * H[37] + A[2] * H[46],
    A[6] * H[11] - A[5] * H[20] - A[3] * H[38] + A[2] * H[47],
    A[6] * H[12] - A[5] * H[21] - A[3] * H[39] + A[2] * H[48],
    A[6] * H[13] - A[5] * H[22] - A[3] * H[40] + A[2] * H[49],
    A[6] * H[14] - A[5] * H[23] - A[3] * H[41] + A[2] * H[50],
    A[6] * H[15] - A[5] * H[24] - A[3] * H[42] + A[2] * H[51],
    A[6] * H[16] - A[5] * H[25] - A[3] * H[43] + A[2] * H[52],
    A[6] * H[17] - A[5] * H[26] - A[3] * H[44] + A[2] * H[53],
    A[6] * H[18] - A[5] * H[27] - A[3] * H[45] + A[2] * H[54],
    A[4] * H[19] - A[6] * H[1] + A[3] * H[28] - A[1] * H[46],
    A[4] * H[20] - A[6] * H[2] + A[3] * H[29] - A[1] * H[47],
    A[4] * H[21] - A[6] * H[3] + A[3] * H[30] - A[1] * H[48],
    A[4] * H[22] - A[6] * H[4] + A[3] * H[31] - A[1] * H[49],
    A[4] * H[23] - A[6] * H[5] + A[3] * H[32] - A[1] * H[50],
    A[4] * H[24] - A[6] * H[6] + A[3] * H[33] - A[1] * H[51],
    A[4] * H[25] - A[6] * H[7] + A[3] * H[34] - A[1] * H[52],
    A[4] * H[26] - A[6] * H[8] + A[3] * H[35] - A[1] * H[53],
    A[4] * H[27] - A[6] * H[9] + A[3] * H[36] - A[1] * H[54],
    A[5] * H[1] - A[4] * H[10] - A[2] * H[28] + A[1] * H[37],
    A[5] * H[2] - A[4] * H[11] - A[2] * H[29] + A[1] * H[38],
    A[5] * H[3] - A[4] * H[12] - A[2] * H[30] + A[1] * H[39],
    A[5] * H[4] - A[4] * H[13] - A[2] * H[31] + A[1] * H[40],
    A[5] * H[5] - A[4] * H[14] - A[2] * H[32] + A[1] * H[41],
    A[5] * H[6] - A[4] * H[15] - A[2] * H[33] + A[1] * H[42],
    A[5] * H[7] - A[4] * H[16] - A[2] * H[34] + A[1] * H[43],
    A[5] * H[8] - A[4] * H[17] - A[2] * H[35] + A[1] * H[44],
    A[5] * H[9] - A[4] * H[18] - A[2] * H[36] + A[1] * H[45])
end

@inline function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, H::FourthOrderTensorValue{3,T2}) where {T1,T2}

  FourthOrderTensorValue{3}(A[5] * H[9] - A[6] * H[8] - A[8] * H[6] + A[9] * H[5],
    A[6] * H[7] - A[4] * H[9] + A[7] * H[6] - A[9] * H[4],
    A[4] * H[8] - A[5] * H[7] - A[7] * H[5] + A[8] * H[4],
    A[3] * H[8] - A[2] * H[9] + A[8] * H[3] - A[9] * H[2],
    A[1] * H[9] - A[3] * H[7] - A[7] * H[3] + A[9] * H[1],
    A[2] * H[7] - A[1] * H[8] + A[7] * H[2] - A[8] * H[1],
    A[2] * H[6] - A[3] * H[5] - A[5] * H[3] + A[6] * H[2],
    A[3] * H[4] - A[1] * H[6] + A[4] * H[3] - A[6] * H[1],
    A[1] * H[5] - A[2] * H[4] - A[4] * H[2] + A[5] * H[1],
    A[5] * H[18] - A[6] * H[17] - A[8] * H[15] + A[9] * H[14],
    A[6] * H[16] - A[4] * H[18] + A[7] * H[15] - A[9] * H[13],
    A[4] * H[17] - A[5] * H[16] - A[7] * H[14] + A[8] * H[13],
    A[3] * H[17] - A[2] * H[18] + A[8] * H[12] - A[9] * H[11],
    A[1] * H[18] - A[3] * H[16] - A[7] * H[12] + A[9] * H[10],
    A[2] * H[16] - A[1] * H[17] + A[7] * H[11] - A[8] * H[10],
    A[2] * H[15] - A[3] * H[14] - A[5] * H[12] + A[6] * H[11],
    A[3] * H[13] - A[1] * H[15] + A[4] * H[12] - A[6] * H[10],
    A[1] * H[14] - A[2] * H[13] - A[4] * H[11] + A[5] * H[10],
    A[5] * H[27] - A[6] * H[26] - A[8] * H[24] + A[9] * H[23],
    A[6] * H[25] - A[4] * H[27] + A[7] * H[24] - A[9] * H[22],
    A[4] * H[26] - A[5] * H[25] - A[7] * H[23] + A[8] * H[22],
    A[3] * H[26] - A[2] * H[27] + A[8] * H[21] - A[9] * H[20],
    A[1] * H[27] - A[3] * H[25] - A[7] * H[21] + A[9] * H[19],
    A[2] * H[25] - A[1] * H[26] + A[7] * H[20] - A[8] * H[19],
    A[2] * H[24] - A[3] * H[23] - A[5] * H[21] + A[6] * H[20],
    A[3] * H[22] - A[1] * H[24] + A[4] * H[21] - A[6] * H[19],
    A[1] * H[23] - A[2] * H[22] - A[4] * H[20] + A[5] * H[19],
    A[5] * H[36] - A[6] * H[35] - A[8] * H[33] + A[9] * H[32],
    A[6] * H[34] - A[4] * H[36] + A[7] * H[33] - A[9] * H[31],
    A[4] * H[35] - A[5] * H[34] - A[7] * H[32] + A[8] * H[31],
    A[3] * H[35] - A[2] * H[36] + A[8] * H[30] - A[9] * H[29],
    A[1] * H[36] - A[3] * H[34] - A[7] * H[30] + A[9] * H[28],
    A[2] * H[34] - A[1] * H[35] + A[7] * H[29] - A[8] * H[28],
    A[2] * H[33] - A[3] * H[32] - A[5] * H[30] + A[6] * H[29],
    A[3] * H[31] - A[1] * H[33] + A[4] * H[30] - A[6] * H[28],
    A[1] * H[32] - A[2] * H[31] - A[4] * H[29] + A[5] * H[28],
    A[5] * H[45] - A[6] * H[44] - A[8] * H[42] + A[9] * H[41],
    A[6] * H[43] - A[4] * H[45] + A[7] * H[42] - A[9] * H[40],
    A[4] * H[44] - A[5] * H[43] - A[7] * H[41] + A[8] * H[40],
    A[3] * H[44] - A[2] * H[45] + A[8] * H[39] - A[9] * H[38],
    A[1] * H[45] - A[3] * H[43] - A[7] * H[39] + A[9] * H[37],
    A[2] * H[43] - A[1] * H[44] + A[7] * H[38] - A[8] * H[37],
    A[2] * H[42] - A[3] * H[41] - A[5] * H[39] + A[6] * H[38],
    A[3] * H[40] - A[1] * H[42] + A[4] * H[39] - A[6] * H[37],
    A[1] * H[41] - A[2] * H[40] - A[4] * H[38] + A[5] * H[37],
    A[5] * H[54] - A[6] * H[53] - A[8] * H[51] + A[9] * H[50],
    A[6] * H[52] - A[4] * H[54] + A[7] * H[51] - A[9] * H[49],
    A[4] * H[53] - A[5] * H[52] - A[7] * H[50] + A[8] * H[49],
    A[3] * H[53] - A[2] * H[54] + A[8] * H[48] - A[9] * H[47],
    A[1] * H[54] - A[3] * H[52] - A[7] * H[48] + A[9] * H[46],
    A[2] * H[52] - A[1] * H[53] + A[7] * H[47] - A[8] * H[46],
    A[2] * H[51] - A[3] * H[50] - A[5] * H[48] + A[6] * H[47],
    A[3] * H[49] - A[1] * H[51] + A[4] * H[48] - A[6] * H[46],
    A[1] * H[50] - A[2] * H[49] - A[4] * H[47] + A[5] * H[46],
    A[5] * H[63] - A[6] * H[62] - A[8] * H[60] + A[9] * H[59],
    A[6] * H[61] - A[4] * H[63] + A[7] * H[60] - A[9] * H[58],
    A[4] * H[62] - A[5] * H[61] - A[7] * H[59] + A[8] * H[58],
    A[3] * H[62] - A[2] * H[63] + A[8] * H[57] - A[9] * H[56],
    A[1] * H[63] - A[3] * H[61] - A[7] * H[57] + A[9] * H[55],
    A[2] * H[61] - A[1] * H[62] + A[7] * H[56] - A[8] * H[55],
    A[2] * H[60] - A[3] * H[59] - A[5] * H[57] + A[6] * H[56],
    A[3] * H[58] - A[1] * H[60] + A[4] * H[57] - A[6] * H[55],
    A[1] * H[59] - A[2] * H[58] - A[4] * H[56] + A[5] * H[55],
    A[5] * H[72] - A[6] * H[71] - A[8] * H[69] + A[9] * H[68],
    A[6] * H[70] - A[4] * H[72] + A[7] * H[69] - A[9] * H[67],
    A[4] * H[71] - A[5] * H[70] - A[7] * H[68] + A[8] * H[67],
    A[3] * H[71] - A[2] * H[72] + A[8] * H[66] - A[9] * H[65],
    A[1] * H[72] - A[3] * H[70] - A[7] * H[66] + A[9] * H[64],
    A[2] * H[70] - A[1] * H[71] + A[7] * H[65] - A[8] * H[64],
    A[2] * H[69] - A[3] * H[68] - A[5] * H[66] + A[6] * H[65],
    A[3] * H[67] - A[1] * H[69] + A[4] * H[66] - A[6] * H[64],
    A[1] * H[68] - A[2] * H[67] - A[4] * H[65] + A[5] * H[64],
    A[5] * H[81] - A[6] * H[80] - A[8] * H[78] + A[9] * H[77],
    A[6] * H[79] - A[4] * H[81] + A[7] * H[78] - A[9] * H[76],
    A[4] * H[80] - A[5] * H[79] - A[7] * H[77] + A[8] * H[76],
    A[3] * H[80] - A[2] * H[81] + A[8] * H[75] - A[9] * H[74],
    A[1] * H[81] - A[3] * H[79] - A[7] * H[75] + A[9] * H[73],
    A[2] * H[79] - A[1] * H[80] + A[7] * H[74] - A[8] * H[73],
    A[2] * H[78] - A[3] * H[77] - A[5] * H[75] + A[6] * H[74],
    A[3] * H[76] - A[1] * H[78] + A[4] * H[75] - A[6] * H[73],
    A[1] * H[77] - A[2] * H[76] - A[4] * H[74] + A[5] * H[73])
end

@inline function Gridap.TensorValues.cross(A::ThirdOrderTensorValue{3,3,3,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  ThirdOrderTensorValue{3,3,3}(A[13] * B[9] - A[16] * B[8] - A[22] * B[6] + A[25] * B[5],
    A[14] * B[9] - A[17] * B[8] - A[23] * B[6] + A[26] * B[5],
    A[15] * B[9] - A[18] * B[8] - A[24] * B[6] + A[27] * B[5],
    A[16] * B[7] - A[10] * B[9] + A[19] * B[6] - A[25] * B[4],
    A[17] * B[7] - A[11] * B[9] + A[20] * B[6] - A[26] * B[4],
    A[18] * B[7] - A[12] * B[9] + A[21] * B[6] - A[27] * B[4],
    A[10] * B[8] - A[13] * B[7] - A[19] * B[5] + A[22] * B[4],
    A[11] * B[8] - A[14] * B[7] - A[20] * B[5] + A[23] * B[4],
    A[12] * B[8] - A[15] * B[7] - A[21] * B[5] + A[24] * B[4],
    A[7] * B[8] - A[4] * B[9] + A[22] * B[3] - A[25] * B[2],
    A[8] * B[8] - A[5] * B[9] + A[23] * B[3] - A[26] * B[2],
    A[9] * B[8] - A[6] * B[9] + A[24] * B[3] - A[27] * B[2],
    A[1] * B[9] - A[7] * B[7] - A[19] * B[3] + A[25] * B[1],
    A[2] * B[9] - A[8] * B[7] - A[20] * B[3] + A[26] * B[1],
    A[3] * B[9] - A[9] * B[7] - A[21] * B[3] + A[27] * B[1],
    A[4] * B[7] - A[1] * B[8] + A[19] * B[2] - A[22] * B[1],
    A[5] * B[7] - A[2] * B[8] + A[20] * B[2] - A[23] * B[1],
    A[6] * B[7] - A[3] * B[8] + A[21] * B[2] - A[24] * B[1],
    A[4] * B[6] - A[7] * B[5] - A[13] * B[3] + A[16] * B[2],
    A[5] * B[6] - A[8] * B[5] - A[14] * B[3] + A[17] * B[2],
    A[6] * B[6] - A[9] * B[5] - A[15] * B[3] + A[18] * B[2],
    A[7] * B[4] - A[1] * B[6] + A[10] * B[3] - A[16] * B[1],
    A[8] * B[4] - A[2] * B[6] + A[11] * B[3] - A[17] * B[1],
    A[9] * B[4] - A[3] * B[6] + A[12] * B[3] - A[18] * B[1],
    A[1] * B[5] - A[4] * B[4] - A[10] * B[2] + A[13] * B[1],
    A[2] * B[5] - A[5] * B[4] - A[11] * B[2] + A[14] * B[1],
    A[3] * B[5] - A[6] * B[4] - A[12] * B[2] + A[15] * B[1])
end

@inline function Gridap.TensorValues.cross(A::SMatrix, B::SMatrix)
  return get_array(TensorValue(A) × TensorValue(B))
end

@inline function Gridap.TensorValues.outer(A::SVector, B::SVector)
  return get_array(VectorValue(A) ⊗ VectorValue(B))
end
