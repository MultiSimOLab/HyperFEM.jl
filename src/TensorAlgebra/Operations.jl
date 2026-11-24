# Tensor operations.
#
# The tensors are stored with full indices (Gridap ≥ 0.20). Gridap provides the products
# (`⊗`, `⋅`, `⋅²`, `⊙` between tensors of the same order) for all the orders. HyperFEM adds:
#   - the products in index notation that the operators above cannot write;
#   - `+`, `-` and the scaling of high-order tensors, and `A ⊗ B`, which are slower in Gridap;
#   - the transpose of third- and fourth-order tensors, and the cross products.
#
# Public operators:
#   a ⊗ b    outer product
#   a ⋅ b    single contraction: last index of `a` with the first index of `b`
#   a ⋅² b   double contraction: last two indices of `a` with the first two indices of `b`
#   a ⊙ b    full contraction: tensors of the same order (Gridap)
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

# Inner product (full contraction) of third- and fourth-order tensors: Gridap's generic fallback
# (`contracted_product`) allocates, even though it is a single unrolled sum (e.g. 81 terms in 3D/4th order).
# An `Expr(:call, :+, terms...)` with this many arguments also allocates: `+` has no method for that many
# arguments, so it lowers to `Core._apply_iterate`/`afoldl`, which boxes them. A left-associated chain of
# binary `+` avoids that.
@inline @generated function Gridap.TensorValues.inner(A::HighOrderTensorValue{S,TA,N,L}, B::HighOrderTensorValue{S,TB,N,L}) where {S,TA,TB,N,L}
  terms = [:(A.data[$i] * B.data[$i]) for i in 1:L]
  reduce((x, y) -> :($x + $y), terms)
end


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

Major transpose of a fourth-order tensor, `Hᵀ[i,j,k,l] = H[k,l,i,j]`, such that `A ⊙ (Hᵀ ⋅² B) = B ⊙ (H ⋅² A)`.
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



# =====================
# Cross products (3D)
# =====================
# Tensor cross product of de Boer (1982), as in Bonet, Gil and Ortigosa (2015), Appendix:
#
#   (A × B)[i,j] = ε[i,k,l] ε[j,m,n] A[k,m] B[l,n]                                    (A.3)
#
# Its derivative with respect to `B` is the fourth-order tensor
#
#   ×ᵢ⁴(A)[i,j,k,l] = ε[i,k,m] ε[j,l,n] A[m,n],    A × B = ×ᵢ⁴(A) ⋅² B,
#
# and the products with tensors of other orders cross the two indices next to `A`:
#
#   H × A = H ⋅² ×ᵢ⁴(A),    A × H = ×ᵢ⁴(A) ⋅² H,    W × A = W ⋅² ×ᵢ⁴(A).
#
# E.g. the Hessian of Ψ(H(F)), with H = ½ F × F, is F × (Ψ_HH × F) + ×ᵢ⁴(Ψ_H), the bilinear forms of
# Bonet et al. (2015), Eq. (24): δv : (F × (M × F)) : δu = (δv × F) : M : (δu × F) and δv : ×ᵢ⁴(Σ) : δu = Σ : (δv × δu).
# Between second-order tensors the product commutes, A × B = B × A; with H it does not, (A × H)ᵀ = Hᵀ × A.
#
# Gridap defines `cross` (that is `LinearAlgebra.cross`) for the vectors only and throws an error for the
# other tensors. The code generated here keeps the nonzero terms of ×ᵢ⁴(A) only (4 out of 9 in each
# contraction), summed in the order of the contraction (column-major). The result is therefore bitwise
# identical to the composition with Gridap's `contracted_product`, e.g. `contracted_product(Val(2), H, ×ᵢ⁴(A))`,
# 3-10× faster and with a negligible compilation time (4 s for that composition).

"Levi-Civita symbol in 3D."
levi_civita(i::Int, j::Int, k::Int) = (i - j) * (j - k) * (k - i) ÷ 2

"""
Component `[i,j,k,l]` of `×ᵢ⁴(A)`: `nothing` if it is zero, or `(s, p, q)` such that `×ᵢ⁴(A)[i,j,k,l] = s A[p,q]`.
"""
function cross_term(i::Int, j::Int, k::Int, l::Int)
  (i == k || j == l) && return nothing
  p, q = 6 - i - k, 6 - j - l
  return levi_civita(i, k, p) * levi_civita(j, l, q), p, q
end

"Expression of the sum of the `terms`, a list of `(sign, expression)`, from left to right."
function signed_sum(terms)
  s, e = first(terms)
  ex = s > 0 ? e : :(-$e)
  for (s, e) in Iterators.drop(terms, 1)
    ex = Expr(:call, s > 0 ? :+ : :-, ex, e)
  end
  return ex
end

"""
    ×ᵢ⁴(A::TensorValue{3,3})::FourthOrderTensorValue{3}

Fourth-order tensor of the cross product with `A`, `×ᵢ⁴(A)[i,j,k,l] = ε[i,k,m] ε[j,l,n] A[m,n]`, such that
`×ᵢ⁴(A) ⋅² B = A × B`. It is the derivative of `A × F` with respect to `F`, e.g. `∂cof(F)/∂F = ×ᵢ⁴(F)`.
"""
@inline @generated function (×ᵢ⁴)(A::TensorValue{3,3,T}) where {T}
  components = map(Iterators.product(1:3, 1:3, 1:3, 1:3)) do (i, j, k, l)
    t = cross_term(i, j, k, l)
    t === nothing && return :(zero($T))
    s, p, q = t
    a = component(:tA, (p, q), 3)
    s > 0 ? a : :(-$a)
  end
  quote
    tA = Tuple(A)
    FourthOrderTensorValue{3,$T}(($(components...),))
  end
end

"""
    cross_contraction(A::TensorValue{3,3}, T)   # ×ᵢ⁴(A) ⋅² T
    contraction_cross(T, A::TensorValue{3,3})   # T ⋅² ×ᵢ⁴(A)

Double contraction of `×ᵢ⁴(A)` with the tensor `T` (of order 2 to 4), without the zero terms of `×ᵢ⁴(A)`.
Only for the tensors with all their components stored: the symmetric ones throw an `ArgumentError`.
"""
@inline @generated function cross_contraction(A::TensorValue{3,3,TA}, B::MultiValue{S,TB}) where {TA,TB,S}
  o = length(size(B))
  all(==(3), size(B)) && o >= 2 || return :(throw(DimensionMismatch("cross_contraction expects 3D tensors of order ≥ 2")))
  components = map(CartesianIndices(ntuple(_ -> 3, o))) do I
    i, j, K = I[1], I[2], Tuple(I)[3:end]
    terms = Tuple{Int,Expr}[]
    for n in 1:3, m in 1:3
      t = cross_term(i, j, m, n)
      t === nothing && continue
      s, p, q = t
      push!(terms, (s, :($(component(:tA, (p, q), 3)) * $(component(:tB, (m, n, K...), 3)))))
    end
    signed_sum(terms)
  end
  quote
    tA, tB = Tuple(A), Tuple(B)
    $(tensor_type(o, 3, promote_type(TA, TB)))(($(components...),))
  end
end

@inline @generated function contraction_cross(B::MultiValue{S,TB}, A::TensorValue{3,3,TA}) where {TA,TB,S}
  o = length(size(B))
  all(==(3), size(B)) && o >= 2 || return :(throw(DimensionMismatch("contraction_cross expects 3D tensors of order ≥ 2")))
  components = map(CartesianIndices(ntuple(_ -> 3, o))) do I
    I_ = Tuple(I)
    F, k, l = I_[1:end-2], I_[end-1], I_[end]
    terms = Tuple{Int,Expr}[]
    for n in 1:3, m in 1:3
      t = cross_term(m, n, k, l)
      t === nothing && continue
      s, p, q = t
      push!(terms, (s, :($(component(:tB, (F..., m, n), 3)) * $(component(:tA, (p, q), 3)))))
    end
    signed_sum(terms)
  end
  quote
    tA, tB = Tuple(A), Tuple(B)
    $(tensor_type(o, 3, promote_type(TA, TB)))(($(components...),))
  end
end

# The generated code indexes all the components: the tensors that store only the independent ones are not supported.
const _SYM_ERROR = "cross products of the symmetric and skew-symmetric tensors are not implemented"
cross_contraction(::TensorValue{3,3}, ::Union{AbstractSymTensorValue,SkewSymTensorValue,SymFourthOrderTensorValue}) = throw(ArgumentError(_SYM_ERROR))
contraction_cross(::Union{AbstractSymTensorValue,SkewSymTensorValue,SymFourthOrderTensorValue}, ::TensorValue{3,3}) = throw(ArgumentError(_SYM_ERROR))

@inline Gridap.TensorValues.cross(A::TensorValue{3,3}, B::TensorValue{3,3}) = cross_contraction(A, B)
@inline Gridap.TensorValues.cross(H::FourthOrderTensorValue{3}, A::TensorValue{3,3}) = contraction_cross(H, A)
@inline Gridap.TensorValues.cross(A::TensorValue{3,3}, H::FourthOrderTensorValue{3}) = cross_contraction(A, H)
@inline Gridap.TensorValues.cross(W::ThirdOrderTensorValue{3,3,3}, A::TensorValue{3,3}) = contraction_cross(W, A)
