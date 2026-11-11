# =====================
# Identity tensors
# =====================

"""Identity N×N matrix."""
_identity(N::Int) = TensorValue{N,N,Float64}(ntuple(α -> begin
  i, j = _ij(α, N)
  i == j ? 1.0 : 0.0
end, N * N))

"""
    I2D::TensorValue{2,2}

Second-order identity tensor in 2D, `I2D[i,j] = δᵢⱼ`.
"""
const I2D = _identity(2)

"""
    I3D::TensorValue{3,3}

Second-order identity tensor in 3D, `I3D[i,j] = δᵢⱼ`.
"""
const I3D = _identity(3)

"""
    𝕀2D::TensorValue{4,4}

Fourth-order identity tensor in 2D, `𝕀2D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀2D ⊙ A = A`.
"""
const 𝕀2D = _identity(4)

"""
    𝕀3D::TensorValue{9,9}

Fourth-order identity tensor in 3D, `𝕀3D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀3D ⊙ A = A`.
"""
const 𝕀3D = _identity(9)


"""Second-order identity tensor in the dimension of the vector."""
@generated Id(::VectorValue{D,Float64}) where {D} = _identity(D)
