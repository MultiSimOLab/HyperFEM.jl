# =====================
# Identity tensors
# =====================

"""Second-order identity tensor in dimension `D`, `I[i,j] = δᵢⱼ`."""
_identity2(D::Int) = TensorValue{D,D,Float64}(Tuple(I[1] == I[2] ? 1.0 : 0.0 for I in CartesianIndices((D, D))))

"""Fourth-order identity tensor in dimension `D`, `𝕀[i,j,k,l] = δᵢₖδⱼₗ`."""
_identity4(D::Int) = FourthOrderTensorValue{D,Float64}(
  Tuple(I[1] == I[3] && I[2] == I[4] ? 1.0 : 0.0 for I in CartesianIndices((D, D, D, D))))

"""
    I2D::TensorValue{2,2}

Second-order identity tensor in 2D, `I2D[i,j] = δᵢⱼ`.
"""
const I2D = _identity2(2)

"""
    I3D::TensorValue{3,3}

Second-order identity tensor in 3D, `I3D[i,j] = δᵢⱼ`.
"""
const I3D = _identity2(3)

"""
    𝕀2D::FourthOrderTensorValue{2}

Fourth-order identity tensor in 2D, `𝕀2D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀2D ⊙ A = A`.
"""
const 𝕀2D = _identity4(2)

"""
    𝕀3D::FourthOrderTensorValue{3}

Fourth-order identity tensor in 3D, `𝕀3D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀3D ⊙ A = A`.
"""
const 𝕀3D = _identity4(3)

"""Second-order identity tensor in the dimension of the vector."""
@generated Id(::VectorValue{D,Float64}) where {D} = _identity2(D)


# ==================
# Tensor conversions
# ==================

flat_tensor(a::VectorValue) = TensorValue(Tuple(a))
flat_tensor(a::ThirdOrderTensorValue) = TensorValue(Tuple(a))
flat_tensor(a::FourthOrderTensorValue) = TensorValue(Tuple(a))

flat_smatrix(a::VectorValue) = get_array(flat_tensor(a))
flat_smatrix(a::ThirdOrderTensorValue) = get_array(flat_tensor(a))
flat_smatrix(a::FourthOrderTensorValue) = get_array(flat_tensor(a))

flat_matrix(a::VectorValue) = Matrix(flat_smatrix(a))
flat_matrix(a::ThirdOrderTensorValue) = Matrix(flat_smatrix(a))
flat_matrix(a::FourthOrderTensorValue) = Matrix(flat_smatrix(a))
