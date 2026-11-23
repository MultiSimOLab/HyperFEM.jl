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

Fourth-order identity tensor in 2D, `𝕀2D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀2D ⋅² A = A`.
"""
const 𝕀2D = _identity4(2)

"""
    𝕀3D::FourthOrderTensorValue{3}

Fourth-order identity tensor in 3D, `𝕀3D[i,j,k,l] = δᵢₖδⱼₗ`, such that `𝕀3D ⋅² A = A`.
"""
const 𝕀3D = _identity4(3)

"""Second-order identity tensor in the dimension of the vector."""
@generated Id(::VectorValue{D,Float64}) where {D} = _identity2(D)


# ==================
# Tensor conversions
# ==================

const _FlattenableTensor = Union{ThirdOrderTensorValue,FourthOrderTensorValue}

"""
    flat_tensor(W::ThirdOrderTensorValue{D1,D2,D3})::TensorValue{D1,D2*D3}
    flat_tensor(H::FourthOrderTensorValue{D})::TensorValue{D²,D²}

Matrix representation of a tensor:
- ThirdOrderTensorValue: D×D×D -> D×D²
- FourthOrderTensorValue: D×D×D×D -> D²×D²
"""
flat_tensor(a::ThirdOrderTensorValue{D1,D2,D3}) where {D1,D2,D3} = TensorValue{D1,D2 * D3}(Tuple(a))
flat_tensor(a::FourthOrderTensorValue) = TensorValue(Tuple(a))

"""
    flat_smatrix(a)::SMatrix

Matrix representation of `a` as an `SMatrix`, see [`flat_tensor`](@ref).
"""
flat_smatrix(a::_FlattenableTensor) = get_array(flat_tensor(a))

"""
    flat_matrix(a)::Matrix

Matrix representation of `a` as a `Matrix`, see [`flat_tensor`](@ref).
"""
flat_matrix(a::_FlattenableTensor) = Matrix(flat_smatrix(a))
