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

# Former names, used by the benchmark scripts of `main` (temporary, see the end of Operations.jl)
Base.@deprecate_binding I3 I3D
Base.@deprecate_binding I4 TensorValue(Tuple(𝕀2D))   # the D²×D² identity matrix, as the former I4
Base.@deprecate_binding I9 TensorValue(Tuple(𝕀3D))   # the D²×D² identity matrix, as the former I9
