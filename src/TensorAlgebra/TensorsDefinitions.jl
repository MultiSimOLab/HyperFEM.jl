
# =====================
# Identity matrix
# =====================

"""The scaling N×N matrix"""
const I_(N) = TensorValue{N,N,Float64}(ntuple(α -> begin
  i,j = _full_idx2(α,N)
  i==j ? 1.0 : 0.0
end,N*N))

"""
    I2::TensorValue{2}

Identity matrix 2D
"""
const I2 = I_(2)

"""
    I3::TensorValue{3}

Identity matrix 3D
"""
const I3 = I_(3)

"""
    I4::TensorValue{4}

Identity fourth-order tensor 2D
"""
const I4 = I_(4)

"""
    I9::TensorValue{9}

Identity fourth-order tensor 3D
"""
const I9 = I_(9)



function Id(::VectorValue{2, Float64})
return I2
end

function Id(::VectorValue{3, Float64})
return I3
end

function Id(::VectorValue{4, Float64})
return I4
end


# =====================
# Zero tensor
# =====================

const zerotensor3 = TensorValue{3,3,Float64}(0,0,0,0,0,0,0,0,0)

const zerotensor9 = TensorValue{9,9,Float64}(
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0)


# =====================
# Delta Kronecker
# =====================

"""
    _Kroneckerδδ(δδ::Function, N::Int)::TensorValue{N*N,N*N,Float64}

Delta Kronecker outer product according to the `δδ(i,j,k,l)` function"""
function _Kroneckerδδ(δδ::Function, N::Int)
  TensorValue{N*N,N*N,Float64}(ntuple(α -> begin
    i, j, k, l = _full_idx4(α,N)
    δδ(i,j,k,l) ? 1.0 : 0.0
  end,
  N*N*N*N))
end

"""
    δᵢⱼδₖₗ2D::TensorValue{4}

Delta Kronecker outer product 2D"""
const δᵢⱼδₖₗ2D = _Kroneckerδδ((i,j,k,l) -> i==j && k==l, 2)

"""
    δᵢₖδⱼₗ2D::TensorValue{4}

Delta Kronecker outer product 2D"""
const δᵢₖδⱼₗ2D = _Kroneckerδδ((i,j,k,l) -> i==k && j==l, 2)

"""
    δᵢₗδⱼₖ2D::TensorValue{4}

Delta Kronecker outer product 2D"""
const δᵢₗδⱼₖ2D = _Kroneckerδδ((i,j,k,l) -> i==l && j==k, 2)

"""
    δᵢⱼδₖₗ3D::TensorValue{9}

Delta Kronecker outer product 3D"""
const δᵢⱼδₖₗ3D = _Kroneckerδδ((i,j,k,l) -> i==j && k==l, 3)

"""
    δᵢₖδⱼₗ3D::TensorValue{9}

Delta Kronecker outer product 3D"""
const δᵢₖδⱼₗ3D = _Kroneckerδδ((i,j,k,l) -> i==k && j==l, 3)

"""
    δᵢₗδⱼₖ3D::TensorValue{9}

Delta Kronecker outer product 3D"""
const δᵢₗδⱼₖ3D = _Kroneckerδδ((i,j,k,l) -> i==l && j==k, 3)
