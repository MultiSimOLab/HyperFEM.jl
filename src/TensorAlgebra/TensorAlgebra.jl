module TensorAlgebra

using Gridap
using Gridap.TensorValues
using Gridap.TensorValues: MultiValue, ThirdOrderTensorValue, HighOrderTensorValue
using StaticArrays
using LinearAlgebra
import Base: *
import Base: +
import Base: -

"""
    FourthOrderTensorValue{D,T,L}
    FourthOrderTensorValue(data::Tuple)
    FourthOrderTensorValue(data::Number...)

Fourth-order tensor in dimension `D`, with full indices `H[i,j,k,l]` (Gridap's `HighOrderTensorValue`).

The components are stored in column-major order, as in its D²×D² matrix (rows `(i,j)`, columns `(k,l)`):

    M = TensorValue(Tuple(H))              # D²×D² matrix, e.g. for `inv` or a linear system
    H = FourthOrderTensorValue(Tuple(M))   # from a D²×D² TensorValue, SMatrix or Matrix

Pass the components as a tuple: splatting 81 components is not type-stable.
"""
const FourthOrderTensorValue{D,T,L} = HighOrderTensorValue{Tuple{D,D,D,D},T,4,L}

FourthOrderTensorValue(data::Tuple) = HighOrderTensorValue(Val(4), data)
FourthOrderTensorValue(data::Number...) = HighOrderTensorValue(Val(4), data)

export (*)
export (+)
export (-)
export (⋅²)  # Gridap's double contraction
export ThirdOrderTensorValue
export FourthOrderTensorValue

# Products in index notation (see Operations.jl)
export (⊗ᵢₖʲ), (⊗ᵢₖʲˡ), (⊗ᵢₗʲᵏ), (⊗ᵢₘʲᵐᵏˡ)

export ×ᵢ⁴
export IIsym
export I2D
export I3D
export 𝕀2D
export 𝕀3D
export Id

export logreg
export ∂log∂J
export ∂∂log∂JJ
export Tensorize
export cof
export push_forward_C_to_F
export regularize_hessian
export hessian_nonaffine

export Box
export Ellipsoid

# outer               ⊗   \otimes
# single contraction  ⋅   \cdot
# double contraction  ⋅²  \cdot\^2
# full contraction    ⊙   \odot
# cross               ×   \times
# index notation      ⊗ᵢₖʲˡ  \otimes\_i\_k\^j\^l

include("Kernel.jl")

include("FunctionalAlgebra.jl")

include("TensorsDefinitions.jl")

include("Operations.jl")

include("Functions.jl")

include("SymmetricEigen.jl")

include("Hessian.jl")

end
