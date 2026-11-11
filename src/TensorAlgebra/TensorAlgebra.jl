module TensorAlgebra

using Gridap
using Gridap.TensorValues
using Gridap.TensorValues: MultiValue, ThirdOrderTensorValue
using StaticArrays
using LinearAlgebra
import Base: *
import Base: +
import Base: -

export (*)
export (+)
export (-)
export (⋅²)  # Gridap's double contraction, extended to the flattened fourth-order tensors

# Products in index notation (see Operations.jl)
export (⊗ᵢʲ), (⊗ᵢⱼᵏ), (⊗ᵢₖʲ), (⊗ᵢʲᵏ), (⊗ᵢⱼₖˡ), (⊗ᵢⱼᵏˡ), (⊗ᵢₖʲˡ), (⊗ᵢₗʲᵏ)
export (⊗ᵢⱼₘᵐ), (⊗ₘᵐⁱʲ), (⊗ᵢⱼₘᵐᵏ), (⊗ᵢₘᵐʲᵏˡ), (⊗ᵢₘʲᵐᵏˡ), (⊗ᵢⱼₘᵐᵏˡ)
export (⊗ᵢₘₙᵐⁿ), (⊗ᵢⱼₘₙᵐⁿ), (⊗ᵢₘⱼₙᵐⁿ), (⊗ₘₙᵐⁿⁱʲ), (⊗ᵢⱼₘₙᵐⁿᵏˡ)

export ×ᵢ⁴
export IIsym
export I2D
export I3D
export 𝕀2D
export 𝕀3D
export Id

export _flat_idx
export _ij
export _ijk
export _ijkl

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
# index notation      ⊗ᵢⱼₘₙᵐⁿᵏˡ  \otimes\_i\_j\_m\_n\^m\^n\^k\^l

include("FlatIndexing.jl")

include("Kernel.jl")

include("FunctionalAlgebra.jl")

include("TensorsDefinitions.jl")

include("Operations.jl")

include("Functions.jl")

include("SymmetricEigen.jl")

include("Hessian.jl")

end
