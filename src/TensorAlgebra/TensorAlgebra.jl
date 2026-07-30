module TensorAlgebra

using Gridap
using Gridap.TensorValues
using StaticArrays
using LinearAlgebra
import Base: *
import Base: +

export (*)
export (+)
export (⊗₁₂³)
export (⊗₁₃²)
export (⊗₁²³)
export (⊗₁₂³⁴)
export (⊗₁₃²⁴)
export (⊗₁₄²³)
export (⊗₁²)
export ×ᵢ⁴
export IIsym
export I3
export I9
export I2
export I4
export Id
export zerotensor3
export zerotensor9

export logreg
export Tensorize
export δᵢⱼδₖₗ2D
export δᵢₖδⱼₗ2D
export δᵢₗδⱼₖ2D
export δᵢⱼδₖₗ3D
export δᵢₖδⱼₗ3D
export δᵢₗδⱼₖ3D
export δⱼₖδᵢₗ3D
export cof
export contraction_IP_JPKL
export contraction_IP_PJKL
export push_forward_C_to_F

export Box
export Ellipsoid

# outer ⊗ \otimes
# inner ⊙ \odot
# cross ×  \times
# sum   +  +
# dot   ⋅  * 

include("FlatIndexing.jl")

include("FunctionalAlgebra.jl")

include("TensorsDefinitions.jl")

include("Operations.jl")

include("Functions.jl")

end