module WeakForms

using Gridap
using HyperFEM.TensorAlgebra
using HyperFEM.PhysicalModels

export residual
export jacobian
export energy
export mass_term
export (+)

import Gridap.Algebra:residual,jacobian

"""
    energy(op, x)

Incremental energy of an operator at the state `x` (see `HyperFEM.Solvers.ModelOperator`); the
methods for the physical models follow.
"""
function energy end
import Base: +

# ===================
# Coupling management
# ===================

function (+)(::Nothing, b::Gridap.CellData.DomainContribution)
b
end
function (+)(b::Gridap.CellData.DomainContribution, ::Nothing)
b
end
function (+)(a::Nothing, ::Nothing)
    a
end

# ===================
# Mechanics
# ===================

"""
    residual(...)::Gridap.CellData.Integrand

Calculate the residual using the given constitutive model and finite element functions.
"""
function residual(physicalmodel::Mechano, km::KinematicModel, u, v, dΩ, Λ=1.0, vars...)
  _, ∂Ψu, _ = physicalmodel()
  F, _, _   = get_Kinematics(km; Λ=Λ)
  ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', vars...)))dΩ
end

"""
    jacobian(...)::Gridap.CellData.Integrand

Calculate the jacobian using the given constitutive model and finite element functions.
"""
function jacobian(physicalmodel::Mechano, km::KinematicModel, u, du, v, dΩ, Λ=1.0, vars...)
  _, _, ∂Ψuu = physicalmodel()
  F, _, _    = get_Kinematics(km; Λ=Λ)
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)', vars...)) ⊙ ∇(du)'))dΩ
end


# ===================
# Mass term
# ===================

function mass_term(u, v, Coeff, dΩ)
  ∫(Coeff* (u⋅v))dΩ
end


# ===================
# ThermoElectroMech
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoElectroMechano, ::Type{Mechano}, kine::NTuple{3,KinematicModel},(u, φ, θ), v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψu=DΨ[2]
  return ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMechano, ::Type{Electro}, kine::NTuple{3,KinematicModel}, (u, φ, θ), vφ, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφ=DΨ[3]
  return -1.0*∫(∇(vφ)' ⋅ (∂Ψφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, kine::NTuple{3,KinematicModel}, (u, φ, θ), vθ, dΩ, Λ=1.0, vars...)
  κ = physicalmodel.thermo.κ
  D, ∂D = Dissipation()
  return ∫(κ * ∇(θ) ⋅ ∇(vθ) -D(u, φ, θ, vars...))dΩ
end

function transient_residual(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, kine::NTuple{3,KinematicModel}, (u, φ, θ), (un, φn, θn), vθ, dΩ, Λ, Δt, vars...)
  DΨ = physicalmodel()
  η = -DΨ[4]
  return ∫((1/Δt)*(θ*η(F,E,θ,vars...) - θn*η(Fn,En,θn,vars...) + η(F,E,θ)*(θ - θn)))dΩ
end


function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Mechano}, kine::NTuple{3,KinematicModel}, (u, φ, θ), du, v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψuu = DΨ[5]
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Electro}, kine::NTuple{3,KinematicModel}, (u, φ, θ), dφ, vφ, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφφ = DΨ[6]
  ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, kine::NTuple{3,KinematicModel}, dθ, vθ, dΩ, Λ=1.0)
  κ = physicalmodel.thermo.κ
  ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, kine::NTuple{3,KinematicModel}, (u, φ, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
  κ = physicalmodel.thermo.κ
  ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ElectroMechano}, kine::NTuple{3,KinematicModel},(u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφu = DΨ[8]
  -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(v)')))dΩ -
  ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ 
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ThermoMechano},kine::NTuple{3,KinematicModel}, (u, φ, θ), dθ, v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψuθ = DΨ[9]
  ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ)dΩ 
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ThermoElectro}, kine::NTuple{3,KinematicModel},(u, φ, θ), dθ, vφ, dΩ, Λ=1.0)
  DΨ= physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφθ = DΨ[10]
  -1.0*∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ))dΩ
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoElectroMechano, kine::NTuple{3,KinematicModel}, (u, φ, θ), (v, vφ, vθ), dΩ, Λ=1.0)
  residual(physicalmodel, Mechano, kine, (u, φ, θ), v, dΩ, Λ) +
  residual(physicalmodel, Electro, kine, (u, φ, θ), vφ, dΩ, Λ) +
  residual(physicalmodel, Thermo, kine,(u, φ, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoElectroMechano, kine::NTuple{3,KinematicModel}, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ), dΩ, Λ=1.0)
  jacobian(physicalmodel, Mechano, kine, (u, φ, θ), du, v, dΩ, Λ) +
  jacobian(physicalmodel, Electro, kine, (u, φ, θ), dφ, vφ, dΩ, Λ) +
  jacobian(physicalmodel, Thermo, kine, dθ, vθ, dΩ, Λ) +
  jacobian(physicalmodel, ElectroMechano, kine, (u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ) +
  jacobian(physicalmodel, ThermoMechano, kine, (u, φ, θ), dθ, v, dΩ, Λ) +
  jacobian(physicalmodel, ThermoElectro, kine, (u, φ, θ), dθ, vφ, dΩ, Λ)
end


# ===================
# ThermoMech
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoMechano, ::Type{Mechano}, kine::NTuple{2,KinematicModel}, (u, θ), v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  ∂Ψu = DΨ[2]
  ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), θ)))dΩ
end

function residual(physicalmodel::ThermoMechano, ::Type{Thermo}, kine::NTuple{2,KinematicModel}, (u, θ), vθ, dΩ, Λ=1.0)
  κ = physicalmodel.thermo.κ
  ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Mechano}, kine::NTuple{2,KinematicModel}, (u, θ), du, v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  ∂Ψuu = DΨ[4]
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Thermo}, kine::NTuple{2,KinematicModel},dθ, vθ, dΩ, Λ=1.0)
  κ = physicalmodel.thermo.κ
  ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Thermo}, kine::NTuple{2,KinematicModel}, (u, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
  κ=physicalmodel.thermo.κ
  ∫((κ ∘ (u, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{ThermoMechano}, kine::NTuple{2,KinematicModel}, (u, θ), (du, dθ), v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  ∂Ψuθ = DΨ[6]
  ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), θ)) * dθ)dΩ 
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoMechano, kine::NTuple{2,KinematicModel},(u, θ), (v, vθ), dΩ, Λ=1.0)
  residual(physicalmodel, Mechano, kine, (u, θ), v, dΩ, Λ) +
  residual(physicalmodel, Thermo, kine, (u, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoMechano, kine::NTuple{2,KinematicModel}, (u, θ), (du, dθ), (v, vθ), dΩ, Λ=1.0)
  jacobian(physicalmodel, Mechano, kine, (u, θ), du, v, dΩ, Λ) +
  jacobian(physicalmodel, Thermo, kine,dθ, vθ, dΩ, Λ) +
  jacobian(physicalmodel, ThermoMechano, kine,(u, θ), (du, dθ), v, dΩ, Λ)
end

# ===================
# ElectroMechanics
# ===================

include("ElectroMechanics.jl")


# =====================
# FlexoElectroMechanics
# =====================

# Stagered strategy
# -----------------
 

function residual(physicalmodel::FlexoElectro, ::Type{Mechano},  kine::NTuple{2,KinematicModel}, (u, φ, ϕ₁, ϕ₂, ϕ₃), v, dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  κ = physicalmodel.κ
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψu = DΨ[2]
  Φ = DΨ[7]
  ∫((∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)',X), E∘(∇(φ)))+κ*(∇(u)'-(Φ(ϕ₁,ϕ₂,ϕ₃))))))dΩ
end

function residual(physicalmodel::FlexoElectro, ::Type{Electro}, kine::NTuple{2,KinematicModel}, (u, φ), vφ, dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφ = DΨ[3]
  -1.0*∫((∇(vφ) ⋅ (∂Ψφ ∘ (F∘(∇(u)',X), E∘(∇(φ))))))dΩ
end

function residual(physicalmodel::FlexoElectro, ::Type{FlexoElectro}, kine::NTuple{2,KinematicModel},(u, ϕ₁, ϕ₂, ϕ₃), (δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  κ = physicalmodel.κ
  Φ = DΨ[7]
  ∫((κ⋅(-∇(u)'+(Φ(ϕ₁,ϕ₂,ϕ₃)))) ⊙ (Φ(δϕ₁,δϕ₂,δϕ₃)))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{Mechano}, kine::NTuple{2,KinematicModel},(u, φ), du, v, dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  κ = physicalmodel.κ
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψuu = DΨ[4]
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(du)')))dΩ+
  ∫(κ⋅(∇(v)'⊙ ∇(du)'))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{Electro}, kine::NTuple{2,KinematicModel},(u, φ), dφ, vφ, dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφφ = DΨ[6]
  ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{ElectroMechano}, kine::NTuple{2,KinematicModel},(u, φ), (du, dφ), (v, vφ), dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφu = DΨ[5]
  -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(v)')))dΩ -
  ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(du)')))dΩ 
end


function jacobian(physicalmodel::FlexoElectro, ::Type{FlexoElectro}, kine::NTuple{2,KinematicModel}, (ϕ₁,ϕ₂,ϕ₃), (du, dϕ₁,dϕ₂,dϕ₃), (v, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ=1.0)
  DΨ = physicalmodel()
  κ = physicalmodel.κ
  Φ = DΨ[7]
  ∫((κ⋅(Φ(δϕ₁,δϕ₂,δϕ₃))) ⊙ ((Φ(dϕ₁,dϕ₂,dϕ₃))))dΩ-
  ∫(∇(v)' ⊙ (κ*(Φ(dϕ₁,dϕ₂,dϕ₃))))dΩ-
  ∫(∇(du)' ⊙ (κ*(Φ(δϕ₁,δϕ₂,δϕ₃))))dΩ
end


# Monolithic strategy
# -------------------

function residual(physicalmodel::FlexoElectro, kine::NTuple{2,KinematicModel},(u, φ, ϕ₁, ϕ₂, ϕ₃), (v, vφ, δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ=1.0)
  residual(physicalmodel, Mechano, kine,(u, φ, ϕ₁, ϕ₂, ϕ₃), v, dΩ, X, Λ) +
  residual(physicalmodel, Electro, kine,(u, φ), vφ, dΩ, X, Λ) +
  residual(physicalmodel, FlexoElectro, kine,(u, ϕ₁, ϕ₂, ϕ₃), (δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ)
end

function jacobian(physicalmodel::FlexoElectro, kine::NTuple{2,KinematicModel},(u, φ, ϕ₁,ϕ₂,ϕ₃), (du, dφ, dϕ₁,dϕ₂,dϕ₃), (v, vφ, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ=1.0)
  jacobian(physicalmodel, Mechano, kine,(u, φ), du, v, dΩ, X, Λ) +
  jacobian(physicalmodel, Electro, kine,(u, φ), dφ, vφ, dΩ, X, Λ) +
  jacobian(physicalmodel, ElectroMechano, kine,(u, φ), (du, dφ), (v, vφ), dΩ, X, Λ) +
  jacobian(physicalmodel, FlexoElectro, kine,(ϕ₁,ϕ₂,ϕ₃), (du, dϕ₁,dϕ₂,dϕ₃), (v, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ)
end


# ===================
# ThermoElectroMech_PINNs
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Mechano}, kine::NTuple{2,KinematicModel}, (u, φ, θ), v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψu = DΨ[2]    
  return ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Electro}, kine::NTuple{2,KinematicModel},(u, φ, θ), vφ, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφ = DΨ[3]
  return -1.0*∫(∇(vφ)' ⋅ (∂Ψφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, kine::NTuple{2,KinematicModel},(u, φ, θ), vθ, dΩ, Λ=1.0)
  κ=physicalmodel.κ
  return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Mechano}, kine::NTuple{2,KinematicModel},(u, φ, θ), du, v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψuu = DΨ[5]
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Electro}, kine::NTuple{2,KinematicModel},(u, φ, θ), dφ, vφ, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφφ = DΨ[6]
  ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, kine::NTuple{2,KinematicModel},dθ, vθ, dΩ, Λ=1.0)
  κ = physicalmodel.κ
  ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, kine::NTuple{2,KinematicModel},(u, φ, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
  κ = physicalmodel.κ
  ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ElectroMechano}, kine::NTuple{2,KinematicModel},(u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφu = DΨ[8]
  -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(v)')))dΩ -
  ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ 
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ThermoMechano}, kine::NTuple{2,KinematicModel}, (u, φ, θ), dθ, v, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψuθ = DΨ[9]
  ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ)dΩ 
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ThermoElectro}, kine::NTuple{2,KinematicModel}, (u, φ, θ), dθ, vφ, dΩ, Λ=1.0)
  DΨ = physicalmodel()
  F, _, _ = get_Kinematics(kine[1]; Λ=Λ)
  E       = get_Kinematics(kine[2]; Λ=Λ)
  ∂Ψφθ = DΨ[10]
  -1.0*∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ))dΩ
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoElectroMech_PINNs,  kine::NTuple{2,KinematicModel},(u, φ, θ), (v, vφ, vθ), dΩ, Λ=1.0)
  residual(physicalmodel, Mechano, kine,(u, φ, θ), v, dΩ, Λ) +
  residual(physicalmodel, Electro, kine,(u, φ, θ), vφ, dΩ, Λ) +
  residual(physicalmodel, Thermo, kine,(u, φ, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, kine::NTuple{2,KinematicModel}, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ),  dΩ, Λ=1.0)
  jacobian(physicalmodel, Mechano, kine,(u, φ, θ), du, v, dΩ, Λ) +
  jacobian(physicalmodel, Electro, kine,(u, φ, θ), dφ, vφ, dΩ, Λ) +
  jacobian(physicalmodel, Thermo, kine,dθ, vθ, dΩ, Λ) +
  jacobian(physicalmodel, ElectroMechano, kine,(u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ) +
  jacobian(physicalmodel, ThermoMechano, kine,(u, φ, θ), dθ, v, dΩ, Λ) +
  jacobian(physicalmodel, ThermoElectro, kine,(u, φ, θ), dθ, vφ, dΩ, Λ)
end

function transient_residual(physicalmodel::ThermoElectroMechModel, ::Type{Thermo}, kine::NTuple{3,KinematicModel}, (u, φ, θ), vθ, dΩ, Λ=1.0)
  κ = physicalmodel.thermo.κ
  return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


end