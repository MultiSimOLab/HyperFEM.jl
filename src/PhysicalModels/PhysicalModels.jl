module PhysicalModels

using Gridap
using Gridap.CellData
using Gridap.Helpers
using ForwardDiff
using LinearAlgebra
using StaticArrays

using ..TensorAlgebra
using ..TensorAlgebra: _∂H∂F_2D
using ..TensorAlgebra: trAA

import Base: +
import Gridap.CellData: CellState, update_state!

export Yeoh3D
export PlaneStressIncompressible_I1PD 
export Gent2D
export NeoHookean3D
export IsochoricNeoHookean3D
export IncompressibleNeoHookean3D
export IncompressibleNeoHookean2D
export IncompressibleNeoHookean2D_CV
export ARAP2D
export ARAP2D_regularized
export NonlinearARAP2D
export VolumetricEnergy
export CoerciveVolumetric
export MooneyRivlin3D
export MooneyRivlin2D
export NonlinearMooneyRivlin3D
export NonlinearMooneyRivlin2D
export NonlinearMooneyRivlin2D_CV
export NonlinearNeoHookean_CV
export NonlinearMooneyRivlin_CV
export NonlinearIncompressibleMooneyRivlin2D_CV
export EightChain
export TransverseIsotropy3D
export TransverseIsotropy2D
export LinearElasticity3D
export LinearElasticity2D
export Magnetic
export IdealDielectric
export IdealMagnetic
export IdealMagnetic2D
export HardMagnetic
export HardMagnetic2D
export ThermalModel
export ThermalVolumetric
export ThermalDeviatoric
export ElectroMechModel
export ThermoElectroMechModel
export ThermoMechModel
export ThermoElectroModel
export ThermoMech_Bonet
export ThermoMech_EntropicPolyconvex
export FlexoElectroModel
export ThermoElectroMech_Bonet
export ThermoElectroMech_Govindjee
export ThermoElectroMech_PINNs
export MagnetoMechModel
export GeneralizedMaxwell
export ViscousIncompressible
export ViscousPolyconvex
export HGO_4Fibers
export HGO_1Fiber

export PhysicalModel
export Mechano
export Elasto
export AnisoElastic
export Visco
export ViscoElastic
export Electro
export Magneto
export Thermo
export ElectroMechano
export MagnetoMechano
export ThermoElectroMechano
export ThermoMechano
export ThermoElectro
export FlexoElectro
export EnergyInterpolationScheme
export SecondPiola
export Dissipation

export CellState      # reexport from Gridap
export update_state!  # reexport form Gridap
export initialize_state
export return_mapping
export update_time_step!

export Kinematics
export KinematicDescription
export Solid
export KinematicModel
export EvolutiveKinematics
export get_Kinematics
export getIsoInvariants

export HessianRegularization
export Hessian∇JRegularization

abstract type PhysicalModel end
abstract type Mechano <: PhysicalModel end
abstract type Electro <: PhysicalModel end
abstract type Magneto <: PhysicalModel end
abstract type Thermo <: PhysicalModel end

abstract type Elasto <: Mechano end
abstract type IsoElastic <: Elasto end
abstract type AnisoElastic <: Elasto end
abstract type Volumetric <: IsoElastic end
abstract type Visco <: Mechano end
abstract type ViscoElastic{E<:Elasto} <: Mechano end

abstract type InternalFibers end
abstract type ThermalLaw end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano{E,M} <: MultiPhysicalModel end
abstract type ThermoElectroMechano{T,E,M} <: MultiPhysicalModel end
abstract type ThermoMechano{T,M} <: MultiPhysicalModel end
abstract type ThermoElectro{E} <: MultiPhysicalModel end
abstract type FlexoElectro{EM} <: MultiPhysicalModel end
abstract type MagnetoMechano{G,M} <: MultiPhysicalModel end

include("KinematicModels.jl")

include("MechanicalModels.jl")

include("ViscousModels.jl")

include("ViscousPolyconvex.jl")

include("MagneticModels.jl")

include("ElectricalModels.jl")

include("ThermalModels.jl")

include("ThermoMechanicalModels.jl")

include("ThermoElectroModels.jl")

include("ElectroMechanicalModels.jl")

include("MagnetoMechanicalModels.jl")

include("ThermoElectroMechanicalModels.jl")

include("PINNs.jl")


# ============================================
# Physical models interface
# ============================================

Base.broadcastable(m::PhysicalModel) = Ref(m) # Allows to use the @. syntax for passing a single constitutive model into a vectorized function

"""
    CellState(model, dΩ)

Initialize the state variables for the given constitutive model and discretization.
The constitutive model passed to the function will determine the type of the state variables,
e.g., a vector, tensor, tuple of state variables...
"""
function Gridap.CellData.CellState(::PhysicalModel, args...)
  return nothing
end


"""
    initialize_state(model)

Define the state variable at a Gauss point. Unlike the function [`CellState`](@ref), the returned
state variable is represented by a number or a tensor.
"""
function initialize_state(::PhysicalModel)
  return nothing
end


"""
    update_state!(model, A, F, Fn)

Update the state variables. The state variables must be initialized using the function
[`CellState`](@ref) with the constitutive model.

NOTE: The Gridap function expects the following order of arguments: `update_state!(updater, cell_states, cell_fields)`,
hence, the order of the arguments differ from the standar energy function, like `Ψ(F, Fn, A...)`
"""
function Gridap.CellData.update_state!(::PhysicalModel, vars...)
end


"""
    return_mapping(model, F, Fn, A...)

Update the state variables at a Gauss point. Unlike the function [`update_state!`](@ref), the state variables are represented by a number or a tensor.
"""
function return_mapping(::PhysicalModel, vars...)
end


"""
    Dissipation(model)

Return the dissipation and its derivatives if any.
"""
function Dissipation(::PhysicalModel, args...)
  D(::Any...) = 0.0
end


"""
    SecondPiola(model)

Return the energy density and its derivatives as functions of C instead of F.
"""
function SecondPiola(::T, args...) where {T<:PhysicalModel}
  throw("The function 'SecondPiola' has not been implemented for $T.")
end


"""
    update_time_step!(model, Δt)

Set the time step to be used internally by the constitutive model.
The time step is a reference, hence, the weak forms derived from the 
constitutive model will be automatically updated with the new time step.
"""
function update_time_step!(::PhysicalModel, Δt::Float64)
  Δt
end

end
