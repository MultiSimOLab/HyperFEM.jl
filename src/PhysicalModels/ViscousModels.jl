
using ..TensorAlgebra


# =====================
# Visco elastic models
# =====================

struct ViscousIncompressible <: Visco
  elasto::Elasto
  τ::Float64
  Δt::Ref{Float64}
  function ViscousIncompressible(elasto; τ::Float64)
    new(elasto, τ, 0)
  end
  function (obj::ViscousIncompressible)()
    Ψe, Se, ∂Se∂Ce   = SecondPiola(obj.elasto)
    Ψ(F, Fn, A)      = Energy(obj, Ψe, Se, ∂Se∂Ce, F, Fn, A)
    ∂Ψ∂F(F, Fn, A)   = Piola(obj, Se, ∂Se∂Ce, F, Fn, A)
    ∂Ψ∂F∂F(F, Fn, A) = Tangent(obj, Se, ∂Se∂Ce, F, Fn, A)
    return Ψ, ∂Ψ∂F, ∂Ψ∂F∂F
  end
end

function update_time_step!(obj::ViscousIncompressible, Δt::Float64)
  obj.Δt[] = Δt
end

function initialize_state(::ViscousIncompressible, points::Measure)
  v = VectorValue(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0)
  CellState(v, points)
end

function update_state!(obj::ViscousIncompressible, state, F, Fn)
  _, Se, ∂Se∂Ce = SecondPiola(obj.elasto)
  return_mapping(A, F, Fn) = ReturnMapping(obj, Se, ∂Se∂Ce, F, Fn, A)
  update_state!(return_mapping, state, F, Fn)
end

function Dissipation(obj::ViscousIncompressible)
  _, Se, ∂Se∂Ce = SecondPiola(obj.elasto)
  D(F, Fn, A) = ViscousDissipation(obj, Se, ∂Se∂Ce, F, Fn, A)
end

struct NVisco{N} <: Visco 
  branches::NTuple{N,Visco}
end

function NVisco(branches::Vararg{Visco})
  NVisco(branches)
end

function Base.length(::NVisco{N}) where N
  N
end

function Base.getindex(obj::NVisco,i)
  obj.branches[i]
end

function Base.iterate(obj::NVisco, state=0)
  state >= length(obj) && return
  obj[state+1], state+1
end

function (obj::NVisco)()
  DΨv   = map(b -> b(),  obj)
  Ψα    = getindex.(DΨv, 1)
  ∂ΨαF  = getindex.(DΨv, 2)
  ∂ΨαFF = getindex.(DΨv, 3)
  Ψ(F, Fn, A...)     = mapreduce((Ψi, Ai) -> Ψi(F, Fn, Ai), +, Ψα, A; init=0)
  ∂Ψ∂F(F, Fn, A...)  = mapreduce((∂ΨiF, Ai) -> ∂ΨiF(F, Fn, Ai), +, ∂ΨαF, A; init=zerotensor3)
  ∂Ψ∂FF(F, Fn, A...) = mapreduce((∂ΨiFF, Ai) -> ∂ΨiFF(F, Fn, Ai), +, ∂ΨαFF, A; init=zerotensor9)
  (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end

function update_time_step!(obj::NVisco, Δt::Float64)
  foreach(b -> update_time_step!(b, Δt), obj)
  Δt
end

function initialize_state(obj::NVisco, points::Measure)
  map(b -> initialize_state(b, points), obj)
end

function update_state!(obj::NVisco, states, F, Fn)
  @assert length(obj) == length(states)
  map((b, s) -> update_state!(b, s, F, Fn), obj, states)
end

function Dissipation(obj::NVisco)
  Dα = map(Dissipation, obj)
  D(F, Fn, A...) = mapreduce((Di, Ai) -> Di(F, Fn, Ai), +, Dα, A)
end

struct GeneralizedMaxwell{E<:Elasto} <: ViscoElastic{E}
  longterm::E
  branches::NVisco{N} where N
  Δt::Ref{Float64}
  function GeneralizedMaxwell(longTerm::E, branches::Vararg{Visco}) where {E<:Elasto}
    new{E}(longTerm,NVisco(branches),0)
  end
end

function (obj::GeneralizedMaxwell{<:IsoElastic})()
  Ψe, ∂ΨeF, ∂ΨeFF = obj.longterm()
  Ψv, ∂ΨvF, ∂ΨvFF = obj.branches()
  Ψ(F, Fn, A...)     = Ψe(F) + Ψv(F, Fn, A...)
  ∂Ψ∂F(F, Fn, A...)  = ∂ΨeF(F) + ∂ΨvF(F, Fn, A...)
  ∂Ψ∂FF(F, Fn, A...) = ∂ΨeFF(F) + ∂ΨvFF(F, Fn, A...)
  (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end

function (obj::GeneralizedMaxwell{<:AnisoElastic})()
  Ψe, ∂ΨeF, ∂ΨeFF = obj.longterm()
  Ψv, ∂ΨvF, ∂ΨvFF = obj.branches()
  Ψ(F, n, Fn, A...)     = Ψe(F, n) + Ψv(F, Fn, A...)
  ∂Ψ∂F(F, n, Fn, A...)  = ∂ΨeF(F, n) + ∂ΨvF(F, Fn, A...)
  ∂Ψ∂FF(F, n, Fn, A...) = ∂ΨeFF(F, n) + ∂ΨvFF(F, Fn, A...)
  (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end

function update_time_step!(obj::GeneralizedMaxwell, Δt::Float64)
  update_time_step!(obj.longterm, Δt)
  update_time_step!(obj.branches, Δt)
end

function initialize_state(obj::GeneralizedMaxwell, points::Measure)
  initialize_state(obj.branches, points)
end

function update_state!(obj::GeneralizedMaxwell{<:IsoElastic}, states, F, Fn)
  update_state!(obj.branches, states, F, Fn)
end

function update_state!(obj::GeneralizedMaxwell{<:AnisoElastic}, states, F, n, Fn)
  update_state!(obj.branches, states, F, Fn)
end

function Dissipation(obj::GeneralizedMaxwell{<:IsoElastic})
  Dissipation(obj.branches)
end

function Dissipation(obj::GeneralizedMaxwell{<:AnisoElastic})
  Dvis = Dissipation(obj.branches)
  D(F, n, Fn, A...) = Dvis(F, Fn, A...)
end


# =====================
# Internal functions
# =====================


"""Right Cauchy-Green deformation tensor."""
function Cauchy(F::TensorValue)
  F' · F
end


"""Elastic right Cauchy-Green deformation tensor."""
function ElasticCauchy(C::TensorValue, Uv⁻¹::TensorValue)
  Uv⁻¹' · C · Uv⁻¹
end

"""
Multiplicative decomposition of visous strain.

# Return
- `Ue::TensorValue`
- `Uv::TensorValue`
- `Uv⁻¹::TensorValue`
"""
function ViscousStrain(Ce, C)
  Ue = sqrt(Ce)
  Ue_C_Ue = Ue * C * Ue
  invUe = inv(Ue)
  Uv = invUe * sqrt(Ue_C_Ue) * invUe
  invUv = inv(Uv)
  return Ue, Uv, invUv
end


"""
  return_mapping_algorithm!

Compute the elastic Cauchy deformation tensor and the incompressibility condition.

# Arguments
- `obj::ViscousIncompressible`: The viscous model
- `Se_::Function`: Elastic 2nd Piola-Kirchhoff stress (function of C)    
- `∂Se_∂Ce_::Function`: Derivatives of elastic 2nd Piola-Kirchhoff stress (function of C)  
- `F`: Deformation gradient
- `Ce_trial`: Elastic right Green-Cauchy at intermediate statep
- `Ce`: Elastic right Green-Cauchy deformation tensor
- `λα`: incompressibility constraint (Lagrange multiplier)

# Return
- `Ce`
- `λα`
"""
function return_mapping_algorithm!(obj::ViscousIncompressible,
                            Se::Function, ∂Se∂Ce::Function,
                            C, Ce_trial, Ce, λα)
  γα = obj.τ / (obj.τ + obj.Δt[])
  Se_trial = Se(Ce_trial)
  res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), C, λα)
  maxiter = 20
  tol = 1e-6
  for i in 1:maxiter
    #---------- Update -----------#
    local Δu
    try
      Δu = -∂res \ res[:]
    catch e
      if e isa LinearAlgebra.SingularException
        error("Singular jacobian in return mapping algorithm (singular value at pos $(e.info), iteration $i)")
      else
        rethrow()
      end
    end
    Ce += TensorValue{3,3}(Tuple(Δu[1:end-1]))  # TODO: Check reconstruction of TensorValue. ERROR: MethodError: no method matching (TensorValue{3, 3})(::Vector{Float64})
    λα += Δu[end]
    #---- Residual and jacobian ---------#
    res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), C, λα)
    #---- Monitor convergence ---------#
    if norm(res) < tol
      break
    end
  end
  return Ce, λα
end


"""
Residual of the return mapping algorithm and 
its Jacobian with respect to {Ce,λα} for 
incompressible case

# Arguments

# Return
- `res`
- `∂res`
"""
function JacobianReturnMapping(γα, Ce, Se, Se_trial, ∂Se∂Ce, C, λα)
  Ge = cof(Ce)
  #--------------------------------
  # Residual
  #--------------------------------
  res1 = Se - γα * Se_trial - (1-γα) * λα * Ge
  res2 = det(Ce) - det(C)
  #--------------------------------
  # Derivatives of residual
  #--------------------------------
  ∂res1_∂Ce = ∂Se∂Ce - (1-γα) * λα * ×ᵢ⁴(Ce)
  ∂res1_∂λα = -(1-γα) * Ge
  ∂res2_∂Ce = Ge
  res = [get_array(res1)[:]; res2]
  ∂res = MMatrix{10,10}(zeros(10, 10))  # TODO: It'd be nice to use hvcat: ∂res = [∂res1_Ce ∂res1_∂λα; ∂res2_∂Ce 0.0]
  ∂res[1:9, 1:9] = get_array(∂res1_∂Ce)
  ∂res[1:9, 10] = get_array(∂res1_∂λα)[:]
  ∂res[10, 1:9] = (get_array(∂res2_∂Ce)[:])'
  return res, ∂res
end


"""
  ViscousPiola(Se::Function, Ce::SMatrix, invUv::SMatrix, F::SMatrix)::SMatrix

Viscous 1st Piola-Kirchhoff stress

# Arguments
- `Se` Elastic Piola (function of C)
- `Ce` Elastic right Green-Cauchy deformation tensor
- `invUv` Inverse of viscous strain
- `F` Deformation gradient

# Return
- `Pα::SMatrix`
"""
function ViscousPiola(Se::Function, Ce::TensorValue, invUv::TensorValue, F::TensorValue)
  Sα = invUv' * Se(Ce) * invUv
  F * Sα
end


"""
  ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se_∂Ce_, invUvn, Ce, Ce_trial, λα, F)

Tangent operator of Ce for the incompressible case

# Arguments
- `::ViscousIncompressible` The viscous model
- `γα`: Characteristic time τα / (τα + Δt)
- `∂Se∂Ce_`: Function of C
- ...

# Return
- `∂Ce∂C`
"""
function ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
  C = F' * F
  G = cof(C)
  Ge = cof(Ce)
  ∂Se∂Ce = ∂Se∂Ce_(Ce)
  ∂Se∂Ce_trial = ∂Se∂Ce_(Ce_trial)
  ∂Ce_trial_∂C = invUvn ⊗₁₃²⁴ invUvn
  #------------------------------------------
  # Derivative of return mapping with respect to Ce and λα
  #------------------------------------------   
  K11 = ∂Se∂Ce - (1-γα) * λα * ×ᵢ⁴(Ce)
  K12 = -(1-γα) * Ge
  K21 = Ge
  #------------------------------------------
  # Derivative of return mapping with respect to C
  #------------------------------------------   
  F1 = γα * ∂Se∂Ce_trial * ∂Ce_trial_∂C
  F2 = G
  #------------------------------------------
  # Derivative of {Ce,λα} with respect to C
  #------------------------------------------   
  K = MMatrix{10,10}(zeros(10, 10))
  K[1:9, 1:9] = get_array(K11)    # TODO: Check the TensorValue interface
  K[1:9, 10] = get_array(K12)[:]
  K[10, 1:9] = get_array(K21)[:]  # There is no need to transpose the vector
  F = [get_array(F1); (get_array(F2)[:])']
  ∂u∂C = K \ F
  ∂Ce∂C = ∂u∂C[1:9, 1:9]
  return TensorValue(∂Ce∂C)
end


"""
Tangent operator of Ce at fixed Uv
"""
function ∂Ce_∂C_Uvfixed(invUv)
  invUv ⊗₁₃²⁴ invUv
end


"""
∂Ce∂(Uv^{-1})
"""
function ∂Ce_∂invUv(C, invU)
  invU_C = invU * C
  invU_C ⊗₁₃²⁴ I3 + I3 ⊗₁₃²⁴ invU_C
end


"""
  ViscousTangentOperator::TensorValue

Tangent operator for the incompressible case

# Arguments
- `obj::ViscousIncompressible`
- `Se_::Function`: Function of C
- `∂Se∂Ce_::Function`: Function of C
- `F::TensorValue`: Deformation tensor
- `Ce_trial`: Right Green-Cauchy deformation tensor at intermediate step
- `Ce`: Right Green-Cauchy deformation tensor at curent step
- `invUv`
- `invUvn`
- `λα`

# Return
- `Cv::TensorValue{9,9}`: A fourth-order tensor in flattened notation
"""
function ViscousTangentOperator(obj::ViscousIncompressible,
                  Se_::Function, ∂Se∂Ce_::Function,
                  F::TensorValue, Ce_trial, Ce, invUv, invUvn, λα)
  # -----------------------------------------
  # Characteristic time
  #------------------------------------------
  γα = obj.τ / (obj.τ + obj.Δt[])
  #------------------------------------------
  # Elastic tensor and derivatives
  #------------------------------------------
  C = Cauchy(F)
  DCe_DC = ∂Ce_∂C(obj, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
  DCe_DC_Uvfixed = ∂Ce_∂C_Uvfixed(invUv)
  DCe_DinvUv = ∂Ce_∂invUv(C, invUv)
  DinvUv_DC = inv(DCe_DinvUv) * (DCe_DC - DCe_DC_Uvfixed)
  DCDF = F' ⊗₁₃²⁴ I3 + I3 ⊗₁₄²³ F'
  #------------------------------------------
  # 0.5*δC_{Uvfixed}:DSe[ΔC]
  #------------------------------------------
  C1 = 0.5 * DCe_DC_Uvfixed' * ∂Se∂Ce_(Ce) * DCe_DC
  #------------------------------------------
  # Se:0.5*(DUv^{-1}[ΔC]*δC*Uv^{-1} + Uv^{-1}*δC*DUv^{-1}[ΔC])
  #------------------------------------------
  invUv_Se = invUv * Se_(Ce)
  C2 = 0.5 * (contraction_IP_JPKL(invUv_Se, DinvUv_DC) +
              contraction_IP_PJKL(invUv_Se, DinvUv_DC))
  #------------------------------------------
  # Sv:(D(δC_{Uvfixed})[ΔC])
  #------------------------------------------
  Sv = invUv_Se * invUv
  C3 = I3 ⊗₁₃²⁴ Sv
  #------------------------------------------
  # Total Contribution
  #------------------------------------------
  Cv = DCDF' * (C1 + C2) * DCDF + C3
  Cv
end


function Energy(obj::ViscousIncompressible,
                Ψe::Function, Se_::Function, ∂Se∂Ce_::Function,
                F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Elastic energy
  #------------------------------------------
  Ψe(Ce)
end


"""
  First Piola-Kirchhoff for the incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `F`: Current deformation gradient
- `Fn`: Previous deformation gradient
- `A`: State variables (Uvα and λα)

# Return
- `Pα::Gridap.TensorValues.TensorValue`
"""
function Piola(obj::ViscousIncompressible,
                Se_::Function, ∂Se∂Ce_::Function,
                F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Pα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  Pα = ViscousPiola(Se_, Ce, invUv, F)
  Pα
end


"""
Visco-Elastic model for incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `∇u_`: Current deformation gradient
- `∇un_`: Previous deformation gradient
- `A`: State variables (Uvα and λα)

# Return
- `Cα::Gridap.TensorValues.TensorValue`
"""
function Tangent(obj::ViscousIncompressible,
                 Se_::Function, ∂Se∂Ce_::Function,
                 F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  Cα = ViscousTangentOperator(obj, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Ce, invUv, invUvn, λα)
  return Cα
end


"""
    Return mapping for the incompressible case

    # Arguments
    - `::ViscousIncompressible`
    - `Se_::Function`: Elastic Piola (function of C)
    - `∂Se∂Ce_::Function`: Piola Derivatives (function of C)
    - `∇u_::TensorValue`
    - `∇un_::TensorValue`
    - `A::VectorValue`: State variables (10-component vector gathering Uvα and λα)

    # Return
    - `::bool`: indicates whether the state variables should be updated
    - `::VectorValue`: State variables at new time step
"""
function ReturnMapping(obj::ViscousIncompressible,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get Uv and λα
  #------------------------------------------
  _, Uv, _ = ViscousStrain(Ce, C)
  Cell_ = [get_array(Uv)[:]; λα]  # TODO: Another problem with TensorValue slice
  return true, VectorValue(Cell_)
end


function ViscousDissipation(obj::ViscousIncompressible,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Dissipation
  #------------------------------------------
  τ = obj.τ
  Se = Se_(Ce)
  Ge = cof(Ce)
  ∂Se∂Ce = ∂Se∂Ce_(Ce)
  α = 1.e3abs(tr(∂Se∂Ce))  # Ensure invertibility of the elasticity tensor.
  invCCe = inv(2*∂Se∂Ce + α*Ge⊗Ge)
  ∂Se = -1/τ * (Se - λα*Ge)
  Dvis = -Se ⊙ (invCCe ⊙ ∂Se)
  Dvis
end
