
# ===============================
# Magneto mechanical models
# ===============================

struct MagnetoMechModel{G<:Magneto,M<:Mechano} <: MagnetoMechano{G,M}
  magneto::G
  mechano::M

  function MagnetoMechModel(magneto::G, mechano::M) where {G <: Magneto, M <: Mechano}
    new{G,M}(magneto, mechano)
  end
  
  function MagnetoMechModel(; magneto::G, mechano::M) where {G <: Magneto, M <: Mechano}
    new{G,M}(magneto, mechano)
  end

  function (obj::MagnetoMechModel{<:Magneto,<:IsoElastic})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψmm, ∂Ψmm_u, ∂Ψmm_φ, ∂Ψmm_uu, ∂Ψmm_φu, ∂Ψmm_φφ = _getCoupling(obj.magneto, obj.mechano, Λ)

    Ψ(F, ℋ₀, N) = Ψm(F) + Ψmm(F, ℋ₀, N)
    ∂Ψu(F, ℋ₀, N) = ∂Ψm_u(F) + ∂Ψmm_u(F, ℋ₀, N)
    ∂Ψφ(F, ℋ₀, N) = ∂Ψmm_φ(F, ℋ₀, N)
    ∂Ψuu(F, ℋ₀, N) = ∂Ψm_uu(F) + ∂Ψmm_uu(F, ℋ₀, N)
    ∂Ψφu(F, ℋ₀, N) = ∂Ψmm_φu(F, ℋ₀, N)
    ∂Ψφφ(F, ℋ₀, N) = ∂Ψmm_φφ(F, ℋ₀, N)

    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end

    function (obj::MagnetoMechModel{<:Magneto,<:AnisoElastic})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψmm, ∂Ψmm_u, ∂Ψmm_φ, ∂Ψmm_uu, ∂Ψmm_φu, ∂Ψmm_φφ = _getCoupling(obj.magneto, obj.mechano, Λ)

    Ψ(F, ℋ₀, N) = Ψm(F,N) + Ψmm(F, ℋ₀, N)
    ∂Ψu(F, ℋ₀, N) = ∂Ψm_u(F,N) + ∂Ψmm_u(F, ℋ₀, N)
    ∂Ψφ(F, ℋ₀, N) = ∂Ψmm_φ(F, ℋ₀, N)
    ∂Ψuu(F, ℋ₀, N) = ∂Ψm_uu(F,N) + ∂Ψmm_uu(F, ℋ₀, N)
    ∂Ψφu(F, ℋ₀, N) = ∂Ψmm_φu(F, ℋ₀, N)
    ∂Ψφφ(F, ℋ₀, N) = ∂Ψmm_φφ(F, ℋ₀, N)

    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end


end



function _getCoupling(mag::Union{IdealMagnetic,IdealMagnetic2D}, ::Mechano, Λ::Float64)
  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)

  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)
end


function _getCoupling(mag::HardMagnetic, ::Mechano, Λ::Float64=1.0)

  # Miguel Angel Moreno-Mateos, Mokarram Hossain, Paul Steinmann, Daniel Garcia-Gonzalez,
  # Hard magnetics in ultra-soft magnetorheological elastomers enhance fracture toughness and 
  # delay crack propagation, Journal of the Mechanics and Physics of Solids,


  μ, αr, χr, χt, βcoup, βmok = mag.μ, mag.αr, mag.χr, mag.χt, mag.βcoup, mag.βmok
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
 

  #-------------------------------------------------------------------------------------
  # FIRST TERM
  #-------------------------------------------------------------------------------------
  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)

  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)

  ℋᵣ(N) = αr * Λ* N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I3 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
  ∂Ψcoup_∂uu(F, N) = ∂Ψcoup_∂JF(F, N) + ∂Ψcoup_∂FF(F, N) + ∂Ψcoup_∂J(F, N) * ×ᵢ⁴(F)

  #-------------------------------------------------------------------------------------
  # THIRD TERM
  #-------------------------------------------------------------------------------------

  Ψmok(F, N) = (0.5 * μ * J(F) / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψmok_∂u(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * H(F)
  ∂Ψmok_∂uu(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * ×ᵢ⁴(F)

  #-------------------------------------------------------------------------------------
  # FOURTH TERM
  #-------------------------------------------------------------------------------------
  Hℋᵣ(F, N) = H(F) * ℋᵣ(N)
  Ψtorq(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂H(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂J(F, ℋ₀, N) = -(μ * (1 + χt) / J(F)^2) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂u(F, ℋ₀, N) = ∂Ψtorq_∂H(F, ℋ₀, N) × F + ∂Ψtorq_∂J(F, ℋ₀, N) * H(F)
  ∂Ψtorq_∂φ(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (H(F)' * Hℋᵣ(F, N))

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (I3 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χt) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χt) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = (F × (∂Ψtorq_∂HH(F, ℋ₀, N) × F)) +
                         H(F) ⊗₁₂³⁴ (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) +
                         (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) ⊗₁₂³⁴ H(F) +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗₁₂³⁴ H(F)) +
                         ×ᵢ⁴(∂Ψtorq_∂H(F, ℋ₀, N) + ∂Ψtorq_∂J(F, ℋ₀, N) * F)


  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I3 ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χt)
  ∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) = (-μ / (J(F))^2.0) * (H(F)' * Hℋᵣ(F, N)) * (1 + χt)

  ∂Ψtorq_∂φu(F, ℋ₀, N) = (∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) × F) + (∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) ⊗₁²³ H(F))

  #-------------------------------------------------------------------------------------
  #                           TOTAL ENERGY
  #-------------------------------------------------------------------------------------
  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀) + βcoup * Ψcoup(F, N) + βmok * Ψmok(F, N) + Ψtorq(F, ℋ₀, N)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀) + βcoup * ∂Ψcoup_∂u(F, N) + βmok * ∂Ψmok_∂u(F, N) + ∂Ψtorq_∂u(F, ℋ₀, N)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀) + ∂Ψtorq_∂φ(F, ℋ₀, N)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀) + βcoup * ∂Ψcoup_∂uu(F, N) + βmok * ∂Ψmok_∂uu(F, N) + ∂Ψtorq_∂uu(F, ℋ₀, N)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀) + ∂Ψtorq_∂φu(F, ℋ₀, N)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)
end


function _getCoupling(mag::HardMagnetic2D, ::Mechano, Λ::Float64=1.0)

  # Miguel Angel Moreno-Mateos, Mokarram Hossain, Paul Steinmann, Daniel Garcia-Gonzalez,
  # Hard magnetics in ultra-soft magnetorheological elastomers enhance fracture toughness and 
  # delay crack propagation, Journal of the Mechanics and Physics of Solids,

  μ, αr, χr, χt, βcoup, βmok = mag.μ, mag.αr, mag.χr, mag.χt, mag.βcoup, mag.βmok
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
 
  # #-------------------------------------------------------------------------------------
  # # FIRST TERM
  # #-------------------------------------------------------------------------------------
  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)

  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  ℋᵣ(N) = αr[] * Λ* N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I2 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
  ∂Ψcoup_∂uu(F, N) = ∂Ψcoup_∂JF(F, N) + ∂Ψcoup_∂FF(F, N) + ∂Ψcoup_∂J(F, N) * _∂H∂F_2D()

  #-------------------------------------------------------------------------------------
  # THIRD TERM
  #-------------------------------------------------------------------------------------

  Ψmok(F, N) = (0.5 * μ * J(F) / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψmok_∂u(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * H(F)
  ∂Ψmok_∂uu(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * _∂H∂F_2D()

  #-------------------------------------------------------------------------------------
  # FOURTH TERM
  #-------------------------------------------------------------------------------------
  Hℋᵣ(F, N) = H(F) * ℋᵣ(N)
  Ψtorq(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂H(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂J(F, ℋ₀, N) = -(μ * (1 + χt) / J(F)^2) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂u(F, ℋ₀, N) = (tr(∂Ψtorq_∂H(F, ℋ₀, N)) * I2) - ∂Ψtorq_∂H(F, ℋ₀, N)' + ∂Ψtorq_∂J(F, ℋ₀, N) * H(F)
  ∂Ψtorq_∂φ(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (H(F)' * Hℋᵣ(F, N))

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χt) / J(F)) * (I2 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χt) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χt) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = _∂H∂F_2D()' * ∂Ψtorq_∂HH(F, ℋ₀, N) * _∂H∂F_2D() +
                         _∂H∂F_2D()' * (∂Ψtorq_∂HJ(F, ℋ₀, N) ⊗ H(F)) +
                         (H(F) ⊗ ∂Ψtorq_∂HJ(F, ℋ₀, N)) * _∂H∂F_2D() +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗ H(F)) +
                         ∂Ψtorq_∂J(F, ℋ₀, N) * _∂H∂F_2D()

  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I2 ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χt)
  ∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) = (-μ / (J(F))^2.0) * (H(F)' * Hℋᵣ(F, N)) * (1 + χt)


  ∂Ψtorq_∂φu(F, ℋ₀, N) = (∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) * _∂H∂F_2D()) + (∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) ⊗₁²³ H(F))

  #-------------------------------------------------------------------------------------
  #                           TOTAL ENERGY
  #-------------------------------------------------------------------------------------
  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀) + βcoup * Ψcoup(F, N) + βmok * Ψmok(F, N) + Ψtorq(F, ℋ₀, N)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀) + βcoup * ∂Ψcoup_∂u(F, N) + βmok * ∂Ψmok_∂u(F, N) + ∂Ψtorq_∂u(F, ℋ₀, N)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀) + ∂Ψtorq_∂φ(F, ℋ₀, N)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀) + βcoup * ∂Ψcoup_∂uu(F, N) + βmok * ∂Ψmok_∂uu(F, N) + ∂Ψtorq_∂uu(F, ℋ₀, N)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀) + ∂Ψtorq_∂φu(F, ℋ₀, N)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)
end


function (+)(Model1::Magneto, Model2::Mechano)
  MagnetoMechModel(Model1, Model2)
end
function (+)(Model1::Mechano, Model2::Magneto)
  MagnetoMechModel(Model2, Model1)
end