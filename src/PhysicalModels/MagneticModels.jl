
# ===================
# Magnetic models
# ===================


struct Magnetic <: Magneto
  μ::Float64
  αr::Ref{Float64}
  χe::Float64
  
function Magnetic(; μ0::Float64, αr::Float64, χe::Float64=0.0)
  new(μ0, Ref(αr), χe)
end
function (obj::Magnetic)(Λ::Float64=1.0)
  μ, αr, χe = obj.μ, obj.αr, obj.χe
  ℋᵣ(N) = αr[] * Λ * N
  # Energy #
  Ψmm(ℋ₀, N) = (-μ / 2.0) * ((ℋ₀ + ℋᵣ(N)) ⋅ (ℋ₀ + ℋᵣ(N))) * (1 + χe)
  ∂Ψmm_∂φ(ℋ₀, N) = (-μ) * (ℋ₀ + ℋᵣ(N)) * (1 + χe)
  ∂Ψmm_∂φφ(ℋ₀, N) = (-μ) * Id(N) * (1 + χe)
  return (Ψmm, ∂Ψmm_∂φ, ∂Ψmm_∂φφ)
end


end


struct IdealMagnetic <: Magneto
  μ::Float64
  χe::Float64
  function IdealMagnetic(; μ0::Float64, χe::Float64=0.0)
    new(μ0, χe)
  end
  function (obj::IdealMagnetic)(Λ::Float64=1.0)

    μ, χe = obj.μ, obj.χe
    J(F) = det(F)
    H(F) = J(F) * inv(F)'

    # Energy #
    Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
    Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
    Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

    # First Derivatives #
    ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂u(F, ℋ₀) = ∂Ψmm_∂H(F, ℋ₀) × F + ∂Ψmm_∂J(F, ℋ₀) * H(F)
    ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

    # Second Derivatives #
    ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I3 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂uu(F, ℋ₀) = (F × (∂Ψmm_∂HH(F, ℋ₀) × F)) +
                      H(F) ⊗₁₂³⁴ (∂Ψmm_∂HJ(F, ℋ₀) × F) +
                      (∂Ψmm_∂HJ(F, ℋ₀) × F) ⊗₁₂³⁴ H(F) +
                      ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗₁₂³⁴ H(F)) +
                      ×ᵢ⁴(∂Ψmm_∂H(F, ℋ₀) + ∂Ψmm_∂J(F, ℋ₀) * F)

    ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I3 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)

    ∂Ψmm_∂φu(F, ℋ₀) = (∂Ψmm_∂ℋ₀H(F, ℋ₀) × F) + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
    ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)

    return (Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ)
  end
end


struct IdealMagnetic2D <: Magneto
  μ::Float64
  χe::Float64
  function IdealMagnetic2D(; μ0::Float64, χe::Float64=0.0)
    new(μ0, χe)
  end

  function (obj::IdealMagnetic2D)(Λ::Float64=1.0)

    μ, χe = obj.μ, obj.χe
    J(F) = det(F)
    H(F) = J(F) * inv(F)'

    # Energy #
    Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
    Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
    Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

    # First Derivatives #
    ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂u(F, ℋ₀) = (tr(∂Ψmm_∂H(F, ℋ₀)) * I2) - ∂Ψmm_∂H(F, ℋ₀)' + ∂Ψmm_∂J(F, ℋ₀) * H(F)
    ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

    # Second Derivatives #
    ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I2 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂uu(F, ℋ₀) = _∂H∂F_2D()' * ∂Ψmm_∂HH(F, ℋ₀) * _∂H∂F_2D() + _∂H∂F_2D()' * (∂Ψmm_∂HJ(F, ℋ₀) ⊗ H(F)) +
                      (H(F) ⊗ ∂Ψmm_∂HJ(F, ℋ₀)) * _∂H∂F_2D() + ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗ H(F)) + ∂Ψmm_∂J(F, ℋ₀) * _∂H∂F_2D()

    ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I2 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂φu(F, ℋ₀) = ∂Ψmm_∂ℋ₀H(F, ℋ₀) * _∂H∂F_2D() + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
    ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)
    return (Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ)
  end
end


struct HardMagnetic <: Magneto
  μ::Float64
  αr::Float64
  χe::Float64
  χr::Float64
  χt::Float64
  βmok::Float64
  βcoup::Float64
  function HardMagnetic(; μ0::Float64, αr::Float64, χe::Float64=0.0, χr::Float64=8.0, χt::Union{Float64,Nothing}=nothing, βmok::Float64=0.0, βcoup::Float64=0.0)
    χt_val = isnothing(χt) ? χe : χt
    new(μ0, αr, χe, χr, χt_val, βmok, βcoup)
  end

  function (obj::HardMagnetic)(Λ::Float64=1.0)
    μ, χe = obj.μ, obj.χe
    J(F) = det(F)
    H(F) = J(F) * inv(F)'

    # Energy #
    Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
    Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
    Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

    # First Derivatives #
    ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂u(F, ℋ₀) = ∂Ψmm_∂H(F, ℋ₀) × F + ∂Ψmm_∂J(F, ℋ₀) * H(F)
    ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

    # Second Derivatives #
    ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I3 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂uu(F, ℋ₀) = (F × (∂Ψmm_∂HH(F, ℋ₀) × F)) +
                      H(F) ⊗₁₂³⁴ (∂Ψmm_∂HJ(F, ℋ₀) × F) +
                      (∂Ψmm_∂HJ(F, ℋ₀) × F) ⊗₁₂³⁴ H(F) +
                      ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗₁₂³⁴ H(F)) +
                      ×ᵢ⁴(∂Ψmm_∂H(F, ℋ₀) + ∂Ψmm_∂J(F, ℋ₀) * F)

    ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I3 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)

    ∂Ψmm_∂φu(F, ℋ₀) = (∂Ψmm_∂ℋ₀H(F, ℋ₀) × F) + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
    ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)
    return (Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ)
  end
end


struct HardMagnetic2D <: Magneto
  μ::Float64
  αr::Ref{Float64}
  χe::Float64
  χr::Float64
  χt::Float64
  βmok::Float64
  βcoup::Float64
  function HardMagnetic2D(; μ0::Float64, αr::Float64, χe::Float64=0.0, χr::Float64=8.0, χt::Union{Float64,Nothing}=nothing, βmok::Float64=0.0, βcoup::Float64=0.0)
    χt_val = isnothing(χt) ? χe : χt
    new(μ0, Ref(αr), χe, χr, χt_val, βmok, βcoup)
  end

  function (obj::HardMagnetic2D)(Λ::Float64=1.0)

    μ, χe = obj.μ, obj.χe
    J(F) = det(F)
    H(F) = J(F) * inv(F)'

    # Energy #
    Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
    Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
    Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

    # First Derivatives #
    ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂u(F, ℋ₀) = (tr(∂Ψmm_∂H(F, ℋ₀)) * I2) - ∂Ψmm_∂H(F, ℋ₀)' + ∂Ψmm_∂J(F, ℋ₀) * H(F)
    ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

    # Second Derivatives #
    ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I2 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂uu(F, ℋ₀) = _∂H∂F_2D()' * ∂Ψmm_∂HH(F, ℋ₀) * _∂H∂F_2D() + _∂H∂F_2D()' * (∂Ψmm_∂HJ(F, ℋ₀) ⊗ H(F)) +
                      (H(F) ⊗ ∂Ψmm_∂HJ(F, ℋ₀)) * _∂H∂F_2D() + ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗ H(F)) + ∂Ψmm_∂J(F, ℋ₀) * _∂H∂F_2D()


    ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I2 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂φu(F, ℋ₀) = ∂Ψmm_∂ℋ₀H(F, ℋ₀) * _∂H∂F_2D() + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
    ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)
    return (Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ)
  end
end
