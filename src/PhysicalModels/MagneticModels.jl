
# ===================
# Magnetic models
# ===================


struct Magnetic{T<:Real} <: Magneto
  μ::Real
  αr::Base.RefValue{T}
  χe::Real
  
  function Magnetic(; μ0::Real, αr::T, χe::Real=0.0) where {T<:Real}
    new{T}(μ0, Ref{T}(αr), χe)
  end
end

function (obj::Magnetic)(Λ::Float64=1.0)
  (; μ, χe) = obj
  ℋᵣ(N) = obj.αr * Λ * N
  # Energy #
  Ψmm(ℋ₀, N) = (-μ / 2.0) * ((ℋ₀ + ℋᵣ(N)) ⋅ (ℋ₀ + ℋᵣ(N))) * (1 + χe)
  ∂Ψmm_∂φ(ℋ₀, N) = (-μ) * (ℋ₀ + ℋᵣ(N)) * (1 + χe)
  ∂Ψmm_∂φφ(ℋ₀, N) = (-μ) * Id(N) * (1 + χe)
  return (Ψmm, ∂Ψmm_∂φ, ∂Ψmm_∂φφ)
end


struct IdealMagnetic <: Magneto
  μ::Real
  χe::Real
  function IdealMagnetic(; μ0::Real, χe::Real=0.0)
    new(μ0, χe)
  end
  function (obj::IdealMagnetic)(Λ::Float64=1.0)

    (; μ, χe) = obj
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
  μ::Real
  χe::Real
  function IdealMagnetic2D(; μ0::Real, χe::Real=0.0)
    new(μ0, χe)
  end

  function (obj::IdealMagnetic2D)(Λ::Float64=1.0)

    (; μ, χe) = obj
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
  μ::Real
  αr::Real
  χe::Real
  χr::Real
  χt::Real
  βmok::Real
  βcoup::Real
  function HardMagnetic(; μ0::Real, αr::Real, χe::Real=0.0, χr::Real=8.0, χt::Union{Real,Nothing}=nothing, βmok::Real=0.0, βcoup::Real=0.0)
    χt_val = isnothing(χt) ? χe : χt
    new(μ0, αr, χe, χr, χt_val, βmok, βcoup)
  end

  function (obj::HardMagnetic)(Λ::Float64=1.0)
    (; μ, χe) = obj
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


struct HardMagnetic2D{T<:Real} <: Magneto
  μ::Real
  αr::Base.RefValue{T}
  χe::Real
  χr::Real
  χt::Real
  βmok::Real
  βcoup::Real
  function HardMagnetic2D(; μ0::Real, αr::T, χe::Real=0.0, χr::Real=8.0, χt::Union{Real,Nothing}=nothing, βmok::Real=0.0, βcoup::Real=0.0) where {T<:Real}
    χt_val = isnothing(χt) ? χe : χt
    new{T}(μ0, Ref{T}(αr), χe, χr, χt_val, βmok, βcoup)
  end

  function (obj::HardMagnetic2D)(Λ::Float64=1.0)

    (; μ, χe) = obj
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


function Base.getproperty(obj::Union{Magnetic,HardMagnetic2D}, prop::Symbol)
  if prop === :αr
    return getfield(obj, :αr)[]
  else
    return getfield(obj, prop)
  end
end


function Base.setproperty!(obj::Union{Magnetic,HardMagnetic2D}, prop::Symbol, val)
  if prop === :αr
    return getfield(obj, :αr)[] = val
  else
    return setfield!(obj, prop, val)
  end
end
