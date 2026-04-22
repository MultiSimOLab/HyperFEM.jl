
# ===================
# Thermal models
# ===================

struct ThermalModel <: Thermo
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64
  function ThermalModel(; Cv::Float64, θr::Float64, α::Float64, κ::Float64=10.0)
    new(Cv, θr, α, κ)
  end

  function (obj::ThermalModel)(Λ::Float64=1.0)
    Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
    ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
    ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
    return (Ψ, ∂Ψθ, ∂Ψθθ)
  end
end

struct ThermalVolumetric{T<:Thermo} <: ThermoMechano{T,Volumetric}
  thermo::T
  mechano::Volumetric
  law::ThermalLaw

  function ThermalVolumetric(energy; cv0, θr, α, γ, κ=1.0)
    thermo = ThermalModel(Cv=cv0, θr=θr, α=α, κ=κ)
    law = EntropicElasticityLaw(θr=θr, γ=γ)
    new{ThermalModel}(thermo, energy, law)
  end

  function ThermalVolumetric(; cv0, θr, α, κr, γ, κ=1.0)
    thermo = ThermalModel(Cv=cv0, θr=θr, α=α, κ=κ)
    law = EntropicElasticityLaw(θr=θr, γ=γ)
    energy = VolumetricEnergy(λ=κr)
    new{ThermalModel}(thermo, energy, law)
  end
end

function (obj::ThermalVolumetric)()
  @unpack Cv, θr, α, κ = obj.thermo
  cv0 = Cv  # FIXME
  U, ∂U∂F, ∂∂U∂FF = obj.mechano()
  κr = tangent(obj.mechano)
  f, df, ddf = derivatives(obj.law)
  ζr = 1/df(θr)
  ξr = 1/(θr*ζr*ddf(θr))
  J(F) = det(F)
  H(F) = cof(F)
  ηr(F) = cv0*ξr + 3*α*κr*(J(F) - 1)
  ∂ηr∂J(F) = 3*α*κr
  ∂ηr∂F(F) = ∂ηr∂J(F)*H(F)
  ∂∂ηr∂FF(F) = ×ᵢ⁴(∂ηr∂J(F)*F)
  Ψ(F,θ)      = U(F)      -ηr(F)*ζr*(f(θ) - 1)
  ∂Ψ∂F(F,θ)   = ∂U∂F(F)   -∂ηr∂F(F)*ζr*(f(θ) - 1)
  ∂∂Ψ∂FF(F,θ) = ∂∂U∂FF(F) -∂∂ηr∂FF(F)*ζr*(f(θ) - 1)
  ∂Ψ∂θ(F,θ)   =           -ηr(F)*ζr*df(θ)
  ∂∂Ψ∂θθ(F,θ) =           -ηr(F)*ζr*ddf(θ)
  ∂∂Ψ∂Fθ(F,θ) =           -∂ηr∂F(F)*ζr*df(θ)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end
