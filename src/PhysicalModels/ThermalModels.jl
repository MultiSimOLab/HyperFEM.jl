
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
end

function (obj::ThermalModel)(Λ::Float64=1.0)
  Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
  ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
  ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
  return (Ψ, ∂Ψθ, ∂Ψθθ)
end


# ===================
# Thermal laws
# ===================

struct ConstantEnergyLaw <: ThermalLaw end

function (law::ConstantEnergyLaw)()
  f(θ) = 1.0
  ∂f(θ) = 0.0
  ∂∂f(θ) = 0.0
  return (f, ∂f, ∂∂f)
end

struct ConstantCvLaw <: ThermalLaw
  θr::Float64
  ConstantCvLaw(θr) = new(θr)
  ConstantCvLaw(; θr) = new(θr)
end

function (law::ConstantCvLaw)()
  θr = law.θr
  f(θ) = (θ-θr) -θ*log(θ/θr)
  ∂f(θ) = -log(θ/θr)
  ∂∂f(θ) = -1/θ
  return (f, ∂f, ∂∂f)
end

struct EntropicElasticityLaw <: ThermalLaw
  θr::Float64
  γ::Float64
  EntropicElasticityLaw(; θr, γ) = new(θr, γ)
end

function (law::EntropicElasticityLaw)()
  (; θr, γ) = law
  f(θ) = (θ/θr)^(γ+1)
  ∂f(θ) = (γ+1) * θ^γ / θr^(γ+1)
  ∂∂f(θ) = γ*(γ+1) * θ^(γ-1) / θr^(γ+1)
  return (f, ∂f, ∂∂f)
end

struct NonlinearMeltingLaw <: ThermalLaw
  θr::Float64
  θM::Float64
  γ::Float64
  NonlinearMeltingLaw(; θr, θM, γ) = new(θr, θM, γ)
end

function (law::NonlinearMeltingLaw)()
  (; θr, θM, γ) = law
  f(θ) = (1 - (θ/θM)^(γ+1)) / (1 - (θr/θM)^(γ+1))
  ∂f(θ) = -(γ+1)*θ^γ/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  ∂∂f(θ) = -γ*(γ+1)*θ^(γ-1)/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  return (f, ∂f, ∂∂f)
end

struct NonlinearSofteningLaw <: ThermalLaw
  θr::Float64
  θT::Float64
  γ::Float64
  δ::Float64
  NonlinearSofteningLaw(; θr, θT, γ, δ=0) = new(θr, θT, γ, δ)
end

function (law::NonlinearSofteningLaw)()
  (; θr, θT, γ, δ) = law
  u(θ) = exp(-(θ/θT)^(γ+1))
  C = (1-δ) * u(θr) + δ
  f(θ) = ((1-δ) * u(θ) + δ) / C
  ∂f(θ) = -(1-δ)/C * (γ+1)/θT * (θ/θT)^γ * u(θ)
  ∂∂f(θ) = (1-δ)/C * (γ+1)/θ^2 * (θ/θT)^(γ+1) * ((γ+1)*(θ/θT)^(γ+1)-γ) * u(θ)
  return (f, ∂f, ∂∂f)
end

struct PolynomialLaw <: ThermalLaw
  θr::Float64
  a::Float64
  b::Float64
  c::Float64
  PolynomialLaw(; θr, a, b, c) = new(θr, a, b, c)
end

function (law::PolynomialLaw)()
  (; θr, a, b, c) = law
  f(θ)   = a*((θ-θr)/θr)^3  + b*((θ-θr)/θr)^2 + c*(θ-θr)/θr + 1
  ∂f(θ)  = 3a*(θ-θr)^2/θr^3 + 2b*(θ-θr)/θr^2 + c/θr
  ∂∂f(θ) = 6a*(θ-θr)/θr^3 + 2b/θr^2
  return (f, ∂f, ∂∂f)
end
