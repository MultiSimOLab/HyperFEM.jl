
# ===================
# Common functions
# ===================

function initialize_state(obj::TM, points::Measure) where {TM<:ThermoMechano}
  initialize_state(obj.mechano, points)
end

function update_state!(obj::TM, state, F, θ, args...) where {TM<:ThermoMechano}
  update_state!(obj.mechano, state, F, args...)
end

function update_time_step!(obj::TM, Δt::Float64) where {TM<:ThermoMechano}
  update_time_step!(obj.thermo,  Δt)
  update_time_step!(obj.mechano, Δt)
end

# ===================
# MultiPhysicalModel models
# ===================

struct ThermoMechModel{T<:Thermo,M<:Mechano} <: ThermoMechano
  thermo::T
  mechano::M
  fθ::Function
  dfdθ::Function

  function ThermoMechModel(thermo::T, mechano::M; fθ::Function, dfdθ::Function) where {T <: Thermo, M <: Mechano}
    new{T,M}(thermo, mechano, fθ, dfdθ)
  end

  function ThermoMechModel(; thermo::T, mechano::M, fθ::Function, dfdθ::Function) where {T <: Thermo, M <: Mechano}
    new{T,M}(thermo, mechano, fθ, dfdθ)
  end

  # function ThermoMechModel(thermo::ThermalModel3rdLaw, mechano::M) where {M<:Mechano}
  #   new{ThermalModel3rdLaw,M}(thermo,mechano)
  # end

  function (obj::ThermoMechModel)(Λ::Float64=1.0)
    Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.thermo(Λ)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.thermo, obj.mechano, Λ)
    f(δθ) = (obj.fθ(δθ)::Float64)
    df(δθ) = (obj.dfdθ(δθ)::Float64)
    Ψ(F, δθ) = f(δθ) * (Ψm(F)) + (Ψt(δθ) + Ψtm(F, δθ))
    ∂Ψu(F, δθ) = f(δθ) * (∂Ψm_u(F)) + ∂Ψtm_u(F, δθ)
    ∂Ψθ(F, δθ) = df(δθ) * (Ψm(F)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)
    ∂Ψuu(F, δθ) = f(δθ) * (∂Ψm_uu(F)) + ∂Ψtm_uu(F, δθ)
    ∂Ψθθ(F, δθ) = ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
    ∂Ψuθ(F, δθ) = df(δθ) * (∂Ψm_u(F)) + ∂Ψtm_uθ(F, δθ)
    η(F, δθ) = -∂Ψθ(F, δθ)
    return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ, η)
  end
end


# function (obj::ThermoMechModel{ThermalModel3rdLaw,<:Mechano})(Λ::Float64=0.0)
#   @unpack cv0, θr, α, κ, γv, γd = obj.thermo
#   gv, ∂gv, ∂∂gv = volumetric_law(obj.thermo)
#   gd, ∂gd, ∂∂gd = isochoric_law(obj.thermo)
#   Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
#   ηR, ∂ηR∂F, ∂∂ηR∂FF = _getCoupling(obj.thermo, obj.mechano)
#   Ψ(F, θ, X...)       =  gd(θ)*Ψm(F, X...) - θr*gv(θ)*ηR(F)
#   ∂Ψ∂F(F, θ, X...)    =  gd(θ)*∂Ψm∂F(F, X...) - θr*gv(θ)*∂ηR∂F(F)
#   ∂Ψ∂θ(F, θ, X...)    =  ∂gd(θ)*Ψm(F, X...) - θr*∂gv(θ)*ηR(F)
#   ∂∂Ψ∂FF(F, θ, X...)  =  gd(θ)*∂∂Ψm∂FF(F, E, X...) - θr*gv(θ)*∂∂ηR∂FF(F)
#   ∂∂Ψ∂θθ(F, θ, X...)  =  ∂∂gd(θ)*Ψm(F, X...) - θr*∂∂gv(θ)*ηR(F)
#   ∂∂Ψ∂Fθ(F, θ, X...)  =  ∂gd(θ)*∂Ψm∂F(F, X...) - θr*∂gv(θ)*∂ηR∂F(F)
#   return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
# end

# function _getCoupling(thermo::ThermalModel3rdLaw, ::Mechano)
#   @unpack cv0, θr, α, κ, γv, γd = thermo
#   J(F) = det(F)
#   H(F) = cof(F)
#   ηR(F) = α*(J(F) - 1.0) + cv0/γv
#   ∂ηR∂J(F) = α
#   ∂ηR∂F(F) = ∂ηR∂J(F)*H(F)
#   ∂∂ηR∂FF(F) = ×ᵢ⁴(∂ηR∂J(F) * F)
#   return (ηR, ∂ηR∂F, ∂∂ηR∂FF)
# end

# function Dissipation(obj::ThermoMechModel{ThermalModel3rdLaw,<:Mechano})
#   @unpack cv0, θr, α, κ, γv, γd = obj.thermo
#   gd, ∂gd, ∂∂gd = isochoric_law(obj.thermo)
#   Dvis = Dissipation(obj.mechano)
#   D(F, θ, X...) = gd(θ) * Dvis(F, X...)
#   ∂D∂θ(F, θ, X...) = ∂gd(θ) * Dvis(F, X...)
#   return(D, ∂D∂θ)
# end


abstract type ThermalLaw end

struct VolumetricLaw <: ThermalLaw
  θr::Float64
  γ::Float64
end

function derivatives(law::VolumetricLaw)
  @unpack θr, γ = law
  g(θ) = 1/(γ+1) * ((θ/θr)^(γ+1) -1)
  ∂g(θ) = θ^γ / θr^(γ+1)
  ∂∂g(θ) = γ*θ^(γ-1) / θr^(γ+1)
  return (g, ∂g, ∂∂g)
end

struct DeviatoricLaw <: ThermalLaw
  θr::Float64
  γ::Float64
end

function derivatives(law::DeviatoricLaw)
  @unpack θr, γ = law
  g(θ) = (θ/θr)^(-γ)
  ∂g(θ) = -γ*θ^(-γ-1) * θr^γ
  ∂∂g(θ) = γ*(γ+1)*θ^(-γ-2) * θr^γ
  return (g, ∂g, ∂∂g)
end

struct InterceptLaw <: ThermalLaw
  θr::Float64
  γ::Float64
  δ::Float64
end

function derivatives(law::InterceptLaw)
  @unpack θr, γ, δ = law
  g(θ) = (θ/θr)^(-γ) * (1-δ) + δ
  ∂g(θ) = -γ*θ^(-γ-1) * θr^γ * (1-δ)
  ∂∂g(θ) = γ*(γ+1)*θ^(-γ-2) * θr^γ * (1-δ)
  return (g, ∂g, ∂∂g)
end

struct TrigonometricLaw <: ThermalLaw
  θr::Float64
  θM::Float64
end

function derivatives(law::TrigonometricLaw)
  @unpack θr, θM = law  
  g(θ) = θ/θr * sin(2π*θ/θM)
  G(θ) = 1/2/π * θM/θr * (1 - cos(2π*θ/θM))
  H(θ) = 1/2/π * θM/θr * (θ - θM/2/π * sin(2π*θ/θM))
  f(θ) = (H(θr) - H(θ)) / (H(θM) - H(θr)) + 1.0
  ∂f(θ) = -G(θ) / (H(θM) - H(θr))
  ∂∂f(θ) = g(θ) / θ / (H(θM) - H(θr))
  return (f, ∂f, ∂∂f)
end

struct PolynomialLaw{N} <: ThermalLaw
  a0::Float64
  ai::NTuple{N, Float64}
end

PolynomialLaw(a0::Real, ai::Real...) = PolynomialLaw(Float64(a0), Float64.(ai))

function derivatives(law::PolynomialLaw)
  c0 = (law.a0, law.ai...)
  c1 = ntuple(i -> i * c0[i+1], length(c0) - 1)
  c2 = length(c1) < 1 ? (0.0) : ntuple(i -> i * c1[i+1], length(c1) - 1)
  f(θ)   = evalpoly(θ, c0)
  ∂f(θ)  = evalpoly(θ, c1)
  ∂∂f(θ) = evalpoly(θ, c2)
  return (f, ∂f, ∂∂f)
end

struct ThermoMech_Bonet{T<:Thermo,M<:Mechano} <: ThermoMechano
  thermo::T
  mechano::M
  gv::VolumetricLaw
  gd::ThermalLaw
  gvis::ThermalLaw
end

function ThermoMech_Bonet(thermo::T, mechano::M; γv::Float64, γd::Float64, γvis::Float64=γd) where {T<:Thermo,M<:Mechano}
  gv   = VolumetricLaw(thermo.θr, γv)
  gd   = DeviatoricLaw(thermo.θr, γd)
  gvis = DeviatoricLaw(thermo.θr, γvis)
  ThermoMech_Bonet{T,M}(thermo,mechano,gv,gd,gvis)
end

function entropy(obj::ThermoMech_Bonet)
  cv0, α, γv = obj.thermo.Cv, obj.thermo.α, obj.gv.γ
  J(F) = det(F)
  H(F) = cof(F)
  ηR(F) = α*(J(F) - 1.0) + cv0/γv
  ∂ηR∂J(F) = α
  ∂ηR∂F(F) = ∂ηR∂J(F)*H(F)
  ∂∂ηR∂FF(F) = ×ᵢ⁴(∂ηR∂J(F) * F)
  return (ηR, ∂ηR∂F, ∂∂ηR∂FF)
end

function (obj::ThermoMech_Bonet{<:Thermo,<:Elasto})(Λ::Float64=0.0)
  θr = obj.thermo.θr
  gv, ∂gv, ∂∂gv = derivatives(obj.gv)
  gd, ∂gd, ∂∂gd = derivatives(obj.gd)
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  ηR, ∂ηR∂F, ∂∂ηR∂FF = entropy(obj)
  Ψ(F, θ)       =  gd(θ)*Ψm(F)      - θr*gv(θ)*ηR(F)
  ∂Ψ∂F(F, θ)    =  gd(θ)*∂Ψm∂F(F)   - θr*gv(θ)*∂ηR∂F(F)
  ∂Ψ∂θ(F, θ)    =  ∂gd(θ)*Ψm(F)     - θr*∂gv(θ)*ηR(F)
  ∂∂Ψ∂FF(F, θ)  =  gd(θ)*∂∂Ψm∂FF(F) - θr*gv(θ)*∂∂ηR∂FF(F)
  ∂∂Ψ∂θθ(F, θ)  =  ∂∂gd(θ)*Ψm(F)    - θr*∂∂gv(θ)*ηR(F)
  ∂∂Ψ∂Fθ(F, θ)  =  ∂gd(θ)*∂Ψm∂F(F)  - θr*∂gv(θ)*∂ηR∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:Thermo,<:ViscoElastic})(Λ::Float64=0.0)
  θr = obj.thermo.θr
  gv, ∂gv, ∂∂gv       = derivatives(obj.gv)
  gd, ∂gd, ∂∂gd       = derivatives(obj.gd)
  gvis, ∂gvis, ∂∂gvis = derivatives(obj.gvis)
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF  = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF  = obj.mechano.branches()
  ηR, ∂ηR∂F, ∂∂ηR∂FF  = entropy(obj)
  Ψ(F, θ, X...)       =  gd(θ)*Ψe(F)      + gvis(θ)*Ψv(F, X...)      - θr*gv(θ)*ηR(F)
  ∂Ψ∂F(F, θ, X...)    =  gd(θ)*∂Ψe∂F(F)   + gvis(θ)*∂Ψv∂F(F, X...)   - θr*gv(θ)*∂ηR∂F(F)
  ∂Ψ∂θ(F, θ, X...)    =  ∂gd(θ)*Ψe(F)     + ∂gvis(θ)*Ψv(F, X...)     - θr*∂gv(θ)*ηR(F)
  ∂∂Ψ∂FF(F, θ, X...)  =  gd(θ)*∂∂Ψe∂FF(F) + gvis(θ)*∂∂Ψv∂FF(F, X...) - θr*gv(θ)*∂∂ηR∂FF(F)
  ∂∂Ψ∂θθ(F, θ, X...)  =  ∂∂gd(θ)*Ψe(F)    + ∂∂gvis(θ)*Ψv(F, X...)    - θr*∂∂gv(θ)*ηR(F)
  ∂∂Ψ∂Fθ(F, θ, X...)  =  ∂gd(θ)*∂Ψe∂F(F)  + ∂gvis(θ)*∂Ψv∂F(F, X...)  - θr*∂gv(θ)*∂ηR∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function Dissipation(obj::ThermoMech_Bonet)
  gd, ∂gd, ∂∂gd = derivatives(obj.gvis)
  Dvis = Dissipation(obj.mechano)
  D(F, θ, X...) = gd(θ) * Dvis(F, X...)
  ∂D∂θ(F, θ, X...) = ∂gd(θ) * Dvis(F, X...)
  return(D, ∂D∂θ)
end


struct ThermoMech_EntropicPolyconvex{T<:Thermo,M<:Mechano} <: ThermoMechano
  thermo::T
  mechano::M
  β::Float64
  G::Function
  ϕ::Function
  s::Function

  function ThermoMech_EntropicPolyconvex(thermo::T, mechano::M; β::Float64, G::Function, ϕ::Function, s::Function) where {T <: Thermo, M <: Mechano}
    new{T,M}(thermo, mechano, β, G, ϕ, s)
  end

  function ThermoMech_EntropicPolyconvex(; thermo::T, mechano::M, β::Float64, G::Function, ϕ::Function, s::Function) where {T <: Thermo, M <: Mechano}
    new{T,M}(thermo, mechano, β, G, ϕ, s)
  end

  function (obj::ThermoMech_EntropicPolyconvex)(Λ::Float64=1.0)
    Ψt, _, _ = obj.thermo(Λ)
    Ψm, _, _ = obj.mechano(Λ)
    θr = obj.thermo.θr
    Cv = obj.thermo.Cv
    α = obj.thermo.α
    β = obj.β
    G = obj.G
    ϕ = obj.ϕ
    s = obj.s

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)

    f(δθ) = (δθ + θr) / θr
    eᵣ(F) = α * (J(F) - 1.0)
    L1(δθ) = (1 - β) * Ψt(δθ)
    L2(δθ) = Cv * θr * (1 - β) * G(f(δθ))
    L3(F, δθ) = -Cv * θr * β * s(I1(F), I2(F), I3(F)) * ϕ(f(δθ))

    Ψ(F, δθ) = f(δθ) * Ψm(F) + (1 - f(δθ)) * eᵣ(F) + L1(δθ) + L2(δθ) + L3(F, δθ)

    ∂Ψ_∂∇u(F, δθ) = ForwardDiff.gradient(F -> Ψ(F, δθ), get_array(F))
    ∂Ψ_∂θ(F, δθ) = ForwardDiff.derivative(δθ -> Ψ(get_array(F), δθ), δθ)
    ∂2Ψ_∂2∇u(F, δθ) = ForwardDiff.hessian(F -> Ψ(F, δθ), get_array(F))
    ∂2Ψ_∂2θθ(F, δθ) = ForwardDiff.derivative(δθ -> ∂Ψ_∂θ(get_array(F), δθ), δθ)
    ∂2Ψ_∂2∇uθ(F, δθ) = ForwardDiff.derivative(δθ -> ∂Ψ_∂∇u(get_array(F), δθ), δθ)

    ∂Ψu(F, δθ) = TensorValue(∂Ψ_∂∇u(F, δθ))
    ∂Ψθ(F, δθ) = ∂Ψ_∂θ(F, δθ)
    ∂Ψuu(F, δθ) = TensorValue(∂2Ψ_∂2∇u(F, δθ))
    ∂Ψθθ(F, δθ) = ∂2Ψ_∂2θθ(F, δθ)
    ∂Ψuθ(F, δθ) = TensorValue(∂2Ψ_∂2∇uθ(F, δθ))

    return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ)
  end
end


function _getCoupling(term::Thermo, mec::Mechano, Λ::Float64)
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
  ∂Ψtm_∂J(F, δθ) = -6.0 * term.α * J(F) * δθ
  ∂Ψtm_u(F, δθ) = ∂Ψtm_∂J(F, δθ) * H(F)
  ∂Ψtm_θ(F, δθ) = -3.0 * term.α * (J(F)^2.0 - 1.0)
  ∂Ψtm_uu(F, δθ) = (-6.0 * term.α * δθ) * (H(F) ⊗₁₂³⁴ H(F)) + ×ᵢ⁴(∂Ψtm_∂J(F, δθ) * F)
  ∂Ψtm_uθ(F, δθ) = -6.0 * term.α * J(F) * H(F)
  ∂Ψtm_θθ(F, δθ) = 0.0

  Ψtm(F, δθ) = ∂Ψtm_θ(F, δθ) * δθ

  return (Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ)
end
 