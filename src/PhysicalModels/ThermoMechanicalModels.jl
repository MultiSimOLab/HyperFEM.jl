
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

struct ThermoMechModel{T<:Thermo,M<:Mechano} <: ThermoMechano{T,M}
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


struct VolumetricLaw <: ThermalLaw
  θr::Float64
  γ::Float64
end

function derivatives(law::VolumetricLaw)
  @unpack θr, γ = law
  f(θ) = 1/(γ+1) * ((θ/θr)^(γ+1) -1)
  ∂f(θ) = θ^γ / θr^(γ+1)
  ∂∂f(θ) = γ*θ^(γ-1) / θr^(γ+1)
  return (f, ∂f, ∂∂f)
end

struct EntropicMeltingLaw <: ThermalLaw
  θr::Float64
  θM::Float64
  γ::Float64
end

function derivatives(law::EntropicMeltingLaw)
  @unpack θr, θM, γ = law
  f(θ) = (1 - (θ/θM)^(γ+1)) / (1 - (θr/θM)^(γ+1))
  ∂f(θ) = -(γ+1)*θ^γ/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  ∂∂f(θ) = -γ*(γ+1)*θ^(γ-1)/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  return (f, ∂f, ∂∂f)
end

struct DeviatoricLaw <: ThermalLaw
  θr::Float64
  γ::Float64
end

function derivatives(law::DeviatoricLaw)
  @unpack θr, γ = law
  f(θ) = (θ/θr)^γ
  ∂f(θ) = γ*θ^(γ-1) / θr^γ
  ∂∂f(θ) = γ*(γ-1)*θ^(γ-2) / θr^γ
  return (f, ∂f, ∂∂f)
end

struct SofteningLaw <: ThermalLaw
  θr::Float64
  θt::Float64
  γ::Float64
  δ::Float64
end

function derivatives(law::SofteningLaw)
  @unpack θr, θt, γ, δ = law
  h(θ) = exp((θr/θt)^γ-(θ/θt)^γ) * δ
  f(θ) = h(θ) + 1 - δ
  ∂f(θ) = -γ/θt * (θ/θt)^(γ-1) * h(θ)
  ∂∂f(θ) = 1/θ * (γ -1 -γ*(θ/θt)^γ) * ∂f(θ)
  return (f, ∂f, ∂∂f)
end

struct InterceptLaw <: ThermalLaw
  θr::Float64
  γ::Float64
  δ::Float64
end

function derivatives(law::InterceptLaw)
  @unpack θr, γ, δ = law
  f(θ) = (θ/θr)^(-γ) * (1-δ) + δ
  ∂f(θ) = -γ*θ^(-γ-1) * θr^γ * (1-δ)
  ∂∂f(θ) = γ*(γ+1)*θ^(-γ-2) * θr^γ * (1-δ)
  return (f, ∂f, ∂∂f)
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
  ∂∂f(θ) = -g(θ) / θ / (H(θM) - H(θr))
  return (f, ∂f, ∂∂f)
end

struct PolynomialLaw <: ThermalLaw
  θr::Float64
  a::Float64
  b::Float64
  c::Float64
end

function derivatives(law::PolynomialLaw)
  @unpack θr, a, b, c = law
  f(θ)   = a*((θ-θr)/θr)^3  + b*((θ-θr)/θr)^2 + c*(θ-θr)/θr + 1
  ∂f(θ)  = 3a*(θ-θr)^2/θr^3 + 2b*(θ-θr)/θr^2 + c/θr
  ∂∂f(θ) = 6a*(θ-θr)/θr^3 + 2b/θr^2
  return (f, ∂f, ∂∂f)
end

struct LogisticLaw <: ThermalLaw
  θr::Float64
  μ::Float64
  σ::Float64
end

function derivatives(law::LogisticLaw)
  @unpack θr, μ, σ = law
  z(x) = (log(x) - μ) / σ
  std_pdf(x) = 1/(σ*sqrt(2 * π)) * exp(-z(x)^2 / 2)
  std_cdf(x) = 0.5 * (1 + erf(z(x) / sqrt(2)))
  ξR = 1 / (1-std_cdf(θr))
  f(θ) = ξR * (1-std_cdf(θ))
  ∂f(θ) = -ξR / θ * std_pdf(θ)
  ∂∂f(θ) = ξR / θ^2 * std_pdf(θ) * (1 + z(θ)/σ)
  return (f, ∂f, ∂∂f)
end

struct ThermoMech_Bonet{T<:Thermo,M<:Mechano} <: ThermoMechano{T,M}
  thermo::T
  mechano::M
  lawvol::VolumetricLaw
  lawdev::ThermalLaw
  lawvis::ThermalLaw
end

function ThermoMech_Bonet(thermo::T, mechano::M; γv::Float64, γd::Float64, γvis::Float64=γd) where {T<:Thermo,M<:Mechano}
  lawvol = VolumetricLaw(thermo.θr, γv)
  lawdev = DeviatoricLaw(thermo.θr, γd)
  lawvis = DeviatoricLaw(thermo.θr, γvis)
  ThermoMech_Bonet{T,M}(thermo,mechano,lawvol,lawdev,lawvis)
end

function entropy(obj::ThermoMech_Bonet)
  cv0, α, γv = obj.thermo.Cv, obj.thermo.α, obj.lawvol.γ
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
  fv, ∂fv, ∂∂fv = derivatives(obj.lawvol)
  fd, ∂fd, ∂∂fd = derivatives(obj.lawdev)
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  ηR, ∂ηR∂F, ∂∂ηR∂FF = entropy(obj)
  Ψ(F, θ)       =  fd(θ)*Ψm(F)      - θr*fv(θ)*ηR(F)
  ∂Ψ∂F(F, θ)    =  fd(θ)*∂Ψm∂F(F)   - θr*fv(θ)*∂ηR∂F(F)
  ∂Ψ∂θ(F, θ)    =  ∂fd(θ)*Ψm(F)     - θr*∂fv(θ)*ηR(F)
  ∂∂Ψ∂FF(F, θ)  =  fd(θ)*∂∂Ψm∂FF(F) - θr*fv(θ)*∂∂ηR∂FF(F)
  ∂∂Ψ∂θθ(F, θ)  =  ∂∂fd(θ)*Ψm(F)    - θr*∂∂fv(θ)*ηR(F)
  ∂∂Ψ∂Fθ(F, θ)  =  ∂fd(θ)*∂Ψm∂F(F)  - θr*∂fv(θ)*∂ηR∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:Thermo,<:ViscoElastic})(Λ::Float64=0.0)
  θr = obj.thermo.θr
  fv, ∂fv, ∂∂fv       = derivatives(obj.lawvol)
  fd, ∂fd, ∂∂fd       = derivatives(obj.lawdev)
  fvis, ∂fvis, ∂∂fvis = derivatives(obj.lawvis)
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF  = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF  = obj.mechano.branches()
  ηR, ∂ηR∂F, ∂∂ηR∂FF  = entropy(obj)
  Ψ(F, θ, X...)       =  fd(θ)*Ψe(F)      + fvis(θ)*Ψv(F, X...)      - θr*fv(θ)*ηR(F)
  ∂Ψ∂F(F, θ, X...)    =  fd(θ)*∂Ψe∂F(F)   + fvis(θ)*∂Ψv∂F(F, X...)   - θr*fv(θ)*∂ηR∂F(F)
  ∂Ψ∂θ(F, θ, X...)    =  ∂fd(θ)*Ψe(F)     + ∂fvis(θ)*Ψv(F, X...)     - θr*∂fv(θ)*ηR(F)
  ∂∂Ψ∂FF(F, θ, X...)  =  fd(θ)*∂∂Ψe∂FF(F) + fvis(θ)*∂∂Ψv∂FF(F, X...) - θr*fv(θ)*∂∂ηR∂FF(F)
  ∂∂Ψ∂θθ(F, θ, X...)  =  ∂∂fd(θ)*Ψe(F)    + ∂∂fvis(θ)*Ψv(F, X...)    - θr*∂∂fv(θ)*ηR(F)
  ∂∂Ψ∂Fθ(F, θ, X...)  =  ∂fd(θ)*∂Ψe∂F(F)  + ∂fvis(θ)*∂Ψv∂F(F, X...)  - θr*∂fv(θ)*∂ηR∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function Dissipation(obj::ThermoMech_Bonet)
  fvis, ∂fvis, _ = derivatives(obj.lawvis)
  Dvis = Dissipation(obj.mechano)
  D(F, θ, X...) = fvis(θ) * Dvis(F, X...)
  ∂D∂θ(F, θ, X...) = ∂fvis(θ) * Dvis(F, X...)
  return (D, ∂D∂θ)
end


struct ThermoMech_EntropicPolyconvex{T<:Thermo,M<:Mechano} <: ThermoMechano{T,M}
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
 