
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

struct EntropicElasticityLaw <: ThermalLaw
  θr::Float64
  γ::Float64
  EntropicElasticityLaw(; θr, γ) = new(θr, γ)
end

function derivatives(law::EntropicElasticityLaw)
  @unpack θr, γ = law
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

function derivatives(law::NonlinearMeltingLaw)
  @unpack θr, θM, γ = law
  f(θ) = (1 - (θ/θM)^(γ+1)) / (1 - (θr/θM)^(γ+1))
  ∂f(θ) = -(γ+1)*θ^γ/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  ∂∂f(θ) = -γ*(γ+1)*θ^(γ-1)/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  return (f, ∂f, ∂∂f)
end

struct NonlinearSofteningLaw <: ThermalLaw
  θr::Float64
  θt::Float64
  γ::Float64
  δ::Float64
  NonlinearSofteningLaw(; θr, θt, γ, δ=0) = new(θr, θt, γ, δ)
end

function derivatives(law::NonlinearSofteningLaw)
  @unpack θr, θt, γ, δ = law
  u(θ) = exp(-(θ/θt)^(γ+1))
  C = (1-δ) * u(θr) + δ
  f(θ) = ((1-δ) * u(θ) + δ) / C
  ∂f(θ) = -(1-δ)/C * (γ+1)/θt * (θ/θt)^γ * u(θ)
  ∂∂f(θ) = (1-δ)/C * (γ+1)/θ^2 * (θ/θt)^(γ+1) * ((γ+1)*(θ/θt)^(γ+1)-γ) * u(θ)
  return (f, ∂f, ∂∂f)
end

@deprecate EntropicMeltingLaw NonlinearMeltingLaw
@deprecate SofteningLaw NonlinearSofteningLaw

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

struct ThermoMech_Bonet{M<:Mechano, LE<:ThermalLaw, LV<:ThermalLaw} <: ThermoMechano{ThermalVolumetric,M}
  thermo::ThermalVolumetric
  mechano::M
  lawel::LE
  lawvis::LV

  function ThermoMech_Bonet(thermo::ThermalVolumetric, mechano::M, law::L) where {M<:Elasto, L<:ThermalLaw}
    new{M,L,L}(thermo,mechano,law,law)
  end
  
  function ThermoMech_Bonet(thermo::ThermalVolumetric, mechano::M, lawel::LE, lawvis::LV) where {M<:ViscoElastic, LE<:ThermalLaw, LV<:ThermalLaw}
    new{M,LE,LV}(thermo,mechano,lawel,lawvis)
  end
end

function (obj::ThermoMech_Bonet{<:Elasto})()
  Ψv, ∂Ψv∂F, ∂Ψv∂θ, ∂∂Ψv∂FF, ∂∂Ψv∂θθ, ∂∂Ψv∂Fθ = obj.thermo()
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  f, ∂f, ∂∂f = derivatives(obj.lawel)
  Ψ(F, θ)       =  Ψv(F,θ)      + f(θ)*Ψm(F)
  ∂Ψ∂F(F, θ)    =  ∂Ψv∂F(F,θ)   + f(θ)*∂Ψm∂F(F)
  ∂Ψ∂θ(F, θ)    =  ∂Ψv∂θ(F,θ)   + ∂f(θ)*Ψm(F)
  ∂∂Ψ∂FF(F, θ)  =  ∂∂Ψv∂FF(F,θ) + f(θ)*∂∂Ψm∂FF(F)
  ∂∂Ψ∂θθ(F, θ)  =  ∂∂Ψv∂θθ(F,θ) + ∂∂f(θ)*Ψm(F)
  ∂∂Ψ∂Fθ(F, θ)  =  ∂∂Ψv∂Fθ(F,θ) + ∂f(θ)*∂Ψm∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:ViscoElastic})()
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo()
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches()
  fe, ∂fe, ∂∂fe = derivatives(obj.lawel)
  fv, ∂fv, ∂∂fv = derivatives(obj.lawvis)
  Ψ(F, θ, X...)       =  Ψt(F, θ)      + fe(θ)*Ψe(F)      + fv(θ)*Ψv(F, X...)
  ∂Ψ∂F(F, θ, X...)    =  ∂Ψt∂F(F, θ)   + fe(θ)*∂Ψe∂F(F)   + fv(θ)*∂Ψv∂F(F, X...)
  ∂Ψ∂θ(F, θ, X...)    =  ∂Ψt∂θ(F, θ)   + ∂fe(θ)*Ψe(F)     + ∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂FF(F, θ, X...)  =  ∂∂Ψt∂FF(F, θ) + fe(θ)*∂∂Ψe∂FF(F) + fv(θ)*∂∂Ψv∂FF(F, X...)
  ∂∂Ψ∂θθ(F, θ, X...)  =  ∂∂Ψt∂θθ(F, θ) + ∂∂fe(θ)*Ψe(F)    + ∂∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂Fθ(F, θ, X...)  =  ∂∂Ψt∂Fθ(F, θ) + ∂fe(θ)*∂Ψe∂F(F)  + ∂fv(θ)*∂Ψv∂F(F, X...)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function Dissipation(obj::ThermoMech_Bonet)
  fv, ∂fv, _ = derivatives(obj.lawvis)
  Dvis = Dissipation(obj.mechano)
  D(F, θ, X...) = fv(θ) * Dvis(F, X...)
  ∂D∂θ(F, θ, X...) = ∂fv(θ) * Dvis(F, X...)
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
 