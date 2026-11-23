
# ===================
# Common functions
# ===================

function Gridap.CellData.CellState(obj::ThermoMechano, args...; kwargs...)
  CellState(obj.mechano, args...; kwargs...)
end

function initialize_state(obj::TM) where {TM<:ThermoMechano}
  initialize_state(obj.mechano)
end

function Gridap.CellData.update_state!(obj::TM, state, F, θ, args...; kwargs...) where {TM<:ThermoMechano}
  update_state!(obj.mechano, state, F, args...; kwargs...)
end

function update_time_step!(obj::TM, Δt::Float64) where {TM<:ThermoMechano}
  update_time_step!(obj.thermo,  Δt)
  update_time_step!(obj.mechano, Δt)
end


# ====================
# Multi-physics models
# ====================

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
  (; Cv, θr, α, κ) = obj.thermo
  cv0 = Cv  # FIXME
  U, ∂U∂F, ∂∂U∂FF = obj.mechano()
  κr = tangent(obj.mechano)
  f, df, ddf = obj.law()
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


struct ThermalDeviatoric{M<:Mechano} <: ThermoMechano{Nothing,M}
  mechano::M
  law::ThermalLaw

  function ThermalDeviatoric(mechano::M, law::ThermalLaw) where {M<:Mechano}
    new{M}(mechano, law)
  end
end

# No thermal component: the time step goes to the mechanical model only.
update_time_step!(obj::ThermalDeviatoric, Δt::Float64) = update_time_step!(obj.mechano, Δt)

function (obj::ThermalDeviatoric{<:IsoElastic})()
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  f, df, ddf = obj.law()
  Ψ(F,θ)      = Ψm(F) * f(θ)
  ∂Ψ∂F(F,θ)   = ∂Ψm∂F(F) * f(θ)
  ∂∂Ψ∂FF(F,θ) = ∂∂Ψm∂FF(F) * f(θ)
  ∂Ψ∂θ(F,θ)   = Ψm(F) * df(θ)
  ∂∂Ψ∂θθ(F,θ) = Ψm(F) * ddf(θ)
  ∂∂Ψ∂Fθ(F,θ) = ∂Ψm∂F(F) * df(θ)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end


"""
    ThermoMechModel(thermo, mechano, law)
    ThermoMechModel(; thermo, mechano, law)

Thermo-mechanical model in the temperature increment `δθ = θ - θr`:

```math
\\Psi(F, \\delta\\theta) = f(\\theta)\\,\\Psi_m(F) + \\Psi_t(\\delta\\theta) + \\Psi_{tm}(F, \\delta\\theta),
\\qquad \\theta = \\theta_r + \\delta\\theta
```

with `Ψm` the mechanical model, `Ψt` the thermal model (`ThermalModel`, which gives `θr`), `Ψtm` the
thermo-mechanical coupling `Ψtm(F, δθ) = -3α (J² - 1) δθ` (`α` of the thermal model, a stress per unit
temperature: the thermal expansion coefficient is `2α/K`, see [`ThermalModel`](@ref)) and `f` the thermal law
(`ThermalLaw`, which gives `f`, `f'` and `f''`). The law must
have the same reference temperature as the thermal model. The functor `model()` returns
`(Ψ, ∂Ψ/∂F, ∂Ψ/∂θ, ∂²Ψ/∂F², ∂²Ψ/∂θ², ∂²Ψ/∂F∂θ, η)` as functions of `(F, δθ)`, with `η = -∂Ψ/∂θ`.
"""
struct ThermoMechModel{T<:Thermo,M<:Mechano,L<:ThermalLaw} <: ThermoMechano{T,M}
  thermo::T
  mechano::M
  law::L

  function ThermoMechModel(thermo::T, mechano::M, law::L) where {T<:Thermo,M<:Mechano,L<:ThermalLaw}
    _check_reference_temperature(thermo, law)
    new{T,M,L}(thermo, mechano, law)
  end
end

function ThermoMechModel(; thermo::Thermo, mechano::Mechano, law::Union{ThermalLaw,Nothing}=nothing, kwargs...)
  law === nothing && _missing_thermal_law("ThermoMechModel(thermo, mechano, law)")
  ThermoMechModel(thermo, mechano, law)
end

ThermoMechModel(thermo::Thermo, mechano::Mechano; kwargs...) = _missing_thermal_law("ThermoMechModel(thermo, mechano, law)")

"Error of the constructors without a thermal law (the former keywords `fθ` and `dfdθ` are no longer accepted)."
function _missing_thermal_law(signature)
  throw(ArgumentError("a ThermalLaw is required: $signature, with `law` e.g. EntropicElasticityLaw(θr=θr, γ=0.0) " *
                      "for f(θ) = θ/θr (the keywords fθ and dfdθ are no longer accepted)"))
end

"The thermal law and the thermal model must share the reference temperature."
function _check_reference_temperature(thermo, law)
  if hasproperty(law, :θr) && hasproperty(thermo, :θr) && law.θr != thermo.θr
    throw(ArgumentError("the reference temperature of the thermal law (θr = $(law.θr)) differs from that " *
                        "of the thermal model (θr = $(thermo.θr))"))
  end
  nothing
end

function (obj::ThermoMechModel)()
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.thermo()
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano()
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.thermo, obj.mechano)
  fθ, dfθ, ddfθ = obj.law()
  θr = obj.thermo.θr
  f(δθ) = fθ(δθ + θr)
  df(δθ) = dfθ(δθ + θr)
  ddf(δθ) = ddfθ(δθ + θr)
  Ψ(F, δθ) = f(δθ) * (Ψm(F)) + (Ψt(δθ) + Ψtm(F, δθ))
  ∂Ψu(F, δθ) = f(δθ) * (∂Ψm_u(F)) + ∂Ψtm_u(F, δθ)
  ∂Ψθ(F, δθ) = df(δθ) * (Ψm(F)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)
  ∂Ψuu(F, δθ) = f(δθ) * (∂Ψm_uu(F)) + ∂Ψtm_uu(F, δθ)
  ∂Ψθθ(F, δθ) = ddf(δθ) * (Ψm(F)) + ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
  ∂Ψuθ(F, δθ) = df(δθ) * (∂Ψm_u(F)) + ∂Ψtm_uθ(F, δθ)
  η(F, δθ) = -∂Ψθ(F, δθ)
  return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ, η)
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

function (obj::ThermoMech_Bonet{<:IsoElastic})()
  Ψv, ∂Ψv∂F, ∂Ψv∂θ, ∂∂Ψv∂FF, ∂∂Ψv∂θθ, ∂∂Ψv∂Fθ = obj.thermo()
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  f, ∂f, ∂∂f = obj.lawel()
  Ψ(F, θ)       =  Ψv(F,θ)      + f(θ)*Ψm(F)
  ∂Ψ∂F(F, θ)    =  ∂Ψv∂F(F,θ)   + f(θ)*∂Ψm∂F(F)
  ∂Ψ∂θ(F, θ)    =  ∂Ψv∂θ(F,θ)   + ∂f(θ)*Ψm(F)
  ∂∂Ψ∂FF(F, θ)  =  ∂∂Ψv∂FF(F,θ) + f(θ)*∂∂Ψm∂FF(F)
  ∂∂Ψ∂θθ(F, θ)  =  ∂∂Ψv∂θθ(F,θ) + ∂∂f(θ)*Ψm(F)
  ∂∂Ψ∂Fθ(F, θ)  =  ∂∂Ψv∂Fθ(F,θ) + ∂f(θ)*∂Ψm∂F(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:AnisoElastic})()
  Ψv, ∂Ψv∂F, ∂Ψv∂θ, ∂∂Ψv∂FF, ∂∂Ψv∂θθ, ∂∂Ψv∂Fθ = obj.thermo()
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  f, ∂f, ∂∂f = obj.lawel()
  Ψ(F, θ, N)       =  Ψv(F,θ)      + f(θ)*Ψm(F, N)
  ∂Ψ∂F(F, θ, N)    =  ∂Ψv∂F(F,θ)   + f(θ)*∂Ψm∂F(F, N)
  ∂Ψ∂θ(F, θ, N)    =  ∂Ψv∂θ(F,θ)   + ∂f(θ)*Ψm(F, N)
  ∂∂Ψ∂FF(F, θ, N)  =  ∂∂Ψv∂FF(F,θ) + f(θ)*∂∂Ψm∂FF(F, N)
  ∂∂Ψ∂θθ(F, θ, N)  =  ∂∂Ψv∂θθ(F,θ) + ∂∂f(θ)*Ψm(F, N)
  ∂∂Ψ∂Fθ(F, θ, N)  =  ∂∂Ψv∂Fθ(F,θ) + ∂f(θ)*∂Ψm∂F(F, N)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:ViscoElastic{<:IsoElastic}})()
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo()
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches()
  fe, ∂fe, ∂∂fe = obj.lawel()
  fv, ∂fv, ∂∂fv = obj.lawvis()
  Ψ(F, θ, X::Vararg{Any,K})       where {K} =  Ψt(F, θ)      + fe(θ)*Ψe(F)      + fv(θ)*Ψv(F, X...)
  ∂Ψ∂F(F, θ, X::Vararg{Any,K})    where {K} =  ∂Ψt∂F(F, θ)   + fe(θ)*∂Ψe∂F(F)   + fv(θ)*∂Ψv∂F(F, X...)
  ∂Ψ∂θ(F, θ, X::Vararg{Any,K})    where {K} =  ∂Ψt∂θ(F, θ)   + ∂fe(θ)*Ψe(F)     + ∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂FF(F, θ, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂FF(F, θ) + fe(θ)*∂∂Ψe∂FF(F) + fv(θ)*∂∂Ψv∂FF(F, X...)
  ∂∂Ψ∂θθ(F, θ, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂θθ(F, θ) + ∂∂fe(θ)*Ψe(F)    + ∂∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂Fθ(F, θ, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂Fθ(F, θ) + ∂fe(θ)*∂Ψe∂F(F)  + ∂fv(θ)*∂Ψv∂F(F, X...)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function (obj::ThermoMech_Bonet{<:ViscoElastic{<:AnisoElastic}})()
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo()
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches()
  fe, ∂fe, ∂∂fe = obj.lawel()
  fv, ∂fv, ∂∂fv = obj.lawvis()
  Ψ(F, θ, N, X::Vararg{Any,K})       where {K} =  Ψt(F, θ)      + fe(θ)*Ψe(F, N)      + fv(θ)*Ψv(F, X...)
  ∂Ψ∂F(F, θ, N, X::Vararg{Any,K})    where {K} =  ∂Ψt∂F(F, θ)   + fe(θ)*∂Ψe∂F(F, N)   + fv(θ)*∂Ψv∂F(F, X...)
  ∂Ψ∂θ(F, θ, N, X::Vararg{Any,K})    where {K} =  ∂Ψt∂θ(F, θ)   + ∂fe(θ)*Ψe(F, N)     + ∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂FF(F, θ, N, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂FF(F, θ) + fe(θ)*∂∂Ψe∂FF(F, N) + fv(θ)*∂∂Ψv∂FF(F, X...)
  ∂∂Ψ∂θθ(F, θ, N, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂θθ(F, θ) + ∂∂fe(θ)*Ψe(F, N)    + ∂∂fv(θ)*Ψv(F, X...)
  ∂∂Ψ∂Fθ(F, θ, N, X::Vararg{Any,K})  where {K} =  ∂∂Ψt∂Fθ(F, θ) + ∂fe(θ)*∂Ψe∂F(F, N)  + ∂fv(θ)*∂Ψv∂F(F, X...)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂θθ, ∂∂Ψ∂Fθ)
end

function Dissipation(obj::ThermoMech_Bonet)
  fv, ∂fv, _ = obj.lawvis()
  Dvis = Dissipation(obj.mechano)
  D(F, θ, X::Vararg{Any,K}) where {K} = fv(θ) * Dvis(F, X...)
  ∂D∂θ(F, θ, X::Vararg{Any,K}) where {K} = ∂fv(θ) * Dvis(F, X...)
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

  function (obj::ThermoMech_EntropicPolyconvex)()
    Ψt, _, _ = obj.thermo()
    Ψm, _, _ = obj.mechano()
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
    ∂Ψuu(F, δθ) = FourthOrderTensorValue(Tuple(∂2Ψ_∂2∇u(F, δθ)))
    ∂Ψθθ(F, δθ) = ∂2Ψ_∂2θθ(F, δθ)
    ∂Ψuθ(F, δθ) = TensorValue(∂2Ψ_∂2∇uθ(F, δθ))

    return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ)
  end
end


function _getCoupling(term::Thermo, mec::Mechano)
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
  ∂Ψtm_∂J(F, δθ) = -6.0 * term.α * J(F) * δθ
  ∂Ψtm_u(F, δθ) = ∂Ψtm_∂J(F, δθ) * H(F)
  ∂Ψtm_θ(F, δθ) = -3.0 * term.α * (J(F)^2.0 - 1.0)
  ∂Ψtm_uu(F, δθ) = (-6.0 * term.α * δθ) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψtm_∂J(F, δθ) * F)
  ∂Ψtm_uθ(F, δθ) = -6.0 * term.α * J(F) * H(F)
  ∂Ψtm_θθ(F, δθ) = 0.0

  Ψtm(F, δθ) = ∂Ψtm_θ(F, δθ) * δθ

  return (Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ)
end
 