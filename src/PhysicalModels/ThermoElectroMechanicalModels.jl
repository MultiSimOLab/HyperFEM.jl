
function Gridap.CellData.CellState(obj::ThermoElectroMechano, args...; kwargs...)
  CellState(obj.mechano, args...; kwargs...)
end

function initialize_state(obj::ThermoElectroMechano)
  initialize_state(obj.mechano)
end

function Gridap.CellData.update_state!(obj::ThermoElectroMechano, state, F, E, θ, args...; kwargs...)
  update_state!(obj.mechano, state, F, args...; kwargs...)
end

function update_time_step!(obj::ThermoElectroMechano, Δt::Float64)
  update_time_step!(obj.thermo,  Δt)
  update_time_step!(obj.electro, Δt)
  update_time_step!(obj.mechano, Δt)
end

"""
    ThermoElectroMechModel(thermo, electro, mechano, law)
    ThermoElectroMechModel(; thermo, electro, mechano, law)

Thermo-electro-mechanical model in the temperature increment `δθ = θ - θr`:

```math
\\Psi(F, E, \\delta\\theta) = f(\\theta)\\,\\big(\\Psi_m(F) + \\Psi_{em}(F, E)\\big) + \\Psi_t(\\delta\\theta)
+ \\Psi_{tm}(F, \\delta\\theta), \\qquad \\theta = \\theta_r + \\delta\\theta
```

with the same ingredients as [`ThermoMechModel`](@ref) plus the electro-mechanical model `Ψem`. The functor
`model(Λ=1.0)` returns `(Ψ, ∂Ψ/∂F, ∂Ψ/∂E, ∂Ψ/∂θ, ∂²Ψ/∂F², ∂²Ψ/∂E², ∂²Ψ/∂θ², ∂²Ψ/∂E∂F, ∂²Ψ/∂F∂θ, ∂²Ψ/∂E∂θ, η)`
as functions of `(F, E, δθ)`.
"""
struct ThermoElectroMechModel{T<:Thermo,E<:Electro,M<:Mechano,L<:ThermalLaw} <: ThermoElectroMechano{T,E,M}
  thermo::T
  electro::E
  mechano::M
  law::L

  function ThermoElectroMechModel(thermo::T, electro::E, mechano::M, law::L) where {T<:Thermo,E<:Electro,M<:Mechano,L<:ThermalLaw}
    _check_reference_temperature(thermo, law)
    new{T,E,M,L}(thermo, electro, mechano, law)
  end
end

function ThermoElectroMechModel(; thermo::Thermo, electro::Electro, mechano::Mechano,
                                law::Union{ThermalLaw,Nothing}=nothing, kwargs...)
  law === nothing && _missing_thermal_law("ThermoElectroMechModel(thermo, electro, mechano, law)")
  ThermoElectroMechModel(thermo, electro, mechano, law)
end

ThermoElectroMechModel(thermo::Thermo, electro::Electro, mechano::Mechano; kwargs...) =
  _missing_thermal_law("ThermoElectroMechModel(thermo, electro, mechano, law)")

function (obj::ThermoElectroMechModel)(Λ::Float64=1.0)
  Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.thermo(Λ)
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = obj.electro(Λ)
  Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.thermo, obj.mechano, Λ)
  fθ, dfθ, ddfθ = obj.law()
  θr = obj.thermo.θr
  f(δθ) = fθ(δθ + θr)
  df(δθ) = dfθ(δθ + θr)
  ddf(δθ) = ddfθ(δθ + θr)

  Ψ(F, E, δθ) = f(δθ) * (Ψm(F) + Ψem(F, E)) + (Ψt(δθ) + Ψtm(F, δθ))
  ∂Ψu(F, E, δθ) = f(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_u(F, δθ)
  ∂Ψφ(F, E, δθ) = f(δθ) * ∂Ψem_φ(F, E)
  ∂Ψθ(F, E, δθ) = df(δθ) * (Ψm(F) + Ψem(F, E)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)

  ∂Ψuu(F, E, δθ) = f(δθ) * (∂Ψm_uu(F) + ∂Ψem_uu(F, E)) + ∂Ψtm_uu(F, δθ)
  ∂Ψφu(F, E, δθ) = f(δθ) * ∂Ψem_φu(F, E)
  ∂Ψφφ(F, E, δθ) = f(δθ) * ∂Ψem_φφ(F, E)
  ∂Ψθθ(F, E, δθ) = ddf(δθ) * (Ψm(F) + Ψem(F, E)) + ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
  ∂Ψuθ(F, E, δθ) = df(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_uθ(F, δθ)
  ∂Ψφθ(F, E, δθ) = df(δθ) * ∂Ψem_φ(F, E)
  η(F, E, δθ) = -∂Ψθ(F, E, δθ)
  return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ, η)
end


"""
    ThermoElectroMech_Govindjee(thermo, electro, mechano; fθ, dfdθ, d2fdθ2, gθ, dgdθ, d2gdθ2, β=0.0)
    ThermoElectroMech_Govindjee(; thermo, electro, mechano, fθ, dfdθ, d2fdθ2, gθ, dgdθ, d2gdθ2, β=0.0)

Thermo-electro-mechanical model of Govindjee type in the temperature increment `δθ = θ - θr`:

```math
\\Psi(F, E, \\delta\\theta) = \\big(f(\\delta\\theta) + \\Psi_{L3}(\\delta\\theta)\\big)\\,W(F, E)
  + \\big(1 - f(\\delta\\theta)\\big)\\,\\Psi_{er}(F) + \\Psi_{L1}(\\delta\\theta)
```

with `W = Ψm + Ψem` (mechanical and electro-mechanical energies), `Ψer = α θr (J - 1)`,
`ΨL1 = Cv θr (1 - β) (θ/θr (1 - ln(θ/θr)) - 1)` and `ΨL3 = g(δθ) - g(0) - g'(0) δθ`. `thermo` is a
`ThermalModel` (gives `Cv`, `θr` and `α`). `fθ`, `gθ` and their first and second derivatives are functions
of `δθ`; the derivatives of the model are analytical. The functor `model(Λ=1.0)` returns
`(Ψ, ∂Ψ/∂F, ∂Ψ/∂E, ∂Ψ/∂θ, ∂²Ψ/∂F², ∂²Ψ/∂E², ∂²Ψ/∂θ², ∂²Ψ/∂E∂F, ∂²Ψ/∂F∂θ, ∂²Ψ/∂E∂θ, η)` as functions of
`(F, E, δθ)`, with `η = -∂Ψ/∂θ`.
"""
struct ThermoElectroMech_Govindjee{T<:Thermo,E<:Electro,M<:Mechano,F0,F1,F2,G0,G1,G2} <: ThermoElectroMechano{T,E,M}
  thermo::T
  electro::E
  mechano::M
  fθ::F0
  dfdθ::F1
  d2fdθ2::F2
  gθ::G0
  dgdθ::G1
  d2gdθ2::G2
  β::Float64
end

function ThermoElectroMech_Govindjee(thermo::Thermo, electro::Electro, mechano::Mechano;
                                     fθ, dfdθ, d2fdθ2, gθ, dgdθ, d2gdθ2, β::Real=0.0)
  ThermoElectroMech_Govindjee(thermo, electro, mechano, fθ, dfdθ, d2fdθ2, gθ, dgdθ, d2gdθ2, Float64(β))
end

ThermoElectroMech_Govindjee(; thermo::Thermo, electro::Electro, mechano::Mechano, kwargs...) =
  ThermoElectroMech_Govindjee(thermo, electro, mechano; kwargs...)

function (obj::ThermoElectroMech_Govindjee)(Λ::Float64=1.0)
  Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
  Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = obj.electro(Λ)
  (; Cv, θr, α) = obj.thermo
  (; fθ, dfdθ, d2fdθ2, gθ, dgdθ, d2gdθ2, β) = obj
  g0, dg0 = gθ(0.0), dgdθ(0.0)
  c1 = Cv * (1 - β)

  # W = Ψm + Ψem and its derivatives in F
  W(F, E)   = Ψm(F) + Ψem(F, E)
  WF(F, E)  = ∂Ψm_u(F) + ∂Ψem_u(F, E)
  WFF(F, E) = ∂Ψm_uu(F) + ∂Ψem_uu(F, E)
  # a = f + ΨL3, the factor of W
  a(δθ)   = fθ(δθ) + gθ(δθ) - g0 - dg0 * δθ
  da(δθ)  = dfdθ(δθ) + dgdθ(δθ) - dg0
  dda(δθ) = d2fdθ2(δθ) + d2gdθ2(δθ)
  # Ψer = α θr (J - 1)
  Ψer(F)   = α * θr * (det(F) - 1.0)
  Ψer_F(F) = α * θr * cof(F)
  Ψer_FF(F) = ×ᵢ⁴((α * θr) * F)
  # ΨL1
  ΨL1(δθ)   = c1 * θr * ((δθ + θr) / θr * (1.0 - log((δθ + θr) / θr)) - 1.0)
  ΨL1_θ(δθ)  = -c1 * log((δθ + θr) / θr)
  ΨL1_θθ(δθ) = -c1 / (δθ + θr)

  Ψ(F, E, δθ)     = a(δθ) * W(F, E) + (1 - fθ(δθ)) * Ψer(F) + ΨL1(δθ)
  ∂Ψu(F, E, δθ)   = a(δθ) * WF(F, E) + (1 - fθ(δθ)) * Ψer_F(F)
  ∂ΨE(F, E, δθ)   = a(δθ) * ∂Ψem_φ(F, E)
  ∂Ψθ(F, E, δθ)   = da(δθ) * W(F, E) - dfdθ(δθ) * Ψer(F) + ΨL1_θ(δθ)
  ∂ΨFF(F, E, δθ)  = a(δθ) * WFF(F, E) + (1 - fθ(δθ)) * Ψer_FF(F)
  ∂ΨEE(F, E, δθ)  = a(δθ) * ∂Ψem_φφ(F, E)
  ∂2Ψθθ(F, E, δθ) = dda(δθ) * W(F, E) - d2fdθ2(δθ) * Ψer(F) + ΨL1_θθ(δθ)
  ∂ΨEF(F, E, δθ)  = a(δθ) * ∂Ψem_φu(F, E)
  ∂ΨFθ(F, E, δθ)  = da(δθ) * WF(F, E) - dfdθ(δθ) * Ψer_F(F)
  ∂ΨEθ(F, E, δθ)  = da(δθ) * ∂Ψem_φ(F, E)
  η(F, E, δθ)     = -∂Ψθ(F, E, δθ)

  return (Ψ, ∂Ψu, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂2Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η)
end


struct ThermoElectroMech_Bonet{E<:Electro,M<:Mechano} <: ThermoElectroMechano{ThermalVolumetric,E,M}
  thermo::ThermalVolumetric
  electro::E
  mechano::M
  lawel::ThermalLaw
  lawvis::ThermalLaw
  lawelec::ThermalLaw
end

function ThermoElectroMech_Bonet(thermo::ThermalVolumetric, electro::E, mechano::M; mech::ThermalLaw, elec::ThermalLaw) where {E<:Electro,M<:Elasto}
  ThermoElectroMech_Bonet{E,M}(thermo,electro,mechano,mech,mech,elec)
end

function ThermoElectroMech_Bonet(thermo::ThermalVolumetric, electro::E, mechano::M; el::ThermalLaw, vis::ThermalLaw, elec::ThermalLaw) where {E<:Electro,M<:ViscoElastic}
  ThermoElectroMech_Bonet{E,M}(thermo,electro,mechano,el,vis,elec)
end

function ThermoElectroMech_Bonet(thermo::ThermalVolumetric, electro::ThermoElectro{E}, mechano::M; el::ThermalLaw, vis::ThermalLaw) where {E<:Electro,M<:ViscoElastic}
  ThermoElectroMech_Bonet{E,M}(thermo,electro.electro,mechano,el,vis,electro.law)
end

function (obj::ThermoElectroMech_Bonet{<:Electro,<:IsoElastic})(Λ::Float64=1.0)
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo(Λ)
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano(Λ)
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = obj.electro(Λ)
  fe, dfe, ddfe = obj.lawel()
  felec, dfelec, ddfelec = obj.lawelec()

  Ψ(F, E, θ)       =  Ψt(F,θ)      + fe(θ)*Ψm(F)      + felec(θ)*Ψem(F,E)
  ∂Ψ∂F(F, E, θ)    =  ∂Ψt∂F(F,θ)   + fe(θ)*∂Ψm∂F(F)   + felec(θ)*∂Ψem∂F(F,E)
  ∂Ψ∂E(F, E, θ)    =                                  + felec(θ)*∂Ψem∂E(F,E)
  ∂Ψ∂θ(F, E, θ)    =  ∂Ψt∂θ(F,θ)   + dfe(θ)*Ψm(F)     + dfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂FF(F, E, θ)  =  ∂∂Ψt∂FF(F,θ) + fe(θ)*∂∂Ψm∂FF(F) + felec(θ)*∂Ψem∂FF(F,E)
  ∂∂Ψ∂EE(F, E, θ)  =                                  + felec(θ)*∂∂Ψem∂EE(F,E)
  ∂∂Ψ∂θθ(F, E, θ)  =  ∂∂Ψt∂θθ(F,θ) + ddfe(θ)*Ψm(F)    + ddfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂EF(F, E, θ)  =                                  + felec(θ)*∂Ψem∂EF(F,E)
  ∂∂Ψ∂Fθ(F, E, θ)  =  ∂∂Ψt∂Fθ(F,θ) + dfe(θ)*∂Ψm∂F(F)  + dfelec(θ)*∂Ψem∂F(F,E)
  ∂∂Ψ∂Eθ(F, E, θ)  =                                  + dfelec(θ)*∂Ψem∂E(F,E)

  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂EF, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ)
end

function (obj::ThermoElectroMech_Bonet{<:Electro,<:AnisoElastic})(Λ::Float64=1.0)
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo(Λ)
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano(Λ)
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = obj.electro(Λ)
  fe, dfe, ddfe = obj.lawel()
  felec, dfelec, ddfelec = obj.lawelec()

  Ψ(F, E, θ, N)       =  Ψt(F,θ)      + fe(θ)*Ψm(F,N)      + felec(θ)*Ψem(F,E)
  ∂Ψ∂F(F, E, θ, N)    =  ∂Ψt∂F(F,θ)   + fe(θ)*∂Ψm∂F(F,N)   + felec(θ)*∂Ψem∂F(F,E)
  ∂Ψ∂E(F, E, θ, N)    =                                    + felec(θ)*∂Ψem∂E(F,E)
  ∂Ψ∂θ(F, E, θ, N)    =  ∂Ψt∂θ(F,θ)   + dfe(θ)*Ψm(F,N)     + dfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂FF(F, E, θ, N)  =  ∂∂Ψt∂FF(F,θ) + fe(θ)*∂∂Ψm∂FF(F,N) + felec(θ)*∂Ψem∂FF(F,E)
  ∂∂Ψ∂EE(F, E, θ, N)  =                                    + felec(θ)*∂∂Ψem∂EE(F,E)
  ∂∂Ψ∂θθ(F, E, θ, N)  =  ∂∂Ψt∂θθ(F,θ) + ddfe(θ)*Ψm(F,N)    + ddfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂EF(F, E, θ, N)  =                                    + felec(θ)*∂Ψem∂EF(F,E)
  ∂∂Ψ∂Fθ(F, E, θ, N)  =  ∂∂Ψt∂Fθ(F,θ) + dfe(θ)*∂Ψm∂F(F,N)  + dfelec(θ)*∂Ψem∂F(F,E)
  ∂∂Ψ∂Eθ(F, E, θ, N)  =                                    + dfelec(θ)*∂Ψem∂E(F,E)

  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂EF, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ)
end

function (obj::ThermoElectroMech_Bonet{<:Electro,<:ViscoElastic{<:IsoElastic}})(Λ::Float64=1.0)
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo(Λ)
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm(Λ)
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches(Λ)
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = obj.electro(Λ)
  fe, dfe, ddfe = obj.lawel()
  fv, dfv, ddfv = obj.lawvis()
  felec, dfelec, ddfelec = obj.lawelec()

  Ψ(F, E, θ, A...)       =  Ψt(F,θ)      + fe(θ)*Ψe(F)      + fv(θ)*Ψv(F,A...)      + felec(θ)*Ψem(F,E)
  ∂Ψ∂F(F, E, θ, A...)    =  ∂Ψt∂F(F,θ)   + fe(θ)*∂Ψe∂F(F)   + fv(θ)*∂Ψv∂F(F,A...)   + felec(θ)*∂Ψem∂F(F,E)
  ∂Ψ∂E(F, E, θ, A...)    =                                                          + felec(θ)*∂Ψem∂E(F,E)
  ∂Ψ∂θ(F, E, θ, A...)    =  ∂Ψt∂θ(F,θ)   + dfe(θ)*Ψe(F)     + dfv(θ)*Ψv(F,A...)     + dfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂FF(F, E, θ, A...)  =  ∂∂Ψt∂FF(F,θ) + fe(θ)*∂∂Ψe∂FF(F) + fv(θ)*∂∂Ψv∂FF(F,A...) + felec(θ)*∂Ψem∂FF(F,E)
  ∂∂Ψ∂EE(F, E, θ, A...)  =                                                          + felec(θ)*∂∂Ψem∂EE(F,E)
  ∂∂Ψ∂θθ(F, E, θ, A...)  =  ∂∂Ψt∂θθ(F,θ) + ddfe(θ)*Ψe(F)    + ddfv(θ)*Ψv(F,A...)    + ddfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂EF(F, E, θ, A...)  =                                                          + felec(θ)*∂Ψem∂EF(F,E)
  ∂∂Ψ∂Fθ(F, E, θ, A...)  =  ∂∂Ψt∂Fθ(F,θ) + dfe(θ)*∂Ψe∂F(F)  + dfv(θ)*∂Ψv∂F(F,A...)  + dfelec(θ)*∂Ψem∂F(F,E)
  ∂∂Ψ∂Eθ(F, E, θ, A...)  =                                                          + dfelec(θ)*∂Ψem∂E(F,E)

  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂EF, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ)
end

function (obj::ThermoElectroMech_Bonet{<:Electro,<:ViscoElastic{<:AnisoElastic}})(Λ::Float64=1.0)
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo(Λ)
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm(Λ)
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches(Λ)
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = obj.electro(Λ)
  fe, dfe, ddfe = obj.lawel()
  fv, dfv, ddfv = obj.lawvis()
  felec, dfelec, ddfelec = obj.lawelec()

  Ψ(F, E, θ, N, A...)       =  Ψt(F,θ)      + fe(θ)*Ψe(F,N)      + fv(θ)*Ψv(F,A...)      + felec(θ)*Ψem(F,E)
  ∂Ψ∂F(F, E, θ, N, A...)    =  ∂Ψt∂F(F,θ)   + fe(θ)*∂Ψe∂F(F,N)   + fv(θ)*∂Ψv∂F(F,A...)   + felec(θ)*∂Ψem∂F(F,E)
  ∂Ψ∂E(F, E, θ, N, A...)    =                                                            + felec(θ)*∂Ψem∂E(F,E)
  ∂Ψ∂θ(F, E, θ, N, A...)    =  ∂Ψt∂θ(F,θ)   + dfe(θ)*Ψe(F,N)     + dfv(θ)*Ψv(F,A...)     + dfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂FF(F, E, θ, N, A...)  =  ∂∂Ψt∂FF(F,θ) + fe(θ)*∂∂Ψe∂FF(F,N) + fv(θ)*∂∂Ψv∂FF(F,A...) + felec(θ)*∂Ψem∂FF(F,E)
  ∂∂Ψ∂EE(F, E, θ, N, A...)  =                                                            + felec(θ)*∂∂Ψem∂EE(F,E)
  ∂∂Ψ∂θθ(F, E, θ, N, A...)  =  ∂∂Ψt∂θθ(F,θ) + ddfe(θ)*Ψe(F,N)    + ddfv(θ)*Ψv(F,A...)    + ddfelec(θ)*Ψem(F,E)
  ∂∂Ψ∂EF(F, E, θ, N, A...)  =                                                            + felec(θ)*∂Ψem∂EF(F,E)
  ∂∂Ψ∂Fθ(F, E, θ, N, A...)  =  ∂∂Ψt∂Fθ(F,θ) + dfe(θ)*∂Ψe∂F(F,N)  + dfv(θ)*∂Ψv∂F(F,A...)  + dfelec(θ)*∂Ψem∂F(F,E)
  ∂∂Ψ∂Eθ(F, E, θ, N, A...)  =                                                            + dfelec(θ)*∂Ψem∂E(F,E)

  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂EF, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ)
end


function Dissipation(obj::ThermoElectroMech_Bonet)
  tm = ThermoMech_Bonet(obj.thermo, obj.mechano, obj.lawel, obj.lawvis)
  Dtm, ∂Dtm = Dissipation(tm)
  Dtem(F, E, θ, X...) = Dtm(F, θ, X...)
  ∂Dtem(F, E, θ, X...) = ∂Dtm(F, θ, X...)
  return (Dtem, ∂Dtem)
end
