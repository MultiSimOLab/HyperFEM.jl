
function initialize_state(obj::ThermoElectroMechano, points::Measure)
  initialize_state(obj.mechano, points)
end

function update_state!(obj::ThermoElectroMechano, state, F, E, θ, args...)
  update_state!(obj.mechano, state, F, args...)
end

function update_time_step!(obj::ThermoElectroMechano, Δt::Float64)
  update_time_step!(obj.thermo,  Δt)
  update_time_step!(obj.electro, Δt)
  update_time_step!(obj.mechano, Δt)
end

struct ThermoElectroMechModel{T<:Thermo,E<:Electro,M<:Mechano} <: ThermoElectroMechano{T,E,M}
  thermo::T
  electro::E
  mechano::M
  fθ::Function
  dfdθ::Function

  function ThermoElectroMechModel(thermo::T, electro::E, mechano::M; fθ::Function, dfdθ::Function) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ)
  end

  function ThermoElectroMechModel(; thermo::T, electro::E, mechano::M, fθ::Function, dfdθ::Function) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ)
  end

  function (obj::ThermoElectroMechModel)(Λ::Float64=1.0)
    Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.thermo(Λ)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano, Λ)
    Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.thermo, obj.mechano, Λ)
    f(δθ) = (obj.fθ(δθ)::Float64)
    df(δθ) = (obj.dfdθ(δθ)::Float64)

    Ψ(F, E, δθ) = f(δθ) * (Ψm(F) + Ψem(F, E)) + (Ψt(δθ) + Ψtm(F, δθ))
    ∂Ψu(F, E, δθ) = f(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_u(F, δθ)
    ∂Ψφ(F, E, δθ) = f(δθ) * ∂Ψem_φ(F, E)
    ∂Ψθ(F, E, δθ) = df(δθ) * (Ψm(F) + Ψem(F, E)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)

    ∂Ψuu(F, E, δθ) = f(δθ) * (∂Ψm_uu(F) + ∂Ψem_uu(F, E)) + ∂Ψtm_uu(F, δθ)
    ∂Ψφu(F, E, δθ) = f(δθ) * ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E, δθ) = f(δθ) * ∂Ψem_φφ(F, E)
    ∂Ψθθ(F, E, δθ) = ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
    ∂Ψuθ(F, E, δθ) = df(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_uθ(F, δθ)
    ∂Ψφθ(F, E, δθ) = df(δθ) * ∂Ψem_φ(F, E)
    η(F, E, δθ) = -∂Ψθ(F, E, δθ)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ, η)
  end
end


struct ThermoElectroMech_Govindjee{T<:Thermo,E<:Electro,M<:Mechano} <: ThermoElectroMechano{T,E,M}
  thermo::T
  electro::E
  mechano::M
  fθ::Function
  dfdθ::Function
  gθ::Function
  dgdθ::Function
  β::Float64

  function ThermoElectroMech_Govindjee(thermo::T, electro::E, mechano::M; fθ::Function, dfdθ::Function, gθ::Function, dgdθ::Function, β::Float64=0.0) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ, gθ, dgdθ, β)
  end

  function ThermoElectroMech_Govindjee(; thermo::T, electro::E, mechano::M, fθ::Function, dfdθ::Function, gθ::Function, dgdθ::Function, β::Float64=0.0) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ, gθ, dgdθ, β)
  end

  function (obj::ThermoElectroMech_Govindjee)(Λ::Float64=1.0)
    Ψm, _, _ = obj.mechano(Λ)
    Ψem, _, _, _, _, _ = _getCoupling(obj.electro, obj.mechano, Λ)
    f(δθ) = obj.fθ(δθ)
    df(δθ) = obj.dfdθ(δθ)
    g(δθ) = obj.gθ(δθ)
    dg(δθ) = obj.dgdθ(δθ)

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψer(F) = obj.thermo.α * (J(F) - 1.0) * obj.thermo.θr
    ΨL1(δθ) = obj.thermo.Cv * obj.thermo.θr * (1 - obj.β) * ((δθ + obj.thermo.θr) / obj.thermo.θr * (1.0 - log((δθ + obj.thermo.θr) / obj.thermo.θr)) - 1.0)
    ΨL3(δθ) = g(δθ) - g(0.0) - dg(0.0) * δθ

    Ψ(F, E, δθ) = f(δθ) * (Ψm(F) + Ψem(F, E)) + (1 - f(δθ)) * Ψer(F) + ΨL1(δθ) + ΨL3(δθ) * (Ψm(F) + Ψem(F, E))
    ∂Ψ_∂F(F, E, θ) = ForwardDiff.gradient(F -> Ψ(F, get_array(E), θ), get_array(F))
    ∂Ψ_∂E(F, E, θ) = ForwardDiff.gradient(E -> Ψ(get_array(F), E, θ), get_array(E))
    ∂Ψ_∂θ(F, E, θ) = ForwardDiff.derivative(θ -> Ψ(get_array(F), get_array(E), θ), θ)

    ∂Ψu(F, E, θ) = TensorValue(∂Ψ_∂F(F, E, θ))
    ∂ΨE(F, E, θ) = VectorValue(∂Ψ_∂E(F, E, θ))
    ∂Ψθ(F, E, θ) = ∂Ψ_∂θ(F, E, θ)

    ∂2Ψ_∂2E(F, E, θ) = ForwardDiff.hessian(E -> Ψ(get_array(F), E, θ), get_array(E))
    ∂ΨEE(F, E, θ) = TensorValue(∂2Ψ_∂2E(F, E, θ))
    ∂2Ψθθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂θ(get_array(F), get_array(E), θ), θ)

    ∂2Ψ_∂2Eθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂E(get_array(F), get_array(E), θ), θ)
    ∂ΨEθ(F, E, θ) = VectorValue(∂2Ψ_∂2Eθ(F, E, θ))

    ∂2Ψ_∂2F(F, E, θ) = ForwardDiff.hessian(F -> Ψ(F, get_array(E), θ), get_array(F))
    ∂ΨFF(F, E, θ) = TensorValue(∂2Ψ_∂2F(F, E, θ))

    ∂2Ψ_∂2Fθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂F(get_array(F), get_array(E), θ), θ)
    ∂ΨFθ(F, E, θ) = TensorValue(∂2Ψ_∂2Fθ(F, E, θ))

    ∂2Ψ_∂EF(F, E, θ) = ForwardDiff.jacobian(F -> ∂Ψ_∂E(F, get_array(E), θ), get_array(F))
    ∂ΨEF(F, E, θ) = TensorValue(∂2Ψ_∂EF(F, E, θ))

    η(F, E, θ) = -∂Ψθ(F, E, θ)

    return (Ψ, ∂Ψu, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂2Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η)
  end
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

function ThermoElectroMech_Bonet(thermo::ThermalVolumetric, electro::ThermoElectro, mechano::M; el::ThermalLaw, vis::ThermalLaw) where {M<:ViscoElastic}
  ThermoElectroMech_Bonet{E,M}(thermo,electro.electro,mechano,el,vis,electro.law)
end

function (obj::ThermoElectroMech_Bonet{<:Electro,<:Elasto})()
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo()
  Ψm, ∂Ψm∂F, ∂∂Ψm∂FF = obj.mechano()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = _getCoupling(obj.electro, obj.mechano)
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

function (obj::ThermoElectroMech_Bonet{<:Electro,<:ViscoElastic})()
  Ψt, ∂Ψt∂F, ∂Ψt∂θ, ∂∂Ψt∂FF, ∂∂Ψt∂θθ, ∂∂Ψt∂Fθ = obj.thermo()
  Ψe, ∂Ψe∂F, ∂∂Ψe∂FF = obj.mechano.longterm()
  Ψv, ∂Ψv∂F, ∂∂Ψv∂FF = obj.mechano.branches()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂∂Ψem∂EE = _getCoupling(obj.electro, obj.mechano)
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


function Dissipation(obj::ThermoElectroMech_Bonet)
  tm = ThermoMech_Bonet(obj.thermo, obj.mechano, obj.lawel, obj.lawvis)
  Dtm, ∂Dtm = Dissipation(tm)
  Dtem(F, E, θ, X...) = Dtm(F, θ, X...)
  ∂Dtem(F, E, θ, X...) = ∂Dtm(F, θ, X...)
  return (Dtem, ∂Dtem)
end
