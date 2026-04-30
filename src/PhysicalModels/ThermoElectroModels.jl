
struct ThermoElectroModel{E<:Electro} <: ThermoElectro{E}
  electro::E
  law::ThermalLaw

  function ThermoElectroModel(electro::E, law::ThermalLaw) where {E <: Electro}
    new{E}(electro, law)
  end
end

function (obj::ThermoElectroModel)()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂∂Ψem∂FF, ∂∂Ψem∂EF, ∂∂Ψem∂EE = obj()
  f, df, ddf = law()

  Ψ(F, E, θ)       =  f(θ)*Ψem(F,E)
  ∂Ψ∂F(F, E, θ)    =  f(θ)*∂Ψem∂F(F,E)
  ∂Ψ∂E(F, E, θ)    =  f(θ)*∂Ψem∂E(F,E)
  ∂Ψ∂θ(F, E, θ)    =  df(θ)*Ψem(F,E)
  ∂∂Ψ∂FF(F, E, θ)  =  f(θ)*∂∂Ψem∂FF(F,E)
  ∂∂Ψ∂EE(F, E, θ)  =  f(θ)*∂∂Ψem∂EE(F,E)
  ∂∂Ψ∂θθ(F, E, θ)  =  ddf(θ)*Ψem(F,E)
  ∂∂Ψ∂EF(F, E, θ)  =  f(θ)*∂∂Ψem∂EF(F,E)
  ∂∂Ψ∂Fθ(F, E, θ)  =  df(θ)*∂Ψem∂F(F,E)
  ∂∂Ψ∂Eθ(F, E, θ)  =  df(θ)*∂Ψem∂E(F,E)

  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂EF, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ)
end
