
struct ElectroMechModel{E<:Electro,M<:Mechano} <: ElectroMechano{E,M}
  electro::E
  mechano::M

  function ElectroMechModel(electro::E, mechano::M) where {E<:Electro,M<:Mechano}
    new{E,M}(electro, mechano)
  end

  function ElectroMechModel(; electro::E, mechano::M) where {E<:Electro,M<:Mechano}
    new{E,M}(electro, mechano)
  end
end

"""
    HessianRegularization(model::ElectroMechano; δ=1e-6, switch=nothing, symmetrize=false)

Electro-mechanical model (in E) with its **total** mechanical tangent `∂²Ψ/∂F∂F(F, E)` projected at every point —
the elastic part and the electro-mechanical (Maxwell) part `∂²Ψem/∂F∂F` together, at fixed E: the eigenvalues are
clamped from below at `δ` (`regularize_hessian`). Energy, stresses, the coupling `∂²Ψ/∂E∂F` and the electric block
`∂²Ψ/∂E∂E` are unchanged. In the E formulation the electric block is negative definite, so the Schur complement
`K_FF − K_FE K_EE⁻¹ K_EF` adds a positive semi-definite term to `K_FF`: a loss of stability (pull-in) comes from
`K_FF` itself, and with it projected the two diagonal blocks of the positive-real form are positive definite (what
the block preconditioners and the hybrid Newton's projected step need). With a [`TangentSwitch`](@ref) the
projection applies only while `is_projected(switch)` (the `HybridTangent` of the Newton, register the switch in the
driver with `tangent_switch`). Wrapping only the mechanical model instead (`HessianRegularization(mechano)` inside
`ElectroMechModel`) leaves the Maxwell part unprojected.
"""
struct ElectroMechHessianRegularization{E,M,P<:ElectroMechano{E,M},S} <: ElectroMechano{E,M}
  model::P
  δ::Float64
  switch::S
  symmetrize::Bool
end
HessianRegularization(model::ElectroMechano; δ::Float64=1.0e-6, switch=nothing, symmetrize::Bool=false) =
  ElectroMechHessianRegularization(model, δ, switch, symmetrize)

_projected(::Nothing) = true
_projected(s::TangentSwitch) = is_projected(s)

function (obj::ElectroMechHessianRegularization)(args...; kwargs...)
  Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE = obj.model(args...; kwargs...)
  δ, sym, switch = obj.δ, obj.symmetrize, obj.switch
  function ∂Ψ∂FFr(F, E, vars...)
    C = ∂Ψ∂FF(F, E, vars...)
    _projected(switch) ? regularize_hessian(sym ? _symmetrize(C) : C, δ) : C
  end
  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FFr, ∂Ψ∂EF, ∂Ψ∂EE)
end

# the state, the time step and the dissipation are those of the wrapped model
update_time_step!(obj::ElectroMechHessianRegularization, Δt::Float64) = update_time_step!(obj.model, Δt)
Gridap.CellData.CellState(obj::ElectroMechHessianRegularization, args...; kwargs...) = CellState(obj.model, args...; kwargs...)
initialize_state(obj::ElectroMechHessianRegularization) = initialize_state(obj.model)
Gridap.CellData.update_state!(obj::ElectroMechHessianRegularization, state, args...; kwargs...) = update_state!(obj.model, state, args...; kwargs...)
Dissipation(obj::ElectroMechHessianRegularization) = Dissipation(obj.model)

function (+)(Model1::Electro, Model2::Mechano)
  ElectroMechModel(Model1, Model2)
end

function (+)(Model1::Mechano, Model2::Electro)
  ElectroMechModel(Model2, Model1)
end

function (obj::ElectroMechModel{<:Electro,<:IsoElastic})(Λ::Float64=1.0)
  Ψm, ∂Ψm∂F, ∂Ψm∂FF = obj.mechano()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = obj.electro()
  Ψ(F, E)     = Ψm(F)     + Ψem(F, E)
  ∂Ψ∂F(F, E)  = ∂Ψm∂F(F)  + ∂Ψem∂F(F, E)
  ∂Ψ∂E(F, E)  =             ∂Ψem∂E(F, E)
  ∂Ψ∂FF(F, E) = ∂Ψm∂FF(F) + ∂Ψem∂FF(F, E)
  ∂Ψ∂EF(F, E) =             ∂Ψem∂EF(F, E)
  ∂Ψ∂EE(F, E) =             ∂Ψem∂EE(F, E)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE)
end

function (obj::ElectroMechModel{<:Electro,<:AnisoElastic})(Λ::Float64=1.0)
  Ψm, ∂Ψm∂F, ∂Ψm∂FF = obj.mechano()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = obj.electro()
  Ψ(F, E, N)     = Ψm(F, N)     + Ψem(F, E)
  ∂Ψ∂F(F, E, N)  = ∂Ψm∂F(F, N)  + ∂Ψem∂F(F, E)
  ∂Ψ∂E(F, E, N)  =                ∂Ψem∂E(F, E)
  ∂Ψ∂FF(F, E, N) = ∂Ψm∂FF(F, N) + ∂Ψem∂FF(F, E)
  ∂Ψ∂EF(F, E, N) =                ∂Ψem∂EF(F, E)
  ∂Ψ∂EE(F, E, N) =                ∂Ψem∂EE(F, E)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE)
end

function (obj::ElectroMechModel{<:Electro,<:ViscoElastic{<:IsoElastic}})(Λ::Float64=1.0)
  Ψm, ∂Ψm∂F, ∂Ψm∂FF = obj.mechano()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = obj.electro()
  Ψ(F, E, Fn, A...)     = Ψm(F, Fn, A...)     + Ψem(F, E)
  ∂Ψ∂F(F, E, Fn, A...)  = ∂Ψm∂F(F, Fn, A...)  + ∂Ψem∂F(F, E)
  ∂Ψ∂E(F, E, Fn, A...)  =                       ∂Ψem∂E(F, E)
  ∂Ψ∂FF(F, E, Fn, A...) = ∂Ψm∂FF(F, Fn, A...) + ∂Ψem∂FF(F, E)
  ∂Ψ∂EF(F, E, Fn, A...) =                       ∂Ψem∂EF(F, E)
  ∂Ψ∂EE(F, E, Fn, A...) =                       ∂Ψem∂EE(F, E)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE)
end

function (obj::ElectroMechModel{<:Electro,<:ViscoElastic{<:AnisoElastic}})(Λ::Float64=1.0)
  Ψm, ∂Ψm∂F, ∂Ψm∂FF = obj.mechano()
  Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = obj.electro()
  Ψ(F, E, n, Fn, A...)     = Ψm(F, n, Fn, A...)     + Ψem(F, E)
  ∂Ψ∂F(F, E, n, Fn, A...)  = ∂Ψm∂F(F, n, Fn, A...)  + ∂Ψem∂F(F, E)
  ∂Ψ∂E(F, E, n, Fn, A...)  =                          ∂Ψem∂E(F, E)
  ∂Ψ∂FF(F, E, n, Fn, A...) = ∂Ψm∂FF(F, n, Fn, A...) + ∂Ψem∂FF(F, E)
  ∂Ψ∂EF(F, E, n, Fn, A...) =                          ∂Ψem∂EF(F, E)
  ∂Ψ∂EE(F, E, n, Fn, A...) =                          ∂Ψem∂EE(F, E)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE)
end

function update_time_step!(obj::ElectroMechModel, Δt::Float64)
  update_time_step!(obj.electro, Δt)
  update_time_step!(obj.mechano, Δt)
end

function Gridap.CellData.CellState(obj::ElectroMechModel, args...; kwargs...)
  CellState(obj.mechano, args...; kwargs...)
end

function initialize_state(obj::ElectroMechModel)
  initialize_state(obj.mechano)
end

function Gridap.CellData.update_state!(obj::ElectroMechModel, state, F, E, args...; kwargs...)
  update_state!(obj.mechano, state, F, args...; kwargs...)
end

function Dissipation(obj::ElectroMechModel)
  Dvis = Dissipation(obj.mechano)
  D(F, E, X...) = Dvis(F, X...)
end

struct FlexoElectroModel{EM<:ElectroMechano} <: FlexoElectro{EM}
  electromechano::EM
  κ::Float64

  function FlexoElectroModel(electro::E, mechano::M; κ=1.0) where {E<:Electro,M<:Mechano}
    physmodel = ElectroMechModel(electro, mechano)
    new{ElectroMechModel{E,M}}(physmodel, κ)
  end

  function FlexoElectroModel(; electro::E, mechano::M, κ=1.0) where {E<:Electro,M<:Mechano}
    physmodel = ElectroMechModel(electro, mechano)
    new{ElectroMechModel{E,M}}(physmodel, κ)
  end

  function (obj::FlexoElectroModel)(Λ::Float64=1.0)
    e₁ = VectorValue(1.0, 0.0, 0.0)
    e₂ = VectorValue(0.0, 1.0, 0.0)
    e₃ = VectorValue(0.0, 0.0, 1.0)
    # Φ(ϕ₁,ϕ₂,ϕ₃)=ϕ₁ ⊗₁² e₁+ϕ₂ ⊗₁² e₂+ϕ₃ ⊗₁² e₃
    f1(δϕ) = δϕ ⊗₁² e₁
    f2(δϕ) = δϕ ⊗₁² e₂
    f3(δϕ) = δϕ ⊗₁² e₃
    Φ(ϕ₁, ϕ₂, ϕ₃) = (f1 ∘ (ϕ₁) + f2 ∘ (ϕ₂) + f3 ∘ (ϕ₃))

    Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE = obj.electromechano(Λ)
    return Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE, Φ
  end
end
