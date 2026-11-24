

# Threshold of the regularisation of ln J (`logreg`): below it, ln J is replaced by its quadratic extension,
# in the energies and in their derivatives.
const LOGREG_THRESHOLD = 0.01

# ============================================
# Volumetric Mechanical models
# ============================================

"""
    VolumetricEnergy(; λ)

Volumetric energy term

```math
\\Psi = \\frac{\\lambda}{2} (J-1)^2
```

with `λ` the volumetric penalty (bulk) parameter. It is usually added to an isochoric model, e.g.
`IsochoricNeoHookean3D(μ=μ) + VolumetricEnergy(λ=λ)`. 3D only.
"""
struct VolumetricEnergy <: Volumetric
  λ::Float64
  function VolumetricEnergy(; λ::Float64)
    new(λ)
  end
end

function tangent(obj::Volumetric)
  _, _, ∂∂Ψ = obj()
  ∂∂Ψ(I3D)[1]
end

function (obj::VolumetricEnergy)()
  λ = obj.λ
  J, H = det, cof
  Ψ(F) = (λ / 2.0) * (J(F) - 1)^2
  ∂Ψ_∂J(F) = λ * (J(F) - 1)
  ∂Ψ2_∂J2(F) = λ
  ∂Ψu(F) = ∂Ψ_∂J(F) * H(F)
  ∂Ψuu(F) = ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂J(F) * F)
  return (Ψ, ∂Ψu, ∂Ψuu)
end


"""
    CoerciveVolumetric(; κ)

Coercive volumetric energy term

```math
\\Psi = \\frac{\\kappa}{4} (J^2-1-2\\ln J)
```

which vanishes with its derivative `∂Ψ/∂J = κ/2 (J - 1/J)` at `J = 1` and grows as `J → 0` and `J → ∞`. `ln J` is
regularised below `J = 0.01` (`logreg`), so the energy stays finite for inverted elements. 3D only.
"""
struct CoerciveVolumetric <: Volumetric
  κ::Float64
  function CoerciveVolumetric(; κ::Float64)
    new(κ)
  end
end

function (obj::CoerciveVolumetric)()
  κ = obj.κ
  J, H = det, cof
  Ψ(F) = (κ / 4) * (J(F)^2 - 1 - 2logreg(J(F)))
  ∂Ψ_∂J(F) = (κ / 2) * (J(F) - ∂log∂J(J(F)))
  ∂Ψ2_∂J2(F) = (κ / 2) * (1 - ∂∂log∂JJ(J(F)))
  ∂Ψu(F) = ∂Ψ_∂J(F) * H(F)
  ∂Ψuu(F) = ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂J(F) * F)
  return (Ψ, ∂Ψu, ∂Ψuu)
end


"""
    LogBarrier(; κ, p=2)

Volumetric barrier

```math
\\Psi = \\frac{\\kappa}{2p} (\\ln J)^{2p}
```

negligible at moderate `J` and steep as `J → 0` (the third medium of contact problems, with
`κ = kb·μ·fc` and `p = 2`); `Ψ`, `∂Ψ/∂F` and `∂²Ψ/∂F∂F` are `NaN` for `J ≤ 0`, so that a trial state
with inverted elements is rejected by an energy line search. 2D and 3D (`∂H/∂F` dispatched on the
dimension). Composes with any `IsoElastic` model (`ARAP2D(μ=…) + LogBarrier(κ=…, p=2)`).
"""
struct LogBarrier <: Volumetric
  κ::Float64
  p::Int
  function LogBarrier(; κ::Float64, p::Int=2)
    p >= 1 || throw(ArgumentError("LogBarrier: p must be ≥ 1"))
    new(κ, p)
  end
end

_lnJ(J) = J > 0 ? log(J) : oftype(J, NaN)

function (obj::LogBarrier)()
  κ, p = obj.κ, obj.p
  J, H = det, cof
  Ψ(F) = (κ / (2p)) * _lnJ(J(F))^(2p)
  ∂Ψ_∂J(F) = κ * _lnJ(J(F))^(2p - 1) / J(F)
  ∂Ψ2_∂J2(F) = κ * ((2p - 1) * _lnJ(J(F))^(2p - 2) - _lnJ(J(F))^(2p - 1)) / J(F)^2
  ∂Ψu(F) = ∂Ψ_∂J(F) * H(F)
  ∂Ψuu(F) = ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ∂H∂F(F)
  return (Ψ, ∂Ψu, ∂Ψuu)
end

# ============================================
# Regularization of Mechanical models
# ============================================

"""
    TangentSwitch()

Flag shared by the [`HessianRegularization`](@ref) wrappers of the constitutive models of a problem
and the Newton solver (`HybridTangent`): while `is_projected(switch)` the wrapped tangents are
projected (eigenvalues clamped from below), otherwise they are the consistent ones. The solver flips
it with `set_projected!` through the driver (`tangent_switch = switch`); the weak form is written
once and reads the flag at every quadrature point.
"""
mutable struct TangentSwitch
  projected::Bool
end
TangentSwitch() = TangentSwitch(false)
is_projected(s::TangentSwitch) = s.projected
set_projected!(s::TangentSwitch, v::Bool) = (s.projected = v; s)

"""
    HessianRegularization(mechano; δ=1e-6, switch=nothing, symmetrize=false)

Mechanical model with the material tangent `∂²Ψ/∂F∂F` of `mechano` projected at every point: its
eigenvalues are clamped from below at `δ` (`regularize_hessian`); energy and stress are unchanged.
With a [`TangentSwitch`](@ref) the projection applies only while `is_projected(switch)`, so that
the hybrid Newton switches between the consistent and the projected tangent without a second weak
form. `symmetrize = true` diagonalises `(C + Cᵀ)/2` (for tangents that are not symmetric to
round-off); by default the upper triangle is used, as `eigen(Symmetric(C))`.
"""
struct HessianRegularization{S} <: Mechano
  mechano::Mechano
  δ::Float64
  switch::S
  symmetrize::Bool

  function HessianRegularization(mechano::Mechano; δ::Float64=1.0e-6, switch=nothing, symmetrize::Bool=false)
    new{typeof(switch)}(mechano, δ, switch, symmetrize)
  end

  function HessianRegularization(; mechano::Mechano, δ::Float64=1.0e-6, switch=nothing, symmetrize::Bool=false)
    new{typeof(switch)}(mechano, δ, switch, symmetrize)
  end
end

_symmetrize(C) = 0.5 * (C + C')

function (obj::HessianRegularization{Nothing})()
  Ψs, ∂Ψs, ∂2Ψs = obj.mechano()
  δ, sym = obj.δ, obj.symmetrize
  # extra arguments (the fibre direction of an anisotropic model) are passed through
  ∂2Ψ(F, args::Vararg{Any,K}) where {K} = (C = ∂2Ψs(F, args...); regularize_hessian(sym ? _symmetrize(C) : C, δ))
  return (Ψs, ∂Ψs, ∂2Ψ)
end

function (obj::HessianRegularization{TangentSwitch})()
  Ψs, ∂Ψs, ∂2Ψs = obj.mechano()
  δ, sym, switch = obj.δ, obj.symmetrize, obj.switch
  ∂2Ψ(F, args::Vararg{Any,K}) where {K} = (C = ∂2Ψs(F, args...); is_projected(switch) ? regularize_hessian(sym ? _symmetrize(C) : C, δ) : C)
  return (Ψs, ∂Ψs, ∂2Ψ)
end

# the state, the time step and the dissipation are those of the regularized model (e.g. a viscous one)
update_time_step!(obj::HessianRegularization, Δt::Float64) = update_time_step!(obj.mechano, Δt)
Gridap.CellData.CellState(obj::HessianRegularization, args...; kwargs...) = CellState(obj.mechano, args...; kwargs...)
initialize_state(obj::HessianRegularization) = initialize_state(obj.mechano)
Gridap.CellData.update_state!(obj::HessianRegularization, state, args...; kwargs...) = update_state!(obj.mechano, state, args...; kwargs...)
Dissipation(obj::HessianRegularization) = Dissipation(obj.mechano)


struct Hessian∇JRegularization <: Mechano
  mechano::Mechano
  δ::Float64
  κ::Float64

  function Hessian∇JRegularization(mechano::Mechano; δ::Float64=1.0e-6, κ::Float64=1.0)
    new(mechano, δ, κ)
  end

  function Hessian∇JRegularization(; mechano::Mechano, δ::Float64=1.0e-6, κ::Float64=1.0)
    new(mechano, δ, κ)
  end

  function (obj::Hessian∇JRegularization)()
    Ψs, ∂Ψs, ∂2Ψs = obj.mechano()
    δ, κ = obj.δ, obj.κ

    J, H = det, cof
    Ψ(F, Jh) = Ψs(F) + 0.5 * κ * (J(F) - Jh)^2
    ∂Ψ(F, Jh) = ∂Ψs(F) + κ * (J(F) - Jh) * H(F)
    ∂2Ψ_(F, Jh) = ∂2Ψs(F) + κ * (H(F) ⊗ H(F)) + κ * (J(F) - Jh) * ∂H∂F(F)

    ∂2Ψ(F, Jh) = regularize_hessian(∂2Ψ_(F, Jh), δ)
    return (Ψ, ∂Ψ, ∂2Ψ)
  end
end

# ======================
# Energy interpolations
# ======================
struct EnergyInterpolationScheme <: IsoElastic
  p::Float64
  model1::IsoElastic
  model2::IsoElastic

  function EnergyInterpolationScheme(model1::IsoElastic, model2::IsoElastic; p::Float64=3.0)
    new(p, model1, model2)
  end

  function (obj::EnergyInterpolationScheme)()
    Ψs, ∂Ψs, ∂2Ψs = obj.model1()
    Ψv, ∂Ψv, ∂2Ψv = obj.model2()
    p = obj.p

    Ψ(ρ, F) = ρ^p * Ψs(F) + (1 - ρ^p) * Ψv(F)
    DΨ_Dρ(ρ, F) = p * ρ^(p - 1) * Ψs(F) - (p * ρ^(p - 1)) * Ψv(F)

    ∂Ψ(ρ, F) = ρ^p * ∂Ψs(F) + (1 - ρ^p) * ∂Ψv(F)
    D∂Ψ_Dρ(ρ, F) = p * ρ^(p - 1) * ∂Ψs(F) - (p * ρ^(p - 1)) * ∂Ψv(F)

    ∂2Ψ(ρ, F) = ρ^p * ∂2Ψs(F) + (1 - ρ^p) * ∂2Ψv(F)
    D∂2Ψ_Dρ(ρ, F) = p * ρ^(p - 1) * ∂2Ψs(F) - (p * ρ^(p - 1)) * ∂2Ψv(F)

    return (Ψ, ∂Ψ, ∂2Ψ, DΨ_Dρ, D∂Ψ_Dρ, D∂2Ψ_Dρ)
  end
end


struct ComposedIsoElastic <: IsoElastic
  Model1::IsoElastic
  Model2::IsoElastic
  function ComposedIsoElastic(model1::IsoElastic, model2::IsoElastic)
    new(model1, model2)
  end
  function (obj::ComposedIsoElastic)()
    Ψ1, ∂Ψu1, ∂Ψuu1 = obj.Model1()
    Ψ2, ∂Ψu2, ∂Ψuu2 = obj.Model2()
    Ψ(x) = Ψ1(x) + Ψ2(x)
    ∂Ψ(x) = ∂Ψu1(x) + ∂Ψu2(x)
    ∂∂Ψ(x) = ∂Ψuu1(x) + ∂Ψuu2(x)
    return (Ψ, ∂Ψ, ∂∂Ψ)
  end
end


function (+)(Model1::IsoElastic, Model2::IsoElastic)
  ComposedIsoElastic(Model1, Model2)
end

struct ComposedAnisoElastic <: AnisoElastic
  Model1::IsoElastic
  Model2::AnisoElastic
  function ComposedAnisoElastic(model1::IsoElastic, model2::AnisoElastic)
    new(model1, model2)
  end
  function (obj::ComposedAnisoElastic)()
    DΨ1 = obj.Model1()
    DΨ2 = obj.Model2()
    Ψ, ∂Ψ, ∂∂Ψ = map(_aniso_sum, DΨ1, DΨ2)
    return (Ψ, ∂Ψ, ∂∂Ψ)
  end
end


# x -> ψ1(x) + ψ2(x, N...) for any number of fibre directions N. `Vararg{Any,K}` makes Julia specialize on the
# number of arguments: a vararg that is only passed through is not specialized and allocates at every call.
function _aniso_sum(ψ1, ψ2)
  ψ(x, N::Vararg{Any,K}) where {K} = ψ1(x) + ψ2(x, N...)
  return ψ
end

function (+)(Model1::IsoElastic, Model2::AnisoElastic)
  ComposedAnisoElastic(Model1, Model2)
end
function (+)(Model1::AnisoElastic, Model2::IsoElastic)
  ComposedAnisoElastic(Model2, Model1)
end

struct MultiAnisoElastic <: AnisoElastic
  Models::NTuple{N,AnisoElastic} where N
end

Base.hcat(a::AnisoElastic...) = MultiAnisoElastic(a)


function (obj::MultiAnisoElastic)(args...)
  DΨ = map(a -> a(args...), obj.Models)
  Ψα = getindex.(DΨ, 1)
  ∂Ψα∂F = getindex.(DΨ, 2)
  ∂Ψα∂FF = getindex.(DΨ, 3)
  Ψ(F, N) = mapreduce((Ψi, Ni) -> Ψi(F, Ni), +, Ψα, N)
  ∂Ψ∂F(F, N) = mapreduce((∂Ψi∂F, Ni) -> ∂Ψi∂F(F, Ni), +, ∂Ψα∂F, N)
  ∂Ψ∂FF(F, N) = mapreduce((∂Ψi∂FF, Ni) -> ∂Ψi∂FF(F, Ni), +, ∂Ψα∂FF, N)
  (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end


# ===================
# Mechanical models
# ===================

"""
    Yeoh3D(; λ, C10, C20, C30)

Compressible Yeoh model in 3D

```math
\\Psi = C_{10}(I_1-3) + C_{20}(I_1-3)^2 + C_{30}(I_1-3)^3 - 2C_{10}\\ln J + \\frac{\\lambda}{2}(J-1)^2,
\\qquad I_1 = F:F,\\quad J = \\det F
```

The functor `model()` returns `(Ψ, ∂Ψ/∂F, ∂²Ψ/∂F²)`; `ln J` is regularised below `J = 0.01` (`logreg`)
in the energy and in its derivatives.
"""
struct Yeoh3D <: IsoElastic
  λ::Float64
  C10::Float64
  C20::Float64
  C30::Float64
  function Yeoh3D(; λ::Float64, C10::Float64, C20::Float64, C30::Float64)
    new(λ, C10, C20, C30)
  end

  function (obj::Yeoh3D)()

    λ, C10, C20, C30 = obj.λ, obj.C10, obj.C20, obj.C30

    J, H = det, cof

    # Free energy
    I1(F) = tr((F)' * F)
    ψ(F) = C10 * (I1(F) - 3) + C20 * (I1(F) - 3)^2 + C30 * (I1(F) - 3)^3 - 2 * C10 * logreg(J(F)) + 0.5 * λ * (J(F) - 1)^2

    # First Piola-Kirchhoff
    ∂ψ_∂I1(F) = C10 + 2 * C20 * (I1(F) - 3) + 3 * C30 * (I1(F) - 3)^2
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂ψ_∂J(F) = -2 * C10 * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂ψu(F) = 2 * ∂ψ_∂I1(F) * F + ∂ψ_∂J(F) * H(F)

    # Elasticity
    ∂2ψ_∂I1I1(F) = 2 * C20 + 6 * C30 * (I1(F) - 3)
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂2ψ_∂JJ(F) = -2 * C10 * ∂log2∂J2(J(F)) + λ
    ∂ψuu(F) = 4 * ∂2ψ_∂I1I1(F) * (F ⊗ F) + 2 * ∂ψ_∂I1(F) * 𝕀3D + ∂2ψ_∂JJ(F) * (H(F) ⊗ H(F)) + ∂ψ_∂J(F) * (𝕀3D × F)


    return (ψ, ∂ψu, ∂ψuu)
  end
end

struct LinearElasticity2D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  function LinearElasticity2D(; λ::Float64, μ::Float64, ρ::Float64=0.0)
    new(λ, μ, ρ)
  end

  function (obj::LinearElasticity2D)()
    λ, μ = obj.λ, obj.μ
    δᵢₖδⱼₗ, δᵢₗδⱼₖ, δᵢⱼδₖₗ = I2D ⊗ᵢₖʲˡ I2D, I2D ⊗ᵢₗʲᵏ I2D, I2D ⊗ I2D
    ε(F) = 0.5(F + F') - I2D
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ + δᵢₗδⱼₖ) + λ * δᵢⱼδₖₗ
    ∂Ψu(F) = ∂Ψuu(F) ⋅² (F - I2D)
    Ψ(F) = μ * sum(ε(F) .* ε(F)) + 0.5 * λ * tr(ε(F))^2
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct LinearElasticity3D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  function LinearElasticity3D(; λ::Float64, μ::Float64, ρ::Float64=0.0)
    new(λ, μ, ρ)
  end

  function (obj::LinearElasticity3D)()
    λ, μ = obj.λ, obj.μ
    δᵢₖδⱼₗ, δᵢₗδⱼₖ, δᵢⱼδₖₗ = I3D ⊗ᵢₖʲˡ I3D, I3D ⊗ᵢₗʲᵏ I3D, I3D ⊗ I3D
    ε(F) = 0.5(F + F') - I3D
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ + δᵢₗδⱼₖ) + λ * δᵢⱼδₖₗ
    ∂Ψu(F) = ∂Ψuu(F) ⋅² (F - I3D)
    Ψ(F) = μ * sum(ε(F) .* ε(F)) + 0.5 * λ * tr(ε(F))^2
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NeoHookean3D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  function NeoHookean3D(; λ::Float64, μ::Float64, ρ::Float64=0.0)
    new(λ, μ, ρ)
  end

  function (obj::NeoHookean3D)()
    λ, μ = obj.λ, obj.μ
    J, H = det, cof
    Ψ(F) = μ / 2 * tr((F)' * F) - μ * logreg(J(F)) + (λ / 2) * (J(F) - 1)^2 - 3.0 * (μ / 2.0)

    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))

    ∂Ψ_∂J(F) = -μ * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψu(F) = μ * F + ∂Ψ_∂J(F) * H(F)
    ∂Ψ2_∂J2(F) = -μ * ∂log2∂J2(J(F)) + λ
    ∂Ψuu(F) = μ * 𝕀3D + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct Gent2D <: IsoElastic
  λ::Float64
  μ::Float64
  Jm::Float64
  γ::Float64
  ρ::Float64

  function Gent2D(; λ::Float64, μ::Float64, Jm::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ, Jm, γ, ρ)
  end

  function (obj::Gent2D)()
    λ, μ, Jm, γ, ρ = obj.λ, obj.μ, obj.Jm, obj.γ, obj.ρ

    J, H = det, cof
    I1(F) = tr(F' * F) + 1.0

    Ψiso(F) = -μ * Jm / 2.0 * log(1.0 - (I1(F) - 3.0) / Jm)
    Ψvol(F) = -μ * log(J(F)) + λ * (J(F)^γ + J(F)^(-γ)) - 2.0 * λ
    Ψ(F) = Ψiso(F) + Ψvol(F)

    ∂Ψ_∂F(F) = F * μ / (1.0 - (I1(F) - 3.0) / Jm)
    ∂Ψ_∂J(F) = -μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1.0) - J(F)^(-γ - 1.0))
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = (μ / (1.0 - (I1(F) - 3.0) / Jm)) * 𝕀2D + 2.0 * (μ / (Jm * (1.0 - (I1(F) - 3.0) / Jm)^2)) * (F ⊗ F)
    ∂Ψ2_∂JJ(F) = μ * (1.0 / (J(F)^2)) +
                 λ * γ * ((γ - 1.0) * J(F)^(γ - 2.0) + (γ + 1.0) * J(F)^(-γ - 2.0))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) +
              ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) +
              ∂Ψ_∂J(F) * ∂H∂F(F)

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


"""
    Gent3D(; λ, μ, Jm, γ, ρ=0.0)

Compressible Gent model in 3D, with the volumetric energy of `Gent2D`:

```math
\\Psi = -\\frac{\\mu J_m}{2} \\ln\\left(1 - \\frac{I_1 - 3}{J_m}\\right) - \\mu \\ln J
       + \\lambda \\left(J^\\gamma + J^{-\\gamma}\\right) - 2\\lambda,
\\qquad I_1 = \\mathrm{tr}(F^T F),\\quad J = \\det F.
```

`Jm` limits the chain extensibility: the model requires `I₁ - 3 < Jm` (beyond it `Ψ` throws a `DomainError`
and the derivatives are meaningless). For `Jm → ∞` the isochoric part tends to the neo-Hookean `μ/2 (I₁ - 3)`.
Stress-free in the reference configuration, with shear modulus `μ` and first Lamé parameter `2λγ² + 2μ/Jm`:
`λ` is not the Lamé parameter (with `γ = 1`, the same `λ` as in `NeoHookean3D` doubles it).
"""
struct Gent3D <: IsoElastic
  λ::Float64
  μ::Float64
  Jm::Float64
  γ::Float64
  ρ::Float64

  function Gent3D(; λ::Float64, μ::Float64, Jm::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ, Jm, γ, ρ)
  end

  function (obj::Gent3D)()
    λ, μ, Jm, γ = obj.λ, obj.μ, obj.Jm, obj.γ

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    I1(F) = tr(F' * F)
    g(F) = 1.0 - (I1(F) - 3.0) / Jm

    Ψ(F) = -μ * Jm / 2.0 * log(g(F)) - μ * log(J(F)) + λ * (J(F)^γ + J(F)^(-γ)) - 2.0 * λ

    ∂Ψ_∂J(F) = -μ / J(F) + λ * γ * (J(F)^(γ - 1.0) - J(F)^(-γ - 1.0))
    ∂Ψu(F) = (μ / g(F)) * F + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂JJ(F) = μ / J(F)^2 + λ * γ * ((γ - 1.0) * J(F)^(γ - 2.0) + (γ + 1.0) * J(F)^(-γ - 2.0))
    ∂Ψuu(F) = (μ / g(F)) * 𝕀3D + (2.0 * μ / (Jm * g(F)^2)) * (F ⊗ F) +
              ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end



struct MooneyRivlin3D <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  function MooneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, ρ)
  end

  function (obj::MooneyRivlin3D)()
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    J, H = det, cof

    Ψ(F) = μ1 / 2 * tr((F)' * F) + μ2 / 2.0 * tr((H(F))' * H(F)) - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 - (3.0 / 2.0) * (μ1 + μ2)
    ∂Ψ_∂F(F) = μ1 * F
    ∂Ψ_∂H(F) = μ2 * H(F)
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = μ1 * 𝕀3D + μ2 * (F × (𝕀3D × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct MooneyRivlin2D <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64

  function MooneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, ρ)
  end

  function (obj::MooneyRivlin2D)()
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    J, H = det, cof
    Ψ(F) = (μ1 / 2 + μ2 / 2) * tr((F)' * F) + μ2 / 2.0 * J(F)^2 - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 - (μ1 + μ2) - μ2 / 2
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F) = μ2 * J(F) - (μ1 + 2 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = μ2 - (μ1 + 2 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = (μ1 + μ2) * F + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = (μ1 + μ2) * 𝕀2D + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearMooneyRivlin3D <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  α1::Float64
  α2::Float64
  ρ::Float64
  function NonlinearMooneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, α1::Float64, α2::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, α1, α2, ρ)
  end

  function (obj::NonlinearMooneyRivlin3D)()
    λ, μ1, μ2, α1, α2 = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2
    J, H = det, cof

    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F))^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^α2 - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2

    ∂Ψ_∂F(F) = (μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1)) * F
    ∂Ψ_∂H(F) = (μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1)) * H(F)
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    ∂ΨFF(F) = (2 * μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 2)) * (F ⊗ F) + (μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1)) * 𝕀3D
    ∂ΨHH(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 2)) * (H(F) ⊗ H(F)) + (μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1)) * 𝕀3D
    ∂Ψuu(F) = ∂ΨFF(F) + (F × (∂ΨHH(F) × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end

struct NonlinearMooneyRivlin2D <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  α1::Float64
  α2::Float64
  ρ::Float64
  function NonlinearMooneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, α1::Float64, α2::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, α1, α2, ρ)
  end

  function (obj::NonlinearMooneyRivlin2D)()
    λ, μ1, μ2, α1, α2 = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2
    J, H = det, cof
    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^α2 - (μ1 + 2.0 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * F
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F) = μ2 / (3.0^(α2 - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(α2 - 1) - (μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * 𝕀2D +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 2)) + μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1) + (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F)^2 - (μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearMooneyRivlin2D_CV <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  α1::Float64
  α2::Float64
  γ::Float64
  ρ::Float64
  function NonlinearMooneyRivlin2D_CV(; λ::Float64, μ1::Float64, μ2::Float64, α1::Float64, α2::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, α1, α2, γ, ρ)
  end

  function (obj::NonlinearMooneyRivlin2D_CV)()
    λ, μ1, μ2, α1, α2, γ = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2, obj.γ
    J, H = det, cof
    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^α2 - (μ1 + 2.0 * μ2) * log(J(F)) +
           λ * (J(F)^(γ) + J(F)^(-γ)) +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2 - 2λ

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * F
    ∂Ψ_∂J(F) = μ2 / (3.0^(α2 - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(α2 - 1) - (μ1 + 2.0 * μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * 𝕀2D +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 2)) + μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1) + (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F)^2 + (μ1 + 2.0 * μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end

struct NonlinearMooneyRivlin_CV <: IsoElastic
  λ::Float64
  μ1::Float64
  μ2::Float64
  α1::Float64
  α2::Float64
  γ::Float64
  ρ::Float64
  function NonlinearMooneyRivlin_CV(; λ::Float64, μ1::Float64, μ2::Float64, α1::Float64, α2::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ1, μ2, α1, α2, γ, ρ)
  end

  function (obj::NonlinearMooneyRivlin_CV)()
    λ, μ1, μ2, α1, α2, γ = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2, obj.γ
    J, H = det, cof
    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F))^α1 +
           μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^α2 -
           (μ1 + 2 * μ2) * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ)) +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 +
           -μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2 +
           -2λ

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α1 - 1)) * (trAA(F))^(α1 - 1))) * F
    ∂Ψ_∂H(F) = ((μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1))) * H(F)
    ∂Ψ_∂J(F) = -(μ1 + 2 * μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1))) * 𝕀3D +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 2))) * (F ⊗ F)
    ∂Ψ2_∂HH(F) = ((μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1))) * 𝕀3D +
                 2 * ((μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 2))) * (H(F) ⊗ H(F))
    ∂Ψ2_∂JJ(F) = (μ1 + 2 * μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (F × (∂Ψ2_∂HH(F) × F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearNeoHookean_CV <: IsoElastic
  λ::Float64
  μ::Float64
  α::Float64
  γ::Float64
  ρ::Float64
  function NonlinearNeoHookean_CV(; λ::Float64, μ::Float64, α::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ, α, γ, ρ)
  end

  function (obj::NonlinearNeoHookean_CV)()
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ
    J, H = det, cof
    Ψ(F) = μ / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F))^α - μ * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ)) +
           -μ / (2.0 * α * 3.0^(α - 1)) * 3.0^α - 2λ

    ∂Ψ_∂F(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * F
    ∂Ψ_∂J(F) = -μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * 𝕀3D +
                 2 * ((μ * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F))^(α - 2))) * (F ⊗ F)
    ∂Ψ2_∂JJ(F) = μ * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearIncompressibleMooneyRivlin2D_CV <: IsoElastic
  λ::Float64
  μ::Float64
  α::Float64
  γ::Float64
  ρ::Float64
  function NonlinearIncompressibleMooneyRivlin2D_CV(; λ::Float64, μ::Float64, α::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ, α, γ, ρ)
  end

  function (obj::NonlinearIncompressibleMooneyRivlin2D_CV)()
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ
    J, H = det, cof
    e(F) = (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    ∂e_∂F(F) = 2 * J(F)^(-2 / 3) * F
    ∂e_∂J(F) = -(2 / 3) * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂e2_∂F2(F) = 2 * J(F)^(-2 / 3) * 𝕀2D
    ∂e2_∂J2(F) = (10 / 9) * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂e2_∂FJ(F) = -(4 / 3) * J(F)^(-5 / 3) * F

    Ψ1(F) = μ / (2 * α) * (e(F))^α - μ / (2α) * 3^α
    Ψ2(F) = (λ) * (J(F)^(γ) + J(F)^(-γ)) - 2λ
    Ψ(F) = Ψ1(F) + Ψ2(F)

    ∂Ψ1_∂F(F) = (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂F(F))
    ∂Ψ1_∂J(F) = (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂J(F))
    ∂Ψ2_∂J(F) = λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψ_∂F(F) = ∂Ψ1_∂F(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂F2(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂F2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) ⊗ ∂e_∂F(F))
    ∂Ψ1_∂J2(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂J2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂J(F) * ∂e_∂J(F))
    ∂Ψ1_∂FJ(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂FJ(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) * ∂e_∂J(F))
    ∂Ψ2_∂J2(F) = λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψ_∂F2(F) = ∂Ψ1_∂F2(F)
    ∂Ψ_∂FJ(F) = ∂Ψ1_∂FJ(F)
    ∂Ψ_∂J2(F) = ∂Ψ1_∂J2(F) + ∂Ψ2_∂J2(F)

    ∂Ψuu(F) = ∂Ψ_∂F2(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


"""
Simplified eight-chain model by Arruda and Boyce.
the implementation uses the first five terms of the inverse Langevin function.

```math
\\Psi = C_1 \\sum_{i=1}^{3} \\alpha_i \\beta^{i-1} (I_1^i - 3^i)
```
"""
struct EightChain <: IsoElastic
  μ::Float64
  N::Float64
  EightChain(; μ::Float64, N::Float64) = new(μ, N)
end

function (obj::EightChain)()
  (; μ, N) = obj
  α = (1/2, 1/20, 11/1050, 19/7000, 519/673750)
  β = 1 / N
  C1 = μ / 2 / sum(i*αi*(3*β)^(i-1) for (i, αi) in enumerate(α))
  W(I) = C1 * sum(αi*β^(i-1)*(I^i - 3^i) for (i, αi) in enumerate(α))
  ∂W∂I(I) = C1 * sum(i*αi*β^(i-1)*I^(i-1) for (i, αi) in enumerate(α))
  ∂∂W∂II(I) = C1 * sum(i*(i-1)*α[i]*β^(i-1)*I^(i-2) for i in 2:length(α))
  Ψ(F) = W(I₁iso(F))
  ∂Ψ∂F(F) = ∂W∂I(I₁iso(F)) * ∂I₁iso∂F(F)
  ∂∂Ψ∂FF(F) = ∂∂W∂II(I₁iso(F)) * ∂I₁iso∂F(F) ⊗ ∂I₁iso∂F(F) + ∂W∂I(I₁iso(F)) * ∂∂I₁iso∂FF(F)
  return (Ψ, ∂Ψ∂F, ∂∂Ψ∂FF)
end


struct TransverseIsotropy3D <: AnisoElastic
  μ::Float64
  α1::Float64
  α2::Float64
  ρ::Float64
  function TransverseIsotropy3D(; μ::Float64, α1::Float64, α2::Float64, ρ::Float64=0.0)
    new(μ, α1, α2, ρ)
  end


  function (obj::TransverseIsotropy3D)()
    J, H = det, cof
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α1, α2 = obj.μ, obj.α1, obj.α2
    Ψ(F, N) = μ / (2.0 * α1) * (I4(F, N)^α1 - 1) + μ / (2.0 * α2) * (I5(F, N)^α2 - 1) - μ * logreg(J(F))

    ∂Ψ_∂F(F, N) = (μ * (I4(F, N)^(α1 - 1))) * ((F * N) ⊗ N)
    ∂Ψ_∂H(F, N) = (μ * (I5(F, N)^(α2 - 1))) * ((H(F) * N) ⊗ N)
    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F, N) = -μ * ∂log∂J(J(F))
    ∂Ψ2_∂J2(F, N) = -μ * ∂log2∂J2(J(F))

    ∂Ψu(F, N) = ∂Ψ_∂F(F, N) + ∂Ψ_∂H(F, N) × F + ∂Ψ_∂J(F, N) * H(F)

    ∂ΨFF(F, N) = μ * (I4(F, N)^(α1 - 1)) * (I3D⊗ᵢₖʲˡ(N ⊗ N)) + 2μ * (α1 - 1) * I4(F, N)^(α1 - 2) * (((F * N) ⊗ N) ⊗ ((F * N) ⊗ N))
    ∂ΨHH(F, N) = μ * (I5(F, N)^(α2 - 1)) * (I3D⊗ᵢₖʲˡ(N ⊗ N)) + 2μ * (α2 - 1) * I5(F, N)^(α2 - 2) * (((H(F) * N) ⊗ N) ⊗ ((H(F) * N) ⊗ N))
    ∂Ψuu(F, N) = ∂ΨFF(F, N) + (F × (∂ΨHH(F, N) × F)) + ∂Ψ2_∂J2(F, N) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F, N) + ∂Ψ_∂J(F, N) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct TransverseIsotropy2D <: AnisoElastic
  μ::Float64
  α1::Float64
  α2::Float64
  ρ::Float64
  function TransverseIsotropy2D(; μ::Float64, α1::Float64, α2::Float64, ρ::Float64=0.0)
    new(μ, α1, α2, ρ)
  end

  function (obj::TransverseIsotropy2D)()
    J, H = det, cof
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α1, α2 = obj.μ, obj.α1, obj.α2
    Ψ(F, N) = μ / (2.0 * α1) * (I4(F, N)^α1 - 1) + μ / (2.0 * α2) * (I5(F, N)^α2 - 1) - μ * logreg(J(F))

    ∂I4∂F(F, N) = 2 * ((F * N) ⊗ N)
    ∂I4∂F∂F(F, N) = 2 * (I2D⊗ᵢₖʲˡ(N ⊗ N))
    ∂I5∂F∂F(F, N) = 2 * (I2D ⊗ I2D) - 2 * (I2D ⊗ (N ⊗ N) + (N ⊗ N) ⊗ I2D) + 2 * ((N ⊗ N) ⊗ᵢₖʲˡ I2D)
    ∂I5∂F(F, N) = 2 * tr(F) * I2D - 2 * (N ⋅ (F * N)) * I2D - 2 * tr(F) * (N ⊗ N) + 2 * (N ⊗ (F' * N))

    ∂log∂J(J) = J >= LOGREG_THRESHOLD ? 1 / J : (2 / LOGREG_THRESHOLD - J / (LOGREG_THRESHOLD^2))
    ∂log2∂J2(J) = J >= LOGREG_THRESHOLD ? -1 / (J^2) : (-1 / (LOGREG_THRESHOLD^2))
    ∂Ψ_∂J(F, N) = -μ * ∂log∂J(J(F))
    ∂Ψ2_∂J2(F, N) = -μ * ∂log2∂J2(J(F))

    ∂Ψu(F, N) = (μ / 2 * (I4(F, N)^(α1 - 1))) * ∂I4∂F(F, N) +
                (μ / 2 * (I5(F, N)^(α2 - 1))) * ∂I5∂F(F, N) +
                ∂Ψ_∂J(F, N) * H(F)

    ∂Ψuu(F, N) = μ / 2 * (α1 - 1) * (I4(F, N)^(α1 - 2)) * (∂I4∂F(F, N)) ⊗ (∂I4∂F(F, N)) +
                 μ / 2 * (I4(F, N)^(α1 - 1)) * ∂I4∂F∂F(F, N) +
                 μ / 2 * (α2 - 1) * (I5(F, N)^(α2 - 2)) * (∂I5∂F(F, N)) ⊗ (∂I5∂F(F, N)) +
                 μ / 2 * (I5(F, N)^(α2 - 1)) * ∂I5∂F∂F(F, N) +
                 ∂Ψ2_∂J2(F, N) * (H(F) ⊗ H(F)) +
                 ∂Ψ_∂J(F, N) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct HGO_4Fibers <: AnisoElastic
  c1::Vector{Float64}
  c2::Vector{Float64}
  function HGO_4Fibers(; c1::Vector{Float64}, c2::Vector{Float64})
    @assert length(c1) == length(c2) == 4
    new(c1, c2)
  end

  function (obj::HGO_4Fibers)()
    c1, c2 = obj.c1, obj.c2

    Ψ(F, N1, N2, N3, N4) = begin
      M1 = N1 / norm(N1)
      M2 = N2 / norm(N2)
      M3 = N3 / norm(N3)
      M4 = N4 / norm(N4)
      c1[1] / (4 * c2[1]) * (exp(c2[1] * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) - 1.0) +
      c1[2] / (4 * c2[2]) * (exp(c2[2] * ((F * M2) ⋅ (F * M2) - 1.0)^2.0) - 1.0) +
      c1[3] / (4 * c2[3]) * (exp(c2[3] * ((F * M3) ⋅ (F * M3) - 1.0)^2.0) - 1.0) +
      c1[4] / (4 * c2[4]) * (exp(c2[4] * ((F * M4) ⋅ (F * M4) - 1.0)^2.0) - 1.0)
    end

    ∂Ψ∂F(F, N1, N2, N3, N4) = begin
      M1 = N1 / norm(N1)
      M2 = N2 / norm(N2)
      M3 = N3 / norm(N3)
      M4 = N4 / norm(N4)
      c1[1] * exp(c2[1] * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((F * M1) ⋅ (F * M1) - 1.0) * ((F * M1) ⊗ M1) +
      c1[2] * exp(c2[2] * ((F * M2) ⋅ (F * M2) - 1.0)^2.0) * ((F * M2) ⋅ (F * M2) - 1.0) * ((F * M2) ⊗ M2) +
      c1[3] * exp(c2[3] * ((F * M3) ⋅ (F * M3) - 1.0)^2.0) * ((F * M3) ⋅ (F * M3) - 1.0) * ((F * M3) ⊗ M3) +
      c1[4] * exp(c2[4] * ((F * M4) ⋅ (F * M4) - 1.0)^2.0) * ((F * M4) ⋅ (F * M4) - 1.0) * ((F * M4) ⊗ M4)
    end

    ∂Ψ2∂F∂F(F, N1, N2, N3, N4) = begin
      M1 = N1 / norm(N1)
      M2 = N2 / norm(N2)
      M3 = N3 / norm(N3)
      M4 = N4 / norm(N4)
      c1[1] * exp(c2[1] * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((4 * c2[1] * (((F * M1) ⋅ (F * M1) - 1.0)^2.0) + 2.0) * (((F * M1) ⊗ M1) ⊗ ((F * M1) ⊗ M1)) + ((F * M1) ⋅ (F * M1) - 1.0) * (I3D⊗ᵢₖʲˡ(M1 ⊗ M1))) +
      c1[2] * exp(c2[2] * ((F * M2) ⋅ (F * M2) - 1.0)^2.0) * ((4 * c2[2] * (((F * M2) ⋅ (F * M2) - 1.0)^2.0) + 2.0) * (((F * M2) ⊗ M2) ⊗ ((F * M2) ⊗ M2)) + ((F * M2) ⋅ (F * M2) - 1.0) * (I3D⊗ᵢₖʲˡ(M2 ⊗ M2))) +
      c1[3] * exp(c2[3] * ((F * M3) ⋅ (F * M3) - 1.0)^2.0) * ((4 * c2[3] * (((F * M3) ⋅ (F * M3) - 1.0)^2.0) + 2.0) * (((F * M3) ⊗ M3) ⊗ ((F * M3) ⊗ M3)) + ((F * M3) ⋅ (F * M3) - 1.0) * (I3D⊗ᵢₖʲˡ(M3 ⊗ M3))) +
      c1[4] * exp(c2[4] * ((F * M4) ⋅ (F * M4) - 1.0)^2.0) * ((4 * c2[4] * (((F * M4) ⋅ (F * M4) - 1.0)^2.0) + 2.0) * (((F * M4) ⊗ M4) ⊗ ((F * M4) ⊗ M4)) + ((F * M4) ⋅ (F * M4) - 1.0) * (I3D⊗ᵢₖʲˡ(M4 ⊗ M4)))
    end

    return (Ψ, ∂Ψ∂F, ∂Ψ2∂F∂F)
  end
end



struct HGO_1Fiber <: AnisoElastic
  c1::Float64
  c2::Float64
  function HGO_1Fiber(; c1::Float64, c2::Float64)
    new(c1, c2)
  end

  function (obj::HGO_1Fiber)()
    c1, c2 = obj.c1, obj.c2

    Ψ(F, N1) = begin
      M1 = N1 / norm(N1)
      c1 / (4 * c2) * (exp(c2 * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) - 1.0)
    end

    ∂Ψ∂F(F, N1) = begin
      M1 = N1 / norm(N1)
      c1 * exp(c2 * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((F * M1) ⋅ (F * M1) - 1.0) * ((F * M1) ⊗ M1)
    end

    ∂Ψ2∂F∂F(F, N1) = begin
      M1 = N1 / norm(N1)
      c1 * exp(c2 * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((4 * c2 * (((F * M1) ⋅ (F * M1) - 1.0)^2.0) + 2.0) * (((F * M1) ⊗ M1) ⊗ ((F * M1) ⊗ M1)) + ((F * M1) ⋅ (F * M1) - 1.0) * (I3D⊗ᵢₖʲˡ(M1 ⊗ M1)))
    end

    return (Ψ, ∂Ψ∂F, ∂Ψ2∂F∂F)
  end
end


struct IncompressibleNeoHookean3D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  δ::Float64
  function IncompressibleNeoHookean3D(; λ::Float64, μ::Float64, ρ::Float64=0.0, δ::Float64=0.1)
    new(λ, μ, ρ, δ)
  end

  function (obj::IncompressibleNeoHookean3D)()
    λ, μ, δ = obj.λ, obj.μ, obj.δ
    J_, H = det, cof
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F)) * J(F)^(-2 / 3) - 3μ / 2 * J1^(-2 / 3)
    Ψ2(F) = (λ / 2) * (J_(F) - 1)^2
    Ψ(F) = Ψ1(F) + Ψ2(F) - β * log(J_(F))

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F)) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * (J_(F) - 1)
    ∂Ψ3_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F) + ∂Ψ3_∂J(F)

    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F

    ∂Ψuu(F) = μ * 𝕀3D * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end


struct IncompressibleNeoHookean2D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  δ::Float64
  function IncompressibleNeoHookean2D(; λ::Float64, μ::Float64, ρ::Float64=0.0, δ::Float64=0.1)
    new(λ, μ, ρ, δ)
  end

  function (obj::IncompressibleNeoHookean2D)()
    λ, μ, δ = obj.λ, obj.μ, obj.δ
    J_, H = det, cof
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F) + 1.0) * J(F)^(-2 / 3) - 3μ / 2 * J(I2D)^(-2 / 3)
    Ψ2(F) = (λ / 2) * (J_(F) - 1)^2
    Ψ(F) = Ψ1(F) + Ψ2(F) - β * log(J_(F))

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * (J_(F) - 1)
    ∂Ψ3_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F) + ∂Ψ3_∂J(F)

    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F
    ∂Ψuu(F) = μ * 𝕀2D * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end

struct IncompressibleNeoHookean2D_CV <: IsoElastic
  λ::Float64
  μ::Float64
  γ::Float64
  ρ::Float64
  function IncompressibleNeoHookean2D_CV(; λ::Float64, μ::Float64, γ::Float64, ρ::Float64=0.0)
    new(λ, μ, γ, ρ)
  end

  function (obj::IncompressibleNeoHookean2D_CV)()
    λ, μ, γ = obj.λ, obj.μ, obj.γ
    J, H = det, cof
    Ψ1(F) = μ / 2 * (tr((F)' * F) + 1.0) * J(F)^(-2 / 3) - 3μ / 2
    Ψ2(F) = λ * (J(F)^(γ) + J(F)^(-γ)) - 2λ
    Ψ(F) = Ψ1(F) + Ψ2(F)

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))
    ∂Ψ_∂J2(F) = ∂Ψ1_∂J2(F) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * F
    ∂Ψuu(F) = μ * 𝕀2D * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct ARAP2D_regularized <: IsoElastic
  μ::Float64
  ρ::Float64
  δ::Float64
  function ARAP2D_regularized(; μ::Float64, ρ::Float64=0.0, δ::Float64=0.1)
    new(μ, ρ, δ)
  end

  function (obj::ARAP2D_regularized)()

    μ, δ = obj.μ, obj.δ
    J_, H = det, cof
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-1) - J1^(-2) * ∂J1)
    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F)) - β * log(J_(F)) - μ * J(I2D)^-1

    ∂Ψ1_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)
    ∂Ψ2_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -μ * J(F)^(-2) * F * ∂J(F)
    ∂Ψ_∂FF(F) = μ * J(F)^(-1) * 𝕀2D

    ∂Ψuu(F) = ∂Ψ_∂FF(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ∂H∂F(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct ARAP2D <: IsoElastic
  μ::Float64
  ρ::Float64
  function ARAP2D(; μ::Float64, ρ::Float64=0.0)
    new(μ, ρ)
  end

  function (obj::ARAP2D)()
    μ = obj.μ
    J, H = det, cof
    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F)) - μ
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)
    ∂Ψ_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)

    ∂2Ψ_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂2Ψ_∂FJ(F) = -μ * J(F)^(-2) * F
    ∂2Ψ_∂FF(F) = μ * J(F)^(-1) * 𝕀2D

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = ∂2Ψ_∂FF(F) + ∂2Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂2Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂2Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ∂H∂F(F)

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearARAP2D <: IsoElastic
  μ::Float64
  p::Float64
  ρ::Float64
  function NonlinearARAP2D(; μ::Float64, p::Float64, ρ::Float64=0.0)
    new(μ, p, ρ)
  end

  function (obj::NonlinearARAP2D)()
    μ = obj.μ
    p = obj.p

    J, H = det, cof
    f(F) = 0.5 * J(F)^(-1) * (tr((F)' * F)) - 1.0
    g(x) = x^p
    Ψ(F) = (μ/p) * g(f(F))

    ∂f_∂F(F) = F * J(F)^(-1)
    ∂f_∂J(F) = -1.0 / 2.0 * (tr((F)' * F)) * J(F)^(-2)
    ∂2f_∂J2(F) = J(F)^(-3) * (tr((F)' * F))
    ∂2f_∂FJ(F) = -J(F)^(-2) * F
    ∂2f_∂FF(F) = J(F)^(-1) * 𝕀2D

    ∂g_∂x(x) = p * x^(p-1)
    ∂2g_∂x2(x) = p * (p-1) * x^(p-2)

    ∂fu(F) = ∂f_∂F(F) + ∂f_∂J(F) * H(F)
    ∂fuu(F) = ∂2f_∂FF(F) + ∂2f_∂J2(F) * (H(F) ⊗ H(F)) + ∂2f_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂2f_∂FJ(F) + ∂f_∂J(F) * ∂H∂F(F)

    ∂Ψu(F) = (μ/p) * (∂g_∂x(f(F)) * ∂fu(F))
    ∂Ψuu(F) = (μ/p) * (∂g_∂x(f(F)) * ∂fuu(F) + ∂2g_∂x2(f(F)) * (∂fu(F) ⊗ ∂fu(F)))

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


"""
Neo-Hooke hyperelastic model

```math
\\Psi = \\frac{1}{2}\\mu (I_1 - 3)
```
"""
@kwdef struct IsochoricNeoHookean3D <: IsoElastic
  μ::Float64
end

function (obj::IsochoricNeoHookean3D)()
  W(I) = obj.μ / 2 * (I - 3)
  ∂W∂I(I) = obj.μ / 2
  Ψ(F) = W(I₁iso(F))
  ∂Ψ∂F(F) = ∂W∂I(I₁iso(F)) * ∂I₁iso∂F(F)
  ∂∂Ψ∂FF(F) = ∂W∂I(I₁iso(F)) * ∂∂I₁iso∂FF(F)
  return Ψ, ∂Ψ∂F, ∂∂Ψ∂FF
end

function SecondPiola(obj::IsochoricNeoHookean3D)
  μ = obj.μ
  Ψ(C) = μ / 2 * tr(C) * (det(C))^(-1 / 3) - 3*μ/2
  ∂Ψ∂C(C) = μ / 2 * I3D * (det(C))^(-1 / 3)
  ∂Ψ∂dC(C) = -μ / 6 * tr(C) * (det(C))^(-4 / 3)
  S(C) = 2 * (∂Ψ∂C(C) + ∂Ψ∂dC(C) * cof(C))
  ∂2Ψ∂CdC(C) = -μ / 6 * I3D * (det(C))^(-4 / 3)
  ∂2Ψ∂2dC(C) = 2 * μ / 9 * tr(C) * (det(C))^(-7 / 3)
  ∂S∂C(C) =
    let HC = cof(C)
      2 * (∂2Ψ∂2dC(C) * (HC ⊗ HC) + ∂2Ψ∂CdC(C) ⊗ HC + HC ⊗ ∂2Ψ∂CdC(C) + ∂Ψ∂dC(C) * ×ᵢ⁴(C))
    end
  return (Ψ, S, ∂S∂C)
end

