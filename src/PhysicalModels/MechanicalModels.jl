

# ============================================
# Coercive volumetric Mechanical models
# ============================================

struct VolumetricEnergy <: Volumetric
  λ::Float64
  function VolumetricEnergy(; λ::Float64)
    new(λ)
  end
end

function tangent(obj::Volumetric)
  _, _, ∂∂Ψ = obj()
  ∂∂Ψ(I3)[1]
end

function (obj::VolumetricEnergy)(Λ::Float64=1.0)
  λ = obj.λ
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
  Ψ(F) = (λ / 2.0) * (J(F) - 1)^2
  ∂Ψ_∂J(F) = λ * (J(F) - 1)
  ∂Ψ2_∂J2(F) = λ
  ∂Ψu(F) = ∂Ψ_∂J(F) * H(F)
  ∂Ψuu(F) = ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂J(F) * F)
  return (Ψ, ∂Ψu, ∂Ψuu)
end


# ============================================
# Regularization of Mechanical models
# ============================================

struct HessianRegularization <: Mechano
  mechano::Mechano
  δ::Float64

  function HessianRegularization(mechano::Mechano; δ::Float64=1.0e-6)
    new(mechano, δ)
  end

  function HessianRegularization(; mechano::Mechano, δ::Float64=1.0e-6)
    new(mechano, δ)
  end

  function (obj::HessianRegularization)(Λ::Float64=1.0)
    Ψs, ∂Ψs, ∂2Ψs = obj.mechano()
    δ = obj.δ

    ∂2Ψ(F) = begin
      vecval = eigen(Symmetric(get_array(∂2Ψs(F))))
      vec = real(vecval.vectors)
      val = real(vecval.values)
      TensorValue(vec * diagm(max.(δ, val)) * vec')
    end
    return (Ψs, ∂Ψs, ∂2Ψ)
  end
end


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

  function (obj::Hessian∇JRegularization)(Λ::Float64=1.0)
    Ψs, ∂Ψs, ∂2Ψs = obj.mechano()
    δ, κ = obj.δ, obj.κ

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F, Jh) = Ψs(F) + 0.5 * κ * (J(F) - Jh)^2
    ∂Ψ(F, Jh) = ∂Ψs(F) + κ * (J(F) - Jh) * H(F)
    ∂2Ψ_(F, Jh) = ∂2Ψs(F) + κ * (H(F) ⊗ H(F)) + κ * (J(F) - Jh) * _∂H∂F_2D()

    ∂2Ψ(F, Jh) = begin
      vecval = eigen(Symmetric(get_array(∂2Ψ_(F, Jh))))
      vec = real(vecval.vectors)
      val = real(vecval.values)
      TensorValue(vec * diagm(max.(δ, val)) * vec')
    end
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
  function (obj::ComposedIsoElastic)(Λ::Float64=1.0)
    Ψ1, ∂Ψu1, ∂Ψuu1 = obj.Model1(Λ)
    Ψ2, ∂Ψu2, ∂Ψuu2 = obj.Model2(Λ)
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
  function (obj::ComposedAnisoElastic)(Λ::Float64=1.0)
    DΨ1 = obj.Model1(Λ)
    DΨ2 = obj.Model2(Λ)
    Ψ, ∂Ψ, ∂∂Ψ = map((ψ1, ψ2) -> (x, N...) -> ψ1(x) + ψ2(x, N...), DΨ1, DΨ2)
    return (Ψ, ∂Ψ, ∂∂Ψ)
  end
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
  DΨ     = map(a -> a(args...), obj.Models)
  Ψα     = getindex.(DΨ, 1)
  ∂Ψα∂F  = getindex.(DΨ, 2)
  ∂Ψα∂FF = getindex.(DΨ, 3)
  Ψ(F, N) = mapreduce((Ψi, Ni) -> Ψi(F, Ni), +, Ψα, N)
  ∂Ψ∂F(F, N) = mapreduce((∂Ψi∂F, Ni) -> ∂Ψi∂F(F, Ni), +, ∂Ψα∂F, N)
  ∂Ψ∂FF(F, N) = mapreduce((∂Ψi∂FF, Ni) -> ∂Ψi∂FF(F, Ni), +, ∂Ψα∂FF, N)
  (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end


# ===================
# Mechanical models
# ===================

struct Yeoh3D <: IsoElastic
  λ::Float64
  C10::Float64
  C20::Float64
  C30::Float64
  function Yeoh3D(; λ::Float64, C10::Float64, C20::Float64, C30::Float64)
    new(λ, C10, C20, C30)
  end

  function (obj::Yeoh3D)(Λ::Float64=1.0; Threshold=0.01)

    λ, C10, C20, C30 = obj.λ, obj.C10, obj.C20, obj.C30

    J(F) = det(F)
    H(F) = det(F) * inv(F)'

    # Free energy
    I1(F) = tr((F)' * F)
    ψ(F) = C10 * (I1(F) - 3) + C20 * (I1(F) - 3)^2 + C30 * (I1(F) - 3)^3 - 2 * C10 * log(J(F)) + 0.5 * λ * (J(F) - 1)^2

    # First Piola-Kirchhoff
    ∂ψ_∂I1(F) = C10 + 2 * C20 * (I1(F) - 3) + 3 * C30 * (I1(F) - 3)^2
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂ψ_∂J(F) = -2 * C10 * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂ψu(F) = 2 * ∂ψ_∂I1(F) * F + ∂ψ_∂J(F) * H(F)

    # Elasticity
    ∂2ψ_∂I1I1(F) = 2 * C20 + 6 * C30 * (I1(F) - 3)
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂2ψ_∂JJ(F) = -2 * C10 * ∂log2∂J2(J(F)) + λ
    ∂ψuu(F) = 4 * ∂2ψ_∂I1I1(F) * (F ⊗ F) + 2 * ∂ψ_∂I1(F) * I9 + ∂2ψ_∂JJ(F) * (H(F) ⊗ H(F)) + ∂ψ_∂J(F) * (I9 × F)


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

  function (obj::LinearElasticity2D)(Λ::Float64=1.0)
    λ, μ = obj.λ, obj.μ
    ε(F) = 0.5(F + F') - I2
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ2D + δᵢₗδⱼₖ2D) + λ * δᵢⱼδₖₗ2D
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I2)
    Ψ(F) = μ * sum(ε(F) .* ε(F)) + 0.5 * λ * tr(ε(F))^2
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


mutable struct LinearElasticity3D <: IsoElastic
  λ::Float64
  μ::Float64
  ρ::Float64
  function LinearElasticity3D(; λ::Float64, μ::Float64, ρ::Float64=0.0)
    new(λ, μ, ρ)
  end

  function (obj::LinearElasticity3D)(Λ::Float64=1.0)
    λ, μ = obj.λ, obj.μ
    ε(F) = 0.5(F + F') - I3
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ3D + δᵢₗδⱼₖ3D) + λ * δᵢⱼδₖₗ3D
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I3)
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

  function (obj::NeoHookean3D)(Λ::Float64=1.0; Threshold=0.01)
    λ, μ = obj.λ, obj.μ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = μ / 2 * tr((F)' * F) - μ * logreg(J(F)) + (λ / 2) * (J(F) - 1)^2 - 3.0 * (μ / 2.0)

    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))

    ∂Ψ_∂J(F) = -μ * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψu(F) = μ * F + ∂Ψ_∂J(F) * H(F)
    ∂Ψ2_∂J2(F) = -μ * ∂log2∂J2(J(F)) + λ
    ∂Ψuu(F) = μ * I9 + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
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

  function (obj::Gent2D)(Λ::Float64=1.0)
    λ, μ, Jm, γ, ρ = obj.λ, obj.μ, obj.Jm, obj.γ, obj.ρ

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    I1(F) = tr(F' * F) + 1.0

    Ψiso(F) = -μ * Jm / 2.0 * log(1.0 - (I1(F) - 3.0) / Jm)
    Ψvol(F) = -μ * log(J(F)) + λ * (J(F)^γ + J(F)^(-γ)) - 2.0 * λ
    Ψ(F) = Ψiso(F) + Ψvol(F)

    ∂Ψ_∂F(F) = F * μ / (1.0 - (I1(F) - 3.0) / Jm)
    ∂Ψ_∂J(F) = -μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1.0) - J(F)^(-γ - 1.0))
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = (μ / (1.0 - (I1(F) - 3.0) / Jm)) * I4 + 2.0 * (μ / (Jm * (1.0 - (I1(F) - 3.0) / Jm)^2)) * (F ⊗ F)
    ∂Ψ2_∂JJ(F) = μ * (1.0 / (J(F)^2)) +
                 λ * γ * ((γ - 1.0) * J(F)^(γ - 2.0) + (γ + 1.0) * J(F)^(-γ - 2.0))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) +
              ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) +
              ∂Ψ_∂J(F) * _∂H∂F_2D()

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

  function (obj::MooneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    J(F) = det(F)
    H(F) = det(F) * inv(F)'

    Ψ(F) = μ1 / 2 * tr((F)' * F) + μ2 / 2.0 * tr((H(F))' * H(F)) - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 - (3.0 / 2.0) * (μ1 + μ2)
    ∂Ψ_∂F(F) = μ1 * F
    ∂Ψ_∂H(F) = μ2 * H(F)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = μ1 * I9 + μ2 * (F × (I9 × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)

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

  function (obj::MooneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = (μ1 / 2 + μ2 / 2) * tr((F)' * F) + μ2 / 2.0 * J(F)^2 - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 - (μ1 + μ2) - μ2 / 2
    ∂Ψ_(F) = ForwardDiff.gradient(F -> Ψ(F), get_array(F))
    ∂2Ψ_(F) = ForwardDiff.jacobian(F -> ∂Ψ_(F), get_array(F))

    ∂Ψu(F) = TensorValue(∂Ψ_(F))
    ∂Ψuu(F) = TensorValue(∂2Ψ_(F))
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

  function (obj::NonlinearMooneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
    λ, μ1, μ2, α1, α2 = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2
    J(F) = det(F)
    H(F) = det(F) * inv(F)'

    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F))^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^α2 - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2

    ∂Ψ_∂F(F) = (μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1)) * F
    ∂Ψ_∂H(F) = (μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1)) * H(F)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    ∂ΨFF(F) = (2 * μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 2)) * (F ⊗ F) + (μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1)) * I9
    ∂ΨHH(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 2)) * (H(F) ⊗ H(F)) + (μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1)) * I9
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

  function (obj::NonlinearMooneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
    λ, μ1, μ2, α1, α2 = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^α2 - (μ1 + 2.0 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * F
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = μ2 / (3.0^(α2 - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(α2 - 1) - (μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * I4 +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 2)) + μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1) + (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F)^2 - (μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

  function (obj::NonlinearMooneyRivlin2D_CV)(Λ::Float64=1.0)
    λ, μ1, μ2, α1, α2, γ = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2, obj.γ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^α1 + μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^α2 - (μ1 + 2.0 * μ2) * log(J(F)) +
           λ * (J(F)^(γ) + J(F)^(-γ)) +
           -μ1 / (2.0 * α1 * 3.0^(α1 - 1)) * 3^α1 - μ2 / (2.0 * α2 * 3.0^(α2 - 1)) * 3^α2 - 2λ

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * F
    ∂Ψ_∂J(F) = μ2 / (3.0^(α2 - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(α2 - 1) - (μ1 + 2.0 * μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 1)) + μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1)) * I4 +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F) + 1.0)^(α1 - 2)) + μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 1) + (2 * μ2 * (α2 - 1) / (3.0^(α2 - 1)) * (tr((F)' * F) + J(F)^2)^(α2 - 2)) * J(F)^2 + (μ1 + 2.0 * μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

  function (obj::NonlinearMooneyRivlin_CV)(Λ::Float64=1.0)
    λ, μ1, μ2, α1, α2, γ = obj.λ, obj.μ1, obj.μ2, obj.α1, obj.α2, obj.γ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
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

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 1))) * I9 +
                 2 * ((μ1 * (α1 - 1) / (3.0^(α1 - 1)) * (tr((F)' * F))^(α1 - 2))) * (F ⊗ F)
    ∂Ψ2_∂HH(F) = ((μ2 / (3.0^(α2 - 1)) * (tr((H(F))' * H(F)))^(α2 - 1))) * I9 +
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

  function (obj::NonlinearNeoHookean_CV)(Λ::Float64=1.0)
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = μ / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F))^α - μ * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ)) +
           -μ / (2.0 * α * 3.0^(α - 1)) * 3.0^α - 2λ

    ∂Ψ_∂F(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * F
    ∂Ψ_∂J(F) = -μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * I9 +
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

  function (obj::NonlinearIncompressibleMooneyRivlin2D_CV)(Λ::Float64=1.0)
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    e(F) = (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    ∂e_∂F(F) = 2 * J(F)^(-2 / 3) * F
    ∂e_∂J(F) = -(2 / 3) * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂e2_∂F2(F) = 2 * J(F)^(-2 / 3) * I4
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

    ∂Ψuu(F) = ∂Ψ_∂F2(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


I1iso(F) = det(F)^(-2 / 3) * F ⊙ F
∂I1iso_∂F(F) = 2 * det(F)^(-2 / 3) * F
∂I1iso_∂J(F) = -(2 / 3) * det(F)^(-5 / 3) * F ⊙ F
∂I1iso_∂F∂F(F) = 2 * det(F)^(-2 / 3) * I9
∂I1iso_∂J∂J(F) = (10 / 9) * det(F)^(-8 / 3) * F ⊙ F
∂I1iso_∂F∂J(F) = -(4 / 3) * det(F)^(-5 / 3) * F

∂I1iso_∂Ftotal(F) = ∂I1iso_∂F(F) + ∂I1iso_∂J(F)*cof(F)
∂I1iso_∂F∂Ftotal(F) = ∂I1iso_∂F∂F(F) + ∂I1iso_∂F∂J(F) ⊗ cof(F) + cof(F) ⊗ ∂I1iso_∂F∂J(F) + ∂I1iso_∂J∂J(F)*cof(F) ⊗ cof(F) + ∂I1iso_∂J(F)*×ᵢ⁴(F)


struct EightChain <: IsoElastic
  μ::Float64
  N::Float64
  EightChain(; μ::Float64, N::Float64) = new(μ, N)
end

function (obj::EightChain)(::Float64=0.0)
  (; μ, N) = obj
  α = (1/2, 1/20, 11/1050, 19/7000, 519/673750)
  β = 1 / N
  C1 = μ / 2 / sum(i*αi*(3*β)^(i-1) for (i, αi) in enumerate(α))
  W(I) = C1 * sum(αi*β^(i-1)*(I^i - 3^i) for (i, αi) in enumerate(α))
  ∂W∂I(I) = C1 * sum(i*αi*β^(i-1)*I^(i-1) for (i, αi) in enumerate(α))
  ∂∂W∂II(I) = C1 * sum(i*(i-1)*α[i]*β^(i-1)*I^(i-2) for i in 2:length(α))
  Ψ(F) = W(I1iso(F))
  ∂Ψ∂F(F) = ∂W∂I(I1iso(F)) * ∂I1iso_∂Ftotal(F)
  ∂∂Ψ∂FF(F) = ∂∂W∂II(I1iso(F)) * ∂I1iso_∂Ftotal(F) ⊗ ∂I1iso_∂Ftotal(F) + ∂W∂I(I1iso(F)) * ∂I1iso_∂F∂Ftotal(F)
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


  function (obj::TransverseIsotropy3D)(Λ::Float64=1.0; Threshold=0.01)
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α1, α2 = obj.μ, obj.α1, obj.α2
    Ψ(F, N) = μ / (2.0 * α1) * (I4(F, N)^α1 - 1) + μ / (2.0 * α2) * (I5(F, N)^α2 - 1) - μ * logreg(J(F))

    ∂Ψ_∂F(F, N) = (μ * (I4(F, N)^(α1 - 1))) * ((F * N) ⊗ N)
    ∂Ψ_∂H(F, N) = (μ * (I5(F, N)^(α2 - 1))) * ((H(F) * N) ⊗ N)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F, N) = -μ * ∂log∂J(J(F))
    ∂Ψ2_∂J2(F, N) = -μ * ∂log2∂J2(J(F))

    ∂Ψu(F, N) = ∂Ψ_∂F(F, N) + ∂Ψ_∂H(F, N) × F + ∂Ψ_∂J(F, N) * H(F)

    ∂ΨFF(F, N) = μ * (I4(F, N)^(α1 - 1)) * (I3 ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (α1 - 1) * I4(F, N)^(α1 - 2) * (((F * N) ⊗ N) ⊗ ((F * N) ⊗ N))
    ∂ΨHH(F, N) = μ * (I5(F, N)^(α2 - 1)) * (I3 ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (α2 - 1) * I5(F, N)^(α2 - 2) * (((H(F) * N) ⊗ N) ⊗ ((H(F) * N) ⊗ N))
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

  function (obj::TransverseIsotropy2D)(Λ::Float64=1.0; Threshold=0.01)
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α1, α2 = obj.μ, obj.α1, obj.α2
    Ψ(F, N) = μ / (2.0 * α1) * (I4(F, N)^α1 - 1) + μ / (2.0 * α2) * (I5(F, N)^α2 - 1) - μ * logreg(J(F))

    ∂I4∂F(F, N) = 2 * ((F * N) ⊗ N)
    ∂I4∂F∂F(F, N) = 2 * (I2 ⊗₁₃²⁴ (N ⊗ N))
    ∂I5∂F∂F(F, N) = 2 * (I2 ⊗ I2) - 2 * (I2 ⊗ (N ⊗ N) + (N ⊗ N) ⊗ I2) + 2 * ((N ⊗ N) ⊗₁₃²⁴ I2)
    ∂I5∂F(F, N) = 2 * tr(F) * I2 - 2 * (N ⋅ (F * N)) * I2 - 2 * tr(F) * (N ⊗ N) + 2 * (N ⊗ (F' * N))

    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
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
                 ∂Ψ_∂J(F, N) * _∂H∂F_2D()
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

  function (obj::HGO_4Fibers)(Λ::Float64=1.0; Threshold=0.01)
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
      c1[1] * exp(c2[1] * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((4 * c2[1] * (((F * M1) ⋅ (F * M1) - 1.0)^2.0) + 2.0) * (((F * M1) ⊗ M1) ⊗ ((F * M1) ⊗ M1)) + ((F * M1) ⋅ (F * M1) - 1.0) * (I3 ⊗₁₃²⁴ (M1 ⊗ M1))) +
      c1[2] * exp(c2[2] * ((F * M2) ⋅ (F * M2) - 1.0)^2.0) * ((4 * c2[2] * (((F * M2) ⋅ (F * M2) - 1.0)^2.0) + 2.0) * (((F * M2) ⊗ M2) ⊗ ((F * M2) ⊗ M2)) + ((F * M2) ⋅ (F * M2) - 1.0) * (I3 ⊗₁₃²⁴ (M2 ⊗ M2))) +
      c1[3] * exp(c2[3] * ((F * M3) ⋅ (F * M3) - 1.0)^2.0) * ((4 * c2[3] * (((F * M3) ⋅ (F * M3) - 1.0)^2.0) + 2.0) * (((F * M3) ⊗ M3) ⊗ ((F * M3) ⊗ M3)) + ((F * M3) ⋅ (F * M3) - 1.0) * (I3 ⊗₁₃²⁴ (M3 ⊗ M3))) +
      c1[4] * exp(c2[4] * ((F * M4) ⋅ (F * M4) - 1.0)^2.0) * ((4 * c2[4] * (((F * M4) ⋅ (F * M4) - 1.0)^2.0) + 2.0) * (((F * M4) ⊗ M4) ⊗ ((F * M4) ⊗ M4)) + ((F * M4) ⋅ (F * M4) - 1.0) * (I3 ⊗₁₃²⁴ (M4 ⊗ M4)))
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

  function (obj::HGO_1Fiber)(Λ::Float64=1.0; Threshold=0.01)
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
      c1 * exp(c2 * ((F * M1) ⋅ (F * M1) - 1.0)^2.0) * ((4 * c2 * (((F * M1) ⋅ (F * M1) - 1.0)^2.0) + 2.0) * (((F * M1) ⊗ M1) ⊗ ((F * M1) ⊗ M1)) + ((F * M1) ⋅ (F * M1) - 1.0) * (I3 ⊗₁₃²⁴ (M1 ⊗ M1)))
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

  function (obj::IncompressibleNeoHookean3D)(Λ::Float64=1.0)
    λ, μ, δ = obj.λ, obj.μ, obj.δ
    J_(F) = det(F)
    H(F) = det(F) * inv(F)'
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F)) * J(F)^(-2 / 3)
    Ψ2(F) = (λ / 2) * (J_(F) - 1)^2
    Ψ(F) = Ψ1(F) + Ψ2(F) - β * log(J_(F))

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F)) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * (J_(F) - 1)
    ∂Ψ3_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F) + ∂Ψ3_∂J(F)

    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + (∂Ψ_∂J(F) * ∂J(F)) * H(F)

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F

    ∂Ψuu(F) = μ * I9 * J(F)^(-2 / 3) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
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

  function (obj::IncompressibleNeoHookean2D)(Λ::Float64=1.0)
    λ, μ, δ = obj.λ, obj.μ, obj.δ
    J_(F) = det(F)
    H(F) = det(F) * inv(F)'
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F) + 1.0) * J(F)^(-2 / 3) - 3μ / 2 * J(I2)^(-2 / 3)
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
    ∂Ψuu(F) = μ * I4 * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

  function (obj::IncompressibleNeoHookean2D_CV)(Λ::Float64=1.0)
    λ, μ, γ = obj.λ, obj.μ, obj.γ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
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
    ∂Ψuu(F) = μ * I4 * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

  function (obj::ARAP2D_regularized)(Λ::Float64=1.0)

    μ, δ = obj.μ, obj.δ
    J_(F) = det(F)
    H(F) = det(F) * inv(F)'
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-1) - J1^(-2) * ∂J1)
    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F)) - β * log(J_(F)) - μ * J(I2)^-1

    ∂Ψ1_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)
    ∂Ψ2_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -μ * J(F)^(-2) * F * ∂J(F)
    ∂Ψ_∂FF(F) = μ * J(F)^(-1) * I4

    ∂Ψuu(F) = ∂Ψ_∂FF(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct ARAP2D <: IsoElastic
  μ::Float64
  ρ::Float64
  function ARAP2D(; μ::Float64, ρ::Float64=0.0)
    new(μ, ρ)
  end

  function (obj::ARAP2D)(Λ::Float64=1.0)
    μ = obj.μ
    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F)) - μ
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)
    ∂Ψ_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)

    ∂2Ψ_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂2Ψ_∂FJ(F) = -μ * J(F)^(-2) * F
    ∂2Ψ_∂FF(F) = μ * J(F)^(-1) * I4

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = ∂2Ψ_∂FF(F) + ∂2Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂2Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂2Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


@kwdef struct IsochoricNeoHookean3D <: IsoElastic
  μ::Float64
end

function (obj::IsochoricNeoHookean3D)()
  W(I) = obj.μ / 2 * (I - 3)
  ∂W∂I(I) = obj.μ / 2
  Ψ(F) = W(I1iso(F))
  ∂Ψ∂F(F) = ∂W∂I(I1iso(F)) * ∂I1iso_∂Ftotal(F)
  ∂∂Ψ∂FF(F) = ∂W∂I(I1iso(F)) * ∂I1iso_∂F∂Ftotal(F)
  return Ψ, ∂Ψ∂F, ∂∂Ψ∂FF
end

function SecondPiola(obj::IsochoricNeoHookean3D)
  μ = obj.μ
  H(F) = cof(F)
  Ψ(C) = μ / 2 * tr(C) * (det(C))^(-1 / 3) -3*μ/2
  ∂Ψ∂C(C) = μ / 2 * I3 * (det(C))^(-1 / 3)
  ∂Ψ∂dC(C) = -μ / 6 * tr(C) * (det(C))^(-4 / 3)
  S(C) = 2 * (∂Ψ∂C(C) + ∂Ψ∂dC(C) * H(C))
  ∂2Ψ∂CdC(C) = -μ / 6 * I3 * (det(C))^(-4 / 3)
  ∂2Ψ∂2dC(C) = 2 * μ / 9 * tr(C) * (det(C))^(-7 / 3)
  ∂S∂C(C) = let HC = H(C)
    2 * (∂2Ψ∂2dC(C) * (HC ⊗ HC) + ∂2Ψ∂CdC(C) ⊗ HC + HC ⊗ ∂2Ψ∂CdC(C) + ∂Ψ∂dC(C) * ×ᵢ⁴(C))
  end
  return (Ψ, S, ∂S∂C)
end
