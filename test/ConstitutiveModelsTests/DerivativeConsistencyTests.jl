# Consistency of the constitutive models with their own energy: every derivative returned by a model is
# compared with the finite-difference derivative (central differences with Richardson extrapolation) of the
# quantity of lower order returned by the same model, at several states: small and large deformation,
# compression (J ≈ 0.2) and strong compression (J < 0.01, the regularised branch of `logreg`).
using Gridap.TensorValues
using LinearAlgebra
using Test
using HyperFEM
using HyperFEM.TensorAlgebra

const PM = HyperFEM.PhysicalModels

dc_vals(y::Real) = [Float64(y)]
dc_vals(y) = collect(Float64, Tuple(y))
dc_ncomp(x::Real) = 1
dc_ncomp(x) = length(Tuple(x))
dc_perturb(x::Real, i, h) = x + h
dc_perturb(x, i, h) = (t = Tuple(x); typeof(x)(Base.setindex(t, t[i] + h, i)))

"Finite-difference Jacobian of `f(args...)` with respect to `args[k]` (rows: components of `f`, column-major)."
function dc_fdjac(f, args, k; h=1e-4)
  x = args[k]
  cols = map(1:dc_ncomp(x)) do i
    d(δ) = (dc_vals(f(Base.setindex(args, dc_perturb(x, i, δ), k)...)) .-
            dc_vals(f(Base.setindex(args, dc_perturb(x, i, -δ), k)...))) ./ (2δ)
    (4 .* d(h / 2) .- d(h)) ./ 3
  end
  reduce(hcat, cols)
end

"Relative error between `high(args...)` and the derivative of `low` with respect to `args[k]`."
function dc_error(high, low, args, k)
  D = dc_vals(high(args...))
  M = vec(dc_fdjac(low, args, k))
  norm(D .- M) / max(norm(M), 1e-300)
end

const DC_RTOL = 1e-6

const DC_F3 = (
  small=I3D + 1e-3 * TensorValue(1.0:9.0...),
  large=TensorValue(1.3, 0.1, 0.0, 0.2, 0.9, -0.1, 0.0, 0.15, 1.1),
  compression=TensorValue(0.5, 0.02, 0.0, 0.01, 0.6, 0.0, 0.0, 0.03, 0.67),
  regularised=TensorValue(0.2, 0.01, 0.0, 0.0, 0.2, 0.01, 0.0, 0.0, 0.2))     # J ≈ 0.008
const DC_F2 = (
  small=I2D + 1e-3 * TensorValue(1.0, 2.0, 3.0, 4.0),
  large=TensorValue(1.3, 0.1, 0.2, 0.9),
  compression=TensorValue(0.4, 0.02, 0.01, 0.5),
  regularised=TensorValue(0.08, 0.0, 0.01, 0.08))                               # J ≈ 0.006
const DC_N3 = VectorValue(0.6, 0.0, 0.8)
const DC_N2 = VectorValue(0.6, 0.8)
const DC_E3 = VectorValue(0.3, -0.2, 0.5)

const DC_MECH2 = (
  PM.LinearElasticity2D(λ=3.0, μ=1.0),
  PM.Gent2D(λ=3.0, μ=1.0, Jm=50.0, γ=1.0),
  PM.MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0),
  PM.NonlinearMooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0, α1=1.5, α2=1.3),
  PM.NonlinearMooneyRivlin2D_CV(λ=3.0, μ1=1.0, μ2=2.0, α1=1.5, α2=1.3, γ=2.0),
  PM.IncompressibleNeoHookean2D(λ=10.0, μ=1.0),
  PM.IncompressibleNeoHookean2D_CV(λ=10.0, μ=1.0, γ=2.0),
  PM.NonlinearIncompressibleMooneyRivlin2D_CV(λ=3.0, μ=1.0, α=1.5, γ=2.0),
  PM.ARAP2D(μ=1.0),
  PM.ARAP2D_regularized(μ=1.0),
  PM.NonlinearARAP2D(μ=1.0, p=3.0),
  PM.LogBarrier(κ=2.0, p=2))
const DC_MECH3 = (
  PM.LinearElasticity3D(λ=3.0, μ=1.0),
  PM.NeoHookean3D(λ=3.0, μ=1.0),
  PM.IsochoricNeoHookean3D(μ=1.0),
  PM.IncompressibleNeoHookean3D(λ=10.0, μ=1.0),
  PM.Yeoh3D(λ=3.0, C10=1.0, C20=0.5, C30=0.2),
  PM.MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0),
  PM.NonlinearMooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0, α1=1.5, α2=1.3),
  PM.NonlinearMooneyRivlin_CV(λ=3.0, μ1=1.0, μ2=2.0, α1=1.5, α2=1.3, γ=2.0),
  PM.NonlinearNeoHookean_CV(λ=3.0, μ=1.0, α=1.5, γ=2.0),
  PM.EightChain(μ=1.0, N=5.0),
  PM.VolumetricEnergy(λ=3.0),
  PM.CoerciveVolumetric(κ=3.0),
  PM.LogBarrier(κ=2.0, p=2))

@testset "Derivative consistency: isotropic mechanical models" begin
  for (models, Fs, I) in ((DC_MECH2, DC_F2, I2D), (DC_MECH3, DC_F3, I3D)), m in models
    Ψ, P, C = m()
    @testset "$(nameof(typeof(m))) $(length(Tuple(I)) == 4 ? "2D" : "3D"), $s" for (s, F) in pairs(Fs)
      @test dc_error(P, Ψ, (F,), 1) < DC_RTOL
      @test dc_error(C, P, (F,), 1) < DC_RTOL
    end
    @testset "$(nameof(typeof(m))) at rest" begin
      @test abs(Ψ(I)) < 1e-12
      @test norm(P(I)) < 1e-12
    end
  end
end

@testset "Yeoh3D: regularised energy" begin
  # Yeoh3D: the regularised energy is finite for inverted elements (it was a DomainError)
  Ψ, P, C = PM.Yeoh3D(λ=3.0, C10=1.0, C20=0.5, C30=0.2)()
  Finv = TensorValue(-0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  @test isfinite(Ψ(Finv))
  @test dc_error(P, Ψ, (Finv,), 1) < DC_RTOL
end

@testset "IncompressibleNeoHookean3D with a large regularisation δ" begin
  Ψ, P, C = PM.IncompressibleNeoHookean3D(λ=10.0, μ=1.0, δ=0.7)()
  for F in DC_F3
    @test dc_error(P, Ψ, (F,), 1) < DC_RTOL
    @test dc_error(C, P, (F,), 1) < DC_RTOL
  end
  @test abs(Ψ(I3D)) < 1e-12
  @test norm(P(I3D)) < 1e-12
end

@testset "Derivative consistency: anisotropic mechanical models" begin
  for (m, Fs, N) in ((PM.TransverseIsotropy3D(μ=1.0, α1=1.5, α2=1.3), DC_F3, DC_N3),
                     (PM.TransverseIsotropy2D(μ=1.0, α1=1.5, α2=1.3), DC_F2, DC_N2),
                     (PM.HGO_1Fiber(c1=1.0, c2=0.5), DC_F3, DC_N3),
                     (PM.NeoHookean3D(λ=3.0, μ=1.0) + PM.TransverseIsotropy3D(μ=1.0, α1=1.5, α2=1.3), DC_F3, DC_N3))
    Ψ, P, C = m()
    @testset "$(nameof(typeof(m))), $s" for (s, F) in pairs(Fs)
      @test dc_error(P, Ψ, (F, N), 1) < DC_RTOL
      @test dc_error(C, P, (F, N), 1) < DC_RTOL
    end
    I = length(Tuple(N)) == 3 ? I3D : I2D
    @test abs(Ψ(I, N)) < 1e-12
    @test norm(P(I, N)) < 1e-12
  end
  m = PM.HGO_4Fibers(c1=[1.0, 0.8, 0.6, 0.4], c2=[0.5, 0.4, 0.3, 0.2])
  Ns = (DC_N3, VectorValue(0.0, 1.0, 0.0), VectorValue(1.0, 1.0, 0.0), VectorValue(0.0, 1.0, 1.0))
  Ψ, P, C = m()
  for F in DC_F3
    @test dc_error(P, Ψ, (F, Ns...), 1) < DC_RTOL
    @test dc_error(C, P, (F, Ns...), 1) < DC_RTOL
  end
end

@testset "Hessian∇JRegularization in 2D and 3D" begin
  # with a floor δ below every eigenvalue the projection does not act: the tangent is the derivative of the stress
  for (mech, F) in ((PM.NeoHookean3D(λ=3.0, μ=1.0), DC_F3.large), (PM.ARAP2D(μ=1.0), DC_F2.large))
    Ψ, P, C = PM.Hessian∇JRegularization(mech; κ=2.0, δ=-1e6)()
    Jh = det(F) - 0.5   # the term κ (J - Jh) ∂H/∂F has weight in the tangent
    @test dc_error(P, Ψ, (F, Jh), 1) < DC_RTOL
    @test dc_error(C, P, (F, Jh), 1) < DC_RTOL
  end
end

@testset "Derivative consistency: thermo-mechanical models without a dedicated test" begin
  law = PM.EntropicElasticityLaw(θr=1.0, γ=0.7)
  thermo = PM.ThermalVolumetric(cv0=2.0, θr=1.0, α=0.1, κr=3.0, γ=0.5)
  θ = 1.3
  for m in (PM.ThermalDeviatoric(PM.MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0), law),
            PM.ThermoMech_Bonet(thermo, PM.NeoHookean3D(λ=3.0, μ=1.0), law))
    Ψ, Ψu, Ψθ, Ψuu, Ψθθ, Ψuθ = m()
    @testset "$(nameof(typeof(m))), $s" for (s, F) in pairs(DC_F3)
      s == :regularised && continue
      a = (F, θ)
      @test dc_error(Ψu, Ψ, a, 1) < DC_RTOL
      @test dc_error(Ψθ, Ψ, a, 2) < DC_RTOL
      @test dc_error(Ψuu, Ψu, a, 1) < DC_RTOL
      @test dc_error(Ψθθ, Ψθ, a, 2) < DC_RTOL
      @test dc_error(Ψuθ, Ψu, a, 2) < DC_RTOL
    end
  end
  # viscous branch: the stress is the algorithmic one (internal variable fixed), so the checks are those of
  # the temperature and of the tangent against the stress
  visco = PM.GeneralizedMaxwell(PM.NeoHookean3D(λ=3.0, μ=1.0), PM.ViscousPolyconvex(μ=1.0, τ=1.0))
  update_time_step!(visco, 0.1)
  m = PM.ThermoMech_Bonet(thermo, visco, law, PM.NonlinearMeltingLaw(θr=1.0, θM=3.0, γ=0.7))
  Ψ, Ψu, Ψθ, Ψuu, Ψθθ, Ψuθ = m()
  Fn = I3D + 1e-2 * TensorValue(1.0:9.0...)
  for F in (DC_F3.small, DC_F3.large)
    a = (F, θ, Fn, I3D)
    @test dc_error(Ψθ, Ψ, a, 2) < DC_RTOL
    @test dc_error(Ψθθ, Ψθ, a, 2) < DC_RTOL
    @test dc_error(Ψuu, Ψu, a, 1) < DC_RTOL
    @test dc_error(Ψuθ, Ψu, a, 2) < DC_RTOL
  end
  # thermo-electric model: (Ψ, ∂F, ∂E, ∂θ, ∂FF, ∂EE, ∂θθ, ∂EF, ∂Fθ, ∂Eθ)
  m = PM.ThermoElectroModel(PM.IdealDielectric(ε=1.0), law)
  Ψ, ΨF, ΨE, Ψθ, ΨFF, ΨEE, Ψθθ, ΨEF, ΨFθ, ΨEθ = m()
  for F in (DC_F3.small, DC_F3.large, DC_F3.compression)
    a = (F, DC_E3, θ)
    @test dc_error(ΨF, Ψ, a, 1) < DC_RTOL
    @test dc_error(ΨE, Ψ, a, 2) < DC_RTOL
    @test dc_error(Ψθ, Ψ, a, 3) < DC_RTOL
    @test dc_error(ΨFF, ΨF, a, 1) < DC_RTOL
    @test dc_error(ΨEE, ΨE, a, 2) < DC_RTOL
    @test dc_error(Ψθθ, Ψθ, a, 3) < DC_RTOL
    @test dc_error(ΨEF, ΨE, a, 1) < DC_RTOL
    @test dc_error(ΨFθ, ΨF, a, 3) < DC_RTOL
    @test dc_error(ΨEθ, ΨE, a, 3) < DC_RTOL
  end
end

@testset "Every constitutive model accepts the load factor Λ" begin
  law = PM.EntropicElasticityLaw(θr=1.0, γ=0.7)
  thermo = PM.ThermalVolumetric(cv0=2.0, θr=1.0, α=0.1, κr=3.0, γ=0.5)
  diel = PM.IdealDielectric(ε=1.0)
  nh = PM.NeoHookean3D(λ=3.0, μ=1.0)
  visco = PM.GeneralizedMaxwell(nh, PM.ViscousIncompressible(PM.IsochoricNeoHookean3D(μ=1.0); τ=1.0), PM.ViscousPolyconvex(μ=1.0, τ=1.0))
  models = (nh, PM.EightChain(μ=1.0, N=5.0), diel, visco,
            PM.EnergyInterpolationScheme(nh, PM.MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)),
            PM.HessianRegularization(nh), PM.Hessian∇JRegularization(nh),
            diel + nh, diel + visco, thermo,
            PM.ThermalDeviatoric(nh, law), PM.ThermoMech_Bonet(thermo, nh, law), PM.ThermoMech_Bonet(thermo, visco, law, law),
            PM.ThermoElectroModel(diel, law),
            PM.ThermoElectroMech_Bonet(thermo, diel, nh; mech=law, elec=law),
            PM.ThermoElectroMech_Bonet(thermo, diel, visco; el=law, vis=law, elec=law))
  for m in models
    @test m(1.0) isa Tuple
    @test m() isa Tuple
  end
end

# A test model whose energy depends on the load factor: Λ × NeoHookean3D.
struct DCLoadScaled <: PM.IsoElastic
  m::PM.NeoHookean3D
end
function (obj::DCLoadScaled)(Λ::Float64=1.0)
  Ψ, P, C = obj.m(Λ)
  return (F -> Λ * Ψ(F), F -> Λ * P(F), F -> Λ * C(F))
end

@testset "The composed models forward Λ to their components" begin
  law = PM.EntropicElasticityLaw(θr=1.0, γ=0.7)
  thermo = PM.ThermalVolumetric(cv0=2.0, θr=1.0, α=0.1, κr=3.0, γ=0.5)
  diel = PM.IdealDielectric(ε=1.0)
  m = DCLoadScaled(PM.NeoHookean3D(λ=3.0, μ=1.0))
  F = DC_F3.large
  Ψm(Λ) = m(Λ)[1](F)
  # each case: (wrapped model, arguments after F); the energy of the wrapper differs from Λ = 1 to Λ = 2 by
  # (a multiple of) that of the component
  cases = ((PM.HessianRegularization(m), ()),
           (PM.Hessian∇JRegularization(m), (det(F) - 0.5,)),
           (m + PM.VolumetricEnergy(λ=1.0), ()),
           (PM.GeneralizedMaxwell(m), (I3D,)),
           (diel + m, (DC_E3,)),
           (PM.ThermalDeviatoric(m, law), (1.0,)),
           (PM.ThermoMech_Bonet(thermo, m, law), (1.0,)),
           (PM.ThermoElectroMech_Bonet(thermo, diel, m; mech=law, elec=law), (DC_E3, 1.0)))
  for (w, rest) in cases
    Ψ1 = w(1.0)[1](F, rest...)
    Ψ2 = w(2.0)[1](F, rest...)
    @test Ψ2 - Ψ1 ≈ Ψm(2.0) - Ψm(1.0)
  end
  e = PM.EnergyInterpolationScheme(m, PM.NeoHookean3D(λ=3.0, μ=1.0); p=3.0)
  @test e(2.0)[1](1.0, F) - e(1.0)[1](1.0, F) ≈ Ψm(2.0) - Ψm(1.0)
end
