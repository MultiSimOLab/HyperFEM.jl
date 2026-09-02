using Gridap.TensorValues
using HyperFEM.PhysicalModels
using BenchmarkTools


function benchmark_viscous_model()
  elasto = NeoHookean3D(λ=1e6, μ=1e3)
  visco = ViscousIncompressible(IsochoricNeoHookean3D(μ=1e2), τ=10.)
  visco = ViscousIncompressible(IsochoricNeoHookean3D(μ=1e3), τ=1.)
  visco = ViscousIncompressible(IsochoricNeoHookean3D(μ=1e4), τ=.1)
  visco = ViscousIncompressible(IsochoricNeoHookean3D(μ=1e5), τ=.01)
  model = GeneralizedMaxwell(elasto, visco)
  update_time_step!(model, 1e-2)
  Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
  F = TensorValue(1.:9...) * 1e-3 + I3
  Fn = TensorValue(1.:9...) * 5e-4 + I3
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  J = det(F)
  Uvn *= J^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  SUITE["Constitutive models"]["Visco-elastic Ψ"] = @benchmarkable $Ψ($F, $Fn, $Avn)
  SUITE["Constitutive models"]["Visco-elastic ∂Ψ∂F"] = @benchmarkable $∂Ψ∂F($F, $Fn, $Avn)
  SUITE["Constitutive models"]["Visco-elastic ∂∂Ψ∂FF"] = @benchmarkable $∂∂Ψ∂FF($F, $Fn, $Avn)
end

function benchmark_viscous_polyconvex_model()
  elasto = NeoHookean3D(λ=1e6, μ=1e3)
  visco = ViscousPolyconvex(μ=1e2, τ=10.)
  visco = ViscousPolyconvex(μ=1e3, τ=1.)
  visco = ViscousPolyconvex(μ=1e4, τ=.1)
  visco = ViscousPolyconvex(μ=1e5, τ=.01)
  model = GeneralizedMaxwell(elasto, visco)
  update_time_step!(model, 1e-1)
  Ψ, ∂Ψ∂F, ∂∂Ψ∂FF = model()
  F = TensorValue(1.:9...) * 1e-2 + I3
  Fn = TensorValue(1.:9...) * 5e-3 + I3
  Cvn = 0.25 * (F+Fn) · (F+Fn)'
  F /= det(F)
  Fn /= det(Fn)
  Cvn /= det(Cvn)
  SUITE["Constitutive models"]["Visco-polyconvex Ψ"] = @benchmarkable $Ψ($F, $Fn, $Cvn)
  SUITE["Constitutive models"]["Visco-polyconvex ∂Ψ∂F"] = @benchmarkable $∂Ψ∂F($F, $Fn, $Cvn)
  SUITE["Constitutive models"]["Visco-polyconvex ∂∂Ψ∂FF"] = @benchmarkable $∂∂Ψ∂FF($F, $Fn, $Cvn)
end

benchmark_viscous_model()
benchmark_viscous_polyconvex_model()
