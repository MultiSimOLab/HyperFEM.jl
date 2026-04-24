using ForwardDiff
using HyperFEM
using Test

@testset "EntropicElasticityLaw" begin
  law = EntropicElasticityLaw(θr=273.15, γ=0.55)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test isapprox(df(θ), ForwardDiff.derivative(f, θ), rtol=1e-3)
    @test isapprox(ddf(θ), ForwardDiff.derivative(df, θ), rtol=1e-10)
  end
end

@testset "NonlinearMeltingLaw" begin
  law = NonlinearMeltingLaw(θr=273.15, θM=400.0, γ=0.55)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test isapprox(df(θ), ForwardDiff.derivative(f, θ), rtol=1e-10)
    @test isapprox(ddf(θ), ForwardDiff.derivative(df, θ), rtol=1e-10)
  end
end

@testset "NonlinearSofteningLaw" begin
  law = NonlinearSofteningLaw(θr=273.15, θt=300.0, γ=2.0, δ=0.5)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test isapprox(df(θ), ForwardDiff.derivative(f, θ), rtol=1e-10)
    @test isapprox(ddf(θ), ForwardDiff.derivative(df, θ), rtol=1e-10)
  end
end

@testset "TrigonometricLaw" begin
  law = TrigonometricLaw(273.15, 400.0)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test isapprox(df(θ), ForwardDiff.derivative(f, θ), rtol=1e-10)
    @test isapprox(ddf(θ), ForwardDiff.derivative(df, θ), rtol=1e-10)
  end
end
