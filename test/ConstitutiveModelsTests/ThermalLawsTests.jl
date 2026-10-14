using ForwardDiff
using HyperFEM
using Test

@testset "ConstantEnergyLaw" begin
  law = ConstantEnergyLaw()
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end

@testset "ConstantCvLaw" begin
  law = ConstantCvLaw(θr=273.15)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end

@testset "EntropicElasticityLaw" begin
  law = EntropicElasticityLaw(θr=273.15, γ=0.55)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end

@testset "NonlinearMeltingLaw" begin
  law = NonlinearMeltingLaw(θr=273.15, θM=400.0, γ=0.55)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end

@testset "NonlinearSofteningLaw" begin
  law = NonlinearSofteningLaw(θr=273.15, θT=300.0, γ=2.0, δ=0.5)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end

@testset "PolynomialLaw" begin
  law = PolynomialLaw(θr=273.15, a=1.1, b=2.2, c=3.3)
  f, df, ddf = law()
  for θ ∈ 200.0:50:400
    @test df(θ) ≈ ForwardDiff.derivative(f, θ)
    @test ddf(θ) ≈ ForwardDiff.derivative(df, θ)
  end
end
