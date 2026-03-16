using ForwardDiff
using HyperFEM
using Test

@testset "LogisticLaw" begin
  law = LogisticLaw(273.15, log(300.1), 0.11)
  f, df, ddf = derivatives(law)
  for θ ∈ 200.0:50:400  # NOTE: The numerical derivative of erf is a bad approximation, while the analyitical function uses the exact value.
    @test isapprox(df(θ), ForwardDiff.derivative(f, θ), rtol=1e-3)
    @test isapprox(ddf(θ), ForwardDiff.derivative(df, θ), rtol=1e-10)
  end
end
