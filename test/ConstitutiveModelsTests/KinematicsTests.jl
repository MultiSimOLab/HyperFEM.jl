using Gridap.TensorValues
using ForwardDiff
using LinearAlgebra: det, inv
using HyperFEM.TensorAlgebra
using HyperFEM.PhysicalModels: cof, logreg, ∂log∂J, ∂∂log∂JJ, trAA, _∂H∂F_2D
using Test


@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; threshold=0.01) == 0.014870878346353422
  @test ForwardDiff.derivative(logreg, J) ≈ ∂log∂J(J)
  @test ForwardDiff.derivative(∂log∂J, J) ≈ ∂∂log∂JJ(J)
  # below the threshold, the regularization is a C² extension of log
  t = 0.01
  @test logreg(t; threshold=t) ≈ log(t)
  @test ForwardDiff.derivative(J -> logreg(J; threshold=t), 0.5t) ≈ ∂log∂J(0.5t; threshold=t)
  @test ForwardDiff.derivative(J -> ∂log∂J(J; threshold=t), 0.5t) ≈ ∂∂log∂JJ(0.5t; threshold=t)
end


@testset "cofactor" begin
  A = TensorValue(1.:9...) + I3D
  cofA = det(A) * inv(A')
  @test isapprox(cof(A), cofA)
end


@testset "cofactor derivative 2D" begin
  cof2D(F) = [F[2, 2] -F[2, 1]; -F[1, 2] F[1, 1]]
  F = TensorValue(1.1, 0.2, 0.3, 0.9)
  ∂H∂F = ForwardDiff.jacobian(cof2D, get_array(F))
  @test _∂H∂F_2D() == FourthOrderTensorValue(Tuple(∂H∂F))
end


@testset "trAA" begin
  A = TensorValue(1.:9...)
  @test trAA(A) == A ⊙ A
end
