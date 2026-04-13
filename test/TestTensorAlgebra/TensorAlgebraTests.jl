using Gridap.TensorValues
using Gridap.Arrays
using HyperFEM.TensorAlgebra
using Test


@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; Threshold=0.01) == 0.014870878346353422
end


@testset "outer" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  B = TensorValue(5.0, 6.0, 7.0, 8.0)
  u = VectorValue(1.0, 2.0)
  v = VectorValue(3.0, 4.0)
  @test u ⊗ v     == TensorValue(3.0, 6.0, 4.0, 8.0)
  @test u ⊗₁² v   == TensorValue(3.0, 6.0, 4.0, 8.0)
  @test A ⊗ B     == TensorValue(5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0)
  @test A ⊗₁₂³⁴ B == TensorValue(5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0)
  @test A ⊗₁₃²⁴ B == TensorValue(5.0, 10.0, 6.0, 12.0, 15.0, 20.0, 18.0, 24.0, 7.0, 14.0, 8.0, 16.0, 21.0, 28.0, 24.0, 32.0)
  @test A ⊗₁₄²³ B == TensorValue(5.0, 10.0, 6.0, 12.0, 7.0, 14.0, 8.0, 16.0, 15.0, 20.0, 18.0, 24.0, 21.0, 28.0, 24.0, 32.0)
  @test u ⊗₁²³ A  == TensorValue{2,4}(1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0)
  @test A ⊗₁₂³ u  == TensorValue{2,4}(1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0)
  @test A ⊗₁₃² u  == TensorValue{2,4}(1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0)
end


@testset "cross" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  D = TensorValue([4.6 2.1 1.7 3.2 6.5 1.4 9.2 8.0 9.0;
  1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
  5.3 2.0 3.1 1.9 5.4 9.8 0.4 8.8 3.1] * 1e-3)
  @test norm(×ᵢ⁴(A)) == 0.033763886032268264
  @test norm(A × B) == 6.246230863488799e-5
  @test norm(C × B) == 2.4491455542698976e-6
  @test norm(B × C) == 1.104276381618298e-6
  @test norm(D × A) == 0.00012378691368638284
  @test norm(get_array(A) × get_array(B))== 6.246230863488799e-5
end


@testset "inner" begin
  H = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
  G = TensorValue{2,4}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  V = VectorValue(5.0, 6.0)
  @test inner(H,H) == 1496.0
  @test inner(G,G) == 204.0
  @test inner(A,A) == 30.0
  @test inner(V,V) == 61.0
  @test inner(H,A) == TensorValue(90.0, 100.0, 110.0, 120.0)
  @test inner(G,A) == VectorValue(50.0, 60.0)
  @test inner(G,V) == TensorValue(35.0, 46.0, 57.0, 68.0)
  @test H ⊙ A     == TensorValue(90.0, 100.0, 110.0, 120.0)
  @test G ⊙ A     == VectorValue(50.0, 60.0)
  @test G ⊙ V     == TensorValue(35.0, 46.0, 57.0, 68.0)
end


@testset "sum" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
  B = TensorValue(4.1, 5.2, 6.3, 7.4, 8.5, 9.6, 1.7, 2.8, 3.9)
  @test A + B == TensorValue(5.1, 7.2, 9.3, 11.4, 13.5, 15.6, 8.7, 10.8, 12.9)
  @test norm(A + B) ≈ 32.842807431765024
end


@testset "identity" begin
  I2_ = TensorValue(1.0, 0.0, 0.0, 1.0)
  I3_ = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  I4_ = TensorValue(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  I9_ = TensorValue(
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  @test I2_ == I2
  @test I3_ == I3
  @test I4_ == I4
  @test I9_ == I9
end


@testset "contraction" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  H = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
  @test contraction_IP_PJKL(A,H) == TensorValue(7.0, 10.0, 15.0, 22.0, 23.0, 34.0, 31.0, 46.0, 39.0, 58.0, 47.0, 70.0, 55.0, 82.0, 63.0, 94.0)
  @test contraction_IP_JPKL(A,H) == TensorValue(10.0, 14.0, 14.0, 20.0, 26.0, 38.0, 30.0, 44.0, 42.0, 62.0, 46.0, 68.0, 58.0, 86.0, 62.0, 92.0)
end


@testset "sqrt" begin
  A = TensorValue(1.:9...)
  A = A'*A + I3
  sqrtA = TensorValue(sqrt(get_array(A)))
  @test isapprox(sqrt(A), sqrtA, rtol=1e-14)
end


@testset "cofactor" begin
  A = TensorValue(1.:9...) + I3
  cofA = det(A) * inv(A')
  @test isapprox(cof(A), cofA)
end
