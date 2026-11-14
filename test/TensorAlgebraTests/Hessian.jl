using Gridap
using Gridap.Geometry: simplexify
using Gridap.TensorValues: ThirdOrderTensorValue
using HyperFEM
using Test

# Global quadratics are reproduced exactly by P2 (simplices) and by isoparametric Q2 on any
# bilinear/trilinear cell, so a correct Hessian gives the exact values up to round-off.
@testset "Hessian of FE fields" begin
  dist2(x) = VectorValue(x[1] + 0.12*sin(π*x[1])*sin(2π*x[2]), x[2] + 0.10*sin(2π*x[1])*sin(π*x[2]))
  quads     = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
  quads_d   = CartesianDiscreteModel((0, 1, 0, 1), (4, 4); map = dist2)
  meshes2D  = [("quads", quads, true), ("distorted quads", quads_d, false),
               ("triangles", simplexify(quads), true), ("triangles from distorted", simplexify(quads_d), true)]

  u2(x) = VectorValue(x[1]*x[2], x[1]^2)       # |∇∇u|² = 6
  g2(x) = x[1]^2 + x[1]*x[2] + 2x[2]^2         # ∇∇g = [2 1; 1 4], |∇∇g|² = 22

  function check(model, ufun, T, value, H)
    Ω = Triangulation(model); dΩ = Measure(Ω, 6)
    Γ = SkeletonTriangulation(model); dΓ = Measure(Γ, 6)
    uh = interpolate(ufun, FESpace(model, ReferenceFE(lagrangian, T, 2)))
    area = sum(∫(1)dΩ)
    @test sum(∫(H(uh) ⊙ H(uh))dΩ) ≈ value*area rtol = 1e-10
    @test abs(sum(∫(jump(H(uh)) ⊙ jump(H(uh)))dΓ)) < 1e-12
  end

  for (name, model, affine) in meshes2D
    @testset "$name" begin
      check(model, u2, VectorValue{2,Float64}, 6.0, hessian_nonaffine)
      check(model, g2, Float64, 22.0, hessian_nonaffine)
      if affine    # Gridap's ∇∇ only supports affine cell maps
        check(model, u2, VectorValue{2,Float64}, 6.0, ∇∇)
        check(model, g2, Float64, 22.0, ∇∇)
      end
    end
  end

  @testset "distorted hexahedra" begin
    dist3(x) = VectorValue(x[1] + 0.08*sin(π*x[1])*sin(2π*x[2])*sin(π*x[3]),
                           x[2] + 0.06*sin(π*x[1])*sin(π*x[2])*sin(2π*x[3]), x[3])
    hexes = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2); map = dist3)
    u3(x) = VectorValue(x[1]*x[2], x[1]^2, x[2]*x[3])    # |∇∇u|² = 2 + 4 + 2 = 8
    check(hexes, u3, VectorValue{3,Float64}, 8.0, hessian_nonaffine)
  end

  @testset "outer of vector and tensor is Gridap's" begin
    v = VectorValue(1.0, 2.0); A = TensorValue(1.0, 2.0, 3.0, 4.0)
    @test v ⊗ A isa ThirdOrderTensorValue{2,2,2}
    @test A ⊗ v isa ThirdOrderTensorValue{2,2,2}
    @test Tuple(v ⊗ A) == Tuple(v ⊗₁²³ A)
    @test Tuple(A ⊗ v) == Tuple(A ⊗₁₂³ v)
  end
end
