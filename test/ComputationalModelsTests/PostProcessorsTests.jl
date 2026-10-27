using Gridap
using HyperFEM
using Test


model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (2, 2, 2))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
u(x) = VectorValue(2x[1], -3x[2], x[1] + x[3])
uh = interpolate(u, FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)))


@testset "L2_projection" begin
  f(x) = x[1] + 2x[2] - x[3]
  V0 = FESpace(Ω, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
  V1 = FESpace(Ω, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
  fh0 = L2_projection(f, V0, dΩ)
  fh1 = L2_projection(f, V1, dΩ)
  @test sum(∫(fh0)dΩ) ≈ sum(∫(f)dΩ)    # The mean value is preserved
  @test sum(∫((fh1 - f) * (fh1 - f))dΩ) < 1e-20    # A linear field is exactly represented
  fh0_old = @test_logs (:warn,) L2_projection(f, dΩ, V0)
  @test get_free_dof_values(fh0_old) ≈ get_free_dof_values(fh0)
end


@testset "interpolate_L2_field" begin
  # Constant fields must be recovered exactly
  ∇u = ∇(uh)
  A = ∇(u)(Point(0.5, 0.5, 0.5))
  Ah = interpolate_L2_field(∇u, dΩ)
  @test Ah isa FEFunction
  @test sum(∫((Ah - A) ⊙ (Ah - A))dΩ) < 1e-20
  Jh = interpolate_L2_field(det ∘ ∇u, dΩ)
  @test sum(∫((Jh - det(A)) * (Jh - det(A)))dΩ) < 1e-20
  bh = interpolate_L2_field(∇u ⋅ VectorValue(1.0, 1.0, 1.0), dΩ)
  b = A ⋅ VectorValue(1.0, 1.0, 1.0)
  @test sum(∫((bh - b) ⋅ (bh - b))dΩ) < 1e-20
  # The destination triangulation can be given explicitly
  Ah2 = interpolate_L2_field(∇u, dΩ, Ω)
  @test get_free_dof_values(Ah2) ≈ get_free_dof_values(Ah)
end


@testset "interpolate_L2_field order" begin
  V2 = FESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2))
  u2(x) = VectorValue(x[1]^2, x[2] * x[3], x[3])
  u2h = interpolate(u2, V2)
  dΩ4 = Measure(Ω, 4)
  Ah = interpolate_L2_field(∇(u2h), dΩ4; order=2)
  # A linear gradient is exactly represented with order 2
  @test sum(∫((Ah - ∇(u2)) ⊙ (Ah - ∇(u2)))dΩ4) < 1e-20
  # The default order is 1
  Ah1 = interpolate_L2_field(∇(u2h), dΩ4)
  @test length(get_free_dof_values(Ah1)) < length(get_free_dof_values(Ah))
  @test_throws ArgumentError interpolate_L2_field(∇(u2h), dΩ4; order=0)
end
