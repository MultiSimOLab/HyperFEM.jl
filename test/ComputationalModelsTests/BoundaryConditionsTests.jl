using Gridap
using Gridap.FESpaces
using ForwardDiff


@testset "Boundary conditions API" begin

  model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (2, 2))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "bottom", [1, 2, 5])
  add_tag_from_tags!(labels, "top", [3, 4, 6])
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)

  dirichlet_values(bc, Λ) = get_dirichlet_dof_values(TrialFESpace(TestFESpace(model, reffe, bc), bc, Λ))

  # Constant, separable and masked conditions
  bc = BoundaryConditions(
    BoundaryCondition("bottom", [0.0, 0.0]),
    BoundaryCondition("top", [1.0, 2.0], Λ -> Λ; mask=(true, false)))
  @test get_tags(bc) == ["bottom", "top"]
  @test get_masks(bc) == [[true, true], [true, false]]
  V = TestFESpace(model, reffe, bc)
  @test num_dirichlet_dofs(V) == 3 * 2 + 3 * 1
  U = TrialFESpace(V, bc, 0.5)
  dof_tags = get_dirichlet_dof_tag(V)
  @test all(get_dirichlet_dof_values(U)[dof_tags.==1] .== 0.0)
  @test all(get_dirichlet_dof_values(U)[dof_tags.==2] .≈ 0.5)
  TrialFESpace!(U, bc, 1.0)
  @test all(get_dirichlet_dof_values(U)[dof_tags.==2] .≈ 1.0)

  # Explicit keyword arguments override the masks of the boundary conditions
  @test num_dirichlet_dofs(TestFESpace(model, reffe, bc; dirichlet_masks=[[true, true], [true, true]])) == 12

  # Non-separable prescription: rigid rotation of the top boundary
  rotation(x, Λ) = TensorValue(cos(Λ * π / 2), sin(Λ * π / 2), -sin(Λ * π / 2), cos(Λ * π / 2)) ⋅ x - x
  rot_bc = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", NonSeparable(rotation)))
  @test get_masks(rot_bc) === nothing
  uh = FEFunction(TrialFESpace(TestFESpace(model, reffe, rot_bc), rot_bc, 1.0), zeros(num_free_dofs(TestFESpace(model, reffe, rot_bc))))
  @test uh(Point(1.0, 1.0)) ≈ VectorValue(-2.0, 0.0)

  # Parametric prescription, equivalent to the non-separable one
  par_bc = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", Parametric(Λ -> (x -> rotation(x, Λ)))))
  @test dirichlet_values(par_bc, 0.3) ≈ dirichlet_values(rot_bc, 0.3)

  # Increment between two load parameters
  ΔU = TrialFESpace(TestFESpace(model, reffe, rot_bc), rot_bc)
  TrialFESpace!(ΔU, rot_bc, 1.0, 0.25)
  @test get_dirichlet_dof_values(ΔU) ≈ dirichlet_values(rot_bc, 1.0) - dirichlet_values(rot_bc, 0.75)

  # Derivatives with respect to the load parameter
  drotation(x, Λ) = π / 2 * TensorValue(-sin(Λ * π / 2), cos(Λ * π / 2), -cos(Λ * π / 2), -sin(Λ * π / 2)) ⋅ x
  x = VectorValue(0.3, 0.7)
  der_bc = BoundaryConditions(
    BoundaryCondition("bottom", [1.0, 2.0], Λ -> Λ^2),
    BoundaryCondition("top", x -> x[1], Λ -> sin(Λ)),
    BoundaryCondition("top", NonSeparable(rotation)),
    BoundaryCondition("top", Parametric(Λ -> (x -> rotation(x, Λ)))))
  d = get_time_derivative(der_bc, 0.4)
  @test d[1] ≈ VectorValue(0.8, 1.6)
  @test d[2](x) ≈ 0.3 * cos(0.4)
  @test d[3](x) ≈ drotation(x, 0.4)
  @test d[4](x) ≈ drotation(x, 0.4)

  # ForwardDiff differentiates scalar loads, and VectorValues once wrapped
  d = get_time_derivative(der_bc, 0.4; backend=ForwardDiff.derivative)
  @test d[1] ≈ VectorValue(0.8, 1.6)
  @test d[2](x) ≈ 0.3 * cos(0.4)
  fd_vector = (f, Λ) -> VectorValue(ForwardDiff.derivative(λ -> Gridap.TensorValues.get_array(f(λ)), Λ))
  d = get_time_derivative(BoundaryConditions(der_bc[3], der_bc[4]), 0.4; backend=fd_vector)
  @test d[1](x) ≈ drotation(x, 0.4)
  @test d[2](x) ≈ drotation(x, 0.4)

  # Empty boundary conditions
  V0 = TestFESpace(model, reffe, NothingBC())
  @test num_dirichlet_dofs(V0) == 0
  @test TrialFESpace(V0, NothingBC(), 1.0) === V0
  dΓ0 = get_Neumann_dΓ(model, NothingBC(), 2)
  @test isempty(dΓ0)
  @test norm(assemble_vector(v -> residual_Neumann(NothingBC(), v, dΓ0, 1.0), V0)) == 0.0

  # Neumann conditions share the same type
  neumann = NeumannBC(["top"], [[1.0, 0.0]], [Λ -> 2Λ])
  dΓ = get_Neumann_dΓ(model, neumann, 2)
  @test sum(assemble_vector(v -> residual_Neumann(neumann, v, dΓ, 0.5), V0)) ≈ -1.0

end
