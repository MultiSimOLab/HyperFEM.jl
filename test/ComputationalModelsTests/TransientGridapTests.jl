# The schemes of TransientModel against Gridap's ODE solvers on the same problem: Theta(θ) against
# ThetaMethod (one-leg, the same algebraic problem) and SDIRK(2) against RungeKutta with the same Butcher
# tableau — to round-off, with moving Dirichlet values.
using Gridap, Gridap.FESpaces, Gridap.ODEs
using HyperFEM
using LinearAlgebra
using Test

@testset "TransientModel against Gridap's ODE solvers (heat, moving Dirichlet value)" begin
  model = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [7]); add_tag_from_tags!(labels, "rest", [1, 2, 3, 4, 5, 6, 8])
  reffe = ReferenceFE(lagrangian, Float64, 1)
  κ = 0.3; t_end = 0.4
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  # HyperFEM: the structured level with the mass form
  bc = BoundaryConditions(BoundaryCondition("left", 1.0, EvolutionFunctions.ramp(1.0)), BoundaryCondition("rest", 0.0, EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1); U = TrialFESpace(V, bc, 0.0)
  res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ
  jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  mass = (u, v) -> ∫(u * v)dΩ
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-12, verbose=0)
  hyperfem(scheme, n) = copy(solve!(TransientModel(res, jac, U, V, bc; scheme, mass, nls); stepping=TimeStepping(; Δt=t_end / n, t_end))[1])
  # Gridap: a transient linear operator on a transient trial space with the same Dirichlet ramp
  Vg = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["left", "rest"])
  gl(t) = x -> t; gr(t) = x -> 0.0
  Ug = TransientTrialFESpace(Vg, [gl, gr])
  stiffness(t, du, v) = ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  massg(t, dut, v) = ∫(dut * v)dΩ
  forcing(t, v) = ∫(0.0 * v)dΩ
  op = TransientLinearFEOperator((stiffness, massg), forcing, Ug, Vg; constant_forms=(true, true))
  function gridap(solver)
    uh0 = interpolate_everywhere(0.0, Ug(0.0))
    x = nothing
    for (t, uh) in solve(solver, op, 0.0, t_end, uh0)
      x = copy(get_free_dof_values(uh))
    end
    x
  end
  # the same dofs: both spaces are built on the same model with the same tags, so the free dofs coincide
  @test num_free_dofs(U) == num_free_dofs(Vg)
  for (θ, n) in ((1.0, 8), (0.5, 8), (0.7, 8), (0.5, 32))
    xh = hyperfem(Theta(θ), n)
    xg = gridap(ThetaMethod(LUSolver(), t_end / n, θ))
    @test norm(xh - xg) <= 1e-13 * norm(xg)
  end
  sd = SDIRK(2)
  tableau = Gridap.ODEs.GenericTableau(Matrix(sd.A), Vector(sd.b), 2)
  for n in (8, 32)
    xh = hyperfem(sd, n)
    xg = gridap(RungeKutta(LUSolver(), t_end / n, tableau))
    @test norm(xh - xg) <= 1e-13 * norm(xg)
  end
end
