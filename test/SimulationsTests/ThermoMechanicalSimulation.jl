using Gridap
using HyperFEM
using HyperFEM: jacobian, solve!

# Steady thermo-mechanical problem, monolithic: a bar clamped and kept at the reference temperature at x = 0 and
# heated at x = L (temperature increment ramped with the load factor), with the ThermoMechModel weak forms. The
# steady temperature increment is linear in x (exact with P1 elements) and the heated bar expands.
# Returns the free values of (u, δθ) and the axial displacement of the heated end.
function thermo_mechanical_simulation(; backend=:lazy, verbose=false)
  L, w = 1.0, 0.2
  geometry = CartesianDiscreteModel((0, L, 0, w, 0, w), (8, 2, 2))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "cold", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "hot", CartesianTags.face1YZ⁺)

  law = EntropicElasticityLaw(θr=1.0, γ=0.0)   # f(θ) = θ/θr
  model = ThermoMechModel(ThermalModel(Cv=1.0, θr=1.0, α=0.5, κ=1.0), NeoHookean3D(λ=10.0, μ=1.0), law)
  k = (Kinematics(Mechano, Solid), Kinematics(Thermo, Solid))

  order = 1
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2 * order)

  Du = BoundaryConditions(BoundaryCondition("cold", [0.0, 0.0, 0.0]))
  Dθ = BoundaryConditions(BoundaryCondition("cold", 0.0, EvolutionFunctions.constant()),
                          BoundaryCondition("hot", 0.2, EvolutionFunctions.ramp()))
  D_bc = MultiFieldBC([Du, Dθ])

  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  reffeθ = ReferenceFE(lagrangian, Float64, order)
  Vu = TestFESpace(geometry, reffeu, D_bc[1], conformity=:H1)
  Vθ = TestFESpace(geometry, reffeθ, D_bc[2], conformity=:H1)
  Uu = TrialFESpace(Vu, D_bc[1], 0.0)
  Uθ = TrialFESpace(Vθ, D_bc[2], 0.0)
  V = MultiFieldFESpace([Vu, Vθ])
  U = MultiFieldFESpace([Uu, Uθ])

  res(Λ) = ((u, θ), (v, vθ)) -> residual(model, k, (u, θ), (v, vθ), dΩ)
  jac(Λ) = ((u, θ), (du, dθ), (v, vθ)) -> jacobian(model, k, (u, θ), (du, dθ), (v, vθ), dΩ)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12, verbose)
  comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls, backend)

  x, flag = solve!(comp_model; stepping=(nsteps=4, maxbisec=4))
  uh, θh = get_state(comp_model)
  ux_hot = component_LInf(uh, :x, Ω)
  θmid = θh(Point(L / 2, w / 2, w / 2))
  (; x=copy(x), flag, ux_hot, θmid)
end

if abspath(PROGRAM_FILE) == @__FILE__
  r = thermo_mechanical_simulation(verbose=true)
  @show r.flag r.ux_hot r.θmid
end
