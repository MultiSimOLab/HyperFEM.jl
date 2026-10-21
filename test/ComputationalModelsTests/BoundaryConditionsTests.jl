using Gridap
using Gridap.FESpaces
using ForwardDiff
using GridapSolvers, GridapSolvers.NonlinearSolvers


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
  neumann = BoundaryConditions(BoundaryCondition("top", [1.0, 0.0], Λ -> 2Λ))
  dΓ = get_Neumann_dΓ(model, neumann, 2)
  @test sum(assemble_vector(v -> residual_Neumann(neumann, v, dΓ, 0.5), V0)) ≈ -1.0

  # A load map composes with the loads of the Separable prescriptions
  ramped = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", [1.0, 0.0], Λ -> Λ))
  @test get_space_functions(ramped, 0.5)[2] ≈ VectorValue(0.5, 0.0)
  set_load_map!(ramped, Λ -> 3Λ)
  @test get_space_functions(ramped, 0.5)[2] ≈ VectorValue(1.5, 0.0)
  set_load_map!(ramped, identity)

  # Couplings: a Gridap Interpolable from a non-matching mesh is a space function
  f(x) = VectorValue(x[1]^2, x[1] * x[2])
  reffe2 = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
  uhA = interpolate(f, FESpace(CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (3, 3)), reffe2))
  coupled = BoundaryConditions(
    BoundaryCondition("bottom", [0.0, 0.0]),
    BoundaryCondition("top", Parametric(Λ -> Gridap.CellData.Interpolable(uhA))))
  Vc = TestFESpace(model, reffe2, coupled)
  exact = get_dirichlet_dof_values(TrialFESpace(Vc, (VectorValue(0.0, 0.0), f)))
  @test get_dirichlet_dof_values(TrialFESpace(Vc, coupled, 1.0)) ≈ exact

  # The field itself (no Interpolable) is admissible too
  coupled_field = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", Parametric(Λ -> uhA)))
  Uc = TrialFESpace(Vc, coupled_field, 1.0)
  @test get_dirichlet_dof_values(Uc) ≈ exact

  # Coupling with a field that lives only on the faces of its tag (an FESpace on a BoundaryTriangulation):
  # it must never be evaluated off the interface, and the point search must be cached
  Γ = BoundaryTriangulation(model, tags="top")
  uhΓ = interpolate_everywhere(f, FESpace(Γ, reffe2))
  coupledΓ = BoundaryConditions(
    BoundaryCondition("bottom", [0.0, 0.0]),
    BoundaryCondition("top", Parametric(Λ -> Gridap.CellData.Interpolable(uhΓ))))
  UΓ = TrialFESpace(Vc, coupledΓ, 1.0)
  @test get_dirichlet_dof_values(UΓ) ≈ exact
  get_free_dof_values(uhΓ) .*= 2.0                               # field updated in place (staggered loop)
  TrialFESpace!(UΓ, coupledΓ, 1.0)
  @test get_dirichlet_dof_values(UΓ) ≈ get_dirichlet_dof_values(TrialFESpace(Vc, (VectorValue(0.0, 0.0), x -> 2 * f(x))))
  @test @allocated(TrialFESpace!(UΓ, coupledΓ, 1.0)) < 10^5     # no point-search tree rebuilt
  # a new CellField generated at every Λ, and the increment variant
  scaledΓ = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", Parametric(Λ -> Λ * uhΓ)))
  UlΓ = TrialFESpace(Vc, scaledΓ, 1.0)
  @test get_dirichlet_dof_values(UlΓ) ≈ get_dirichlet_dof_values(UΓ)
  TrialFESpace!(UlΓ, scaledΓ, 1.0, 0.5)
  @test get_dirichlet_dof_values(UlΓ) ≈ 0.5 * get_dirichlet_dof_values(UΓ)

  # The applicator (prescriptions evaluated at the Dirichlet nodes only) gives Gridap's values for every
  # kind of prescription, on vector and scalar spaces, with partial masks and nested TrialFESpaces
  gridap_values(V, bc, Λ) = get_dirichlet_dof_values(TrialFESpace(V, HyperFEM.ComputationalModels.get_space_functions(bc, Λ)))
  uhM = interpolate(f, FESpace(model, reffe2))                   # a field on the same model (Gridap can interpolate it)
  mix = BoundaryConditions(                                       # one condition per tag (tag_7/8: left/right edges)
    BoundaryCondition("bottom", [0.3, -0.2], Λ -> 2Λ; mask=(true, false)),
    BoundaryCondition("top", x -> VectorValue(x[1] * x[2], x[2]^2), Λ -> cos(Λ)),
    BoundaryCondition("tag_7", NonSeparable(rotation); mask=(false, true)),
    BoundaryCondition("tag_8", Parametric(Λ -> Λ * uhM)))
  Vmix = TestFESpace(model, reffe2, mix)
  Umix = TrialFESpace(Vmix, mix, 0.0)
  @test HyperFEM.ComputationalModels._applicator(Umix, mix) !== nothing
  for Λ in (0.0, 0.37, 1.0)
    TrialFESpace!(Umix, mix, Λ)
    @test get_dirichlet_dof_values(Umix) ≈ gridap_values(Vmix, mix, Λ)
    UUmix = TrialFESpace(Umix, mix, Λ)                            # nested, as ∆U of the drivers
    @test get_dirichlet_dof_values(UUmix) ≈ gridap_values(Vmix, mix, Λ)
  end
  @test @allocated(TrialFESpace!(Umix, mix, 0.5)) < 10^5
  reffe1 = ReferenceFE(lagrangian, Float64, 2)
  φA = interpolate(x -> x[1] - x[2]^2, FESpace(CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (3, 3)), reffe1))
  smix = BoundaryConditions(
    BoundaryCondition("bottom", 2.5, Λ -> Λ),
    BoundaryCondition("top", x -> x[1] + x[2]),
    BoundaryCondition("tag_7", NonSeparable((x, Λ) -> Λ * x[1])),
    BoundaryCondition("tag_8", Parametric(Λ -> Gridap.CellData.Interpolable(φA))))
  Vs = TestFESpace(model, reffe1, smix)
  Us = TrialFESpace(Vs, smix, 0.8)
  @test get_dirichlet_dof_values(Us) ≈ gridap_values(Vs, smix, 0.8)

  # Couplings with a partial mask: only the masked components of the field are prescribed
  for mask in ((true, false), (false, true))
    cmask = BoundaryConditions(
      BoundaryCondition("bottom", [0.0, 0.0]),
      BoundaryCondition("top", Parametric(Λ -> Λ * uhM); mask=mask))          # field on the whole model
    Vcm = TestFESpace(model, reffe2, cmask)
    @test num_dirichlet_dofs(Vcm) < num_dirichlet_dofs(Vc)                    # the free component is not Dirichlet
    @test get_dirichlet_dof_values(TrialFESpace(Vcm, cmask, 0.7)) ≈ gridap_values(Vcm, cmask, 0.7)
    cmaskΓ = BoundaryConditions(
      BoundaryCondition("bottom", [0.0, 0.0]),
      BoundaryCondition("top", Parametric(Λ -> Gridap.CellData.Interpolable(uhΓ)); mask=mask))   # field only on the interface
    exactΓ = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", x -> 2 * f(x); mask=mask))
    @test get_dirichlet_dof_values(TrialFESpace(Vcm, cmaskΓ, 1.0)) ≈ get_dirichlet_dof_values(TrialFESpace(Vcm, exactΓ, 1.0))
  end

  # Load map (StaggeredModel): the Separable loads see f(Λ) (composed), the other prescriptions the raw Λ
  set_load_map!(mix, Λ -> 0.5 * Λ)
  TrialFESpace!(Umix, mix, 0.8)
  @test get_dirichlet_dof_values(Umix) ≈ gridap_values(Vmix, mix, 0.8)   # same path, same map
  set_load_map!(mix, identity)
  mapped = get_dirichlet_dof_values(TrialFESpace(Vmix, mix, 0.4))        # Separable at 0.4 ...
  raw = get_dirichlet_dof_values(TrialFESpace(Vmix, mix, 0.8))           # ... the others at 0.8
  tagof = get_dirichlet_dof_tag(Vmix)
  @test get_dirichlet_dof_values(Umix)[tagof.<=2] ≈ mapped[tagof.<=2]
  @test get_dirichlet_dof_values(Umix)[tagof.>=3] ≈ raw[tagof.>=3]
  TrialFESpace!(Umix, mix, 0.8)
  @test get_dirichlet_dof_values(Umix) ≈ raw
  # Increment variant on every kind of prescription: difference of the two states, no full evaluation
  TrialFESpace!(Umix, mix, 0.8, 0.3)
  @test get_dirichlet_dof_values(Umix) ≈ gridap_values(Vmix, mix, 0.8) - gridap_values(Vmix, mix, 0.5)
  @test @allocated(TrialFESpace!(Umix, mix, 0.8, 0.3)) < 2 * 10^5          # two evaluations of the non-Separable ones
  set_load_map!(mix, Λ -> 0.5 * Λ)
  TrialFESpace!(Umix, mix, 0.8, 0.3)
  set_load_map!(mix, identity)
  @test get_dirichlet_dof_values(Umix)[tagof.<=2] ≈ (mapped - get_dirichlet_dof_values(TrialFESpace(Vmix, mix, 0.25)))[tagof.<=2]
  @test get_dirichlet_dof_values(Umix)[tagof.>=3] ≈ (raw - get_dirichlet_dof_values(TrialFESpace(Vmix, mix, 0.5)))[tagof.>=3]

  # Multi-field conditions: one per field, NothingBC leaves the field unconstrained
  mbc = MultiFieldBC(bc, NothingBC())
  Vm = MultiFieldFESpace([V, V0])
  Um = TrialFESpace(Vm, mbc, 0.5)
  @test get_dirichlet_dof_values(Um[1]) ≈ dirichlet_values(bc, 0.5)
  @test Um[2] === V0

end


@testset "Dirichlet BC testing analytical mapping" begin

  meshfile = "test_BC1.msh"
  geomodel = GmshDiscreteModel(projdir("test/models/" * meshfile))
  # Domains
  order = 1
  degree = 1 * order
  Ω = Interior(geomodel, tags=["Air"])
  dΩ = Measure(Ω, degree)

  function Mapping(x, Λ)
    L = 0.1
    A = 0.98 * L
    xd = [x[1], x[2] + A * sin(2 * π * (x[1] + L / 2) / L)]
    u = (xd - [x[1], x[2]]) * Λ
    return VectorValue(u)
  end

  evolu(Λ) = Λ
  Du = BoundaryConditions(
    BoundaryCondition("uair_fixed", [0.0, 0.0], evolu),
    BoundaryCondition("Interface", Parametric(Λ -> (x -> Mapping(x, Λ)))))

  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)

  Vu = TestFESpace(Ω, reffeu, Du, conformity=:H1)
  Uu = TrialFESpace(Vu, Du, 1.0)

  @test isapprox(norm(Uu.dirichlet_values), 0.30990321069650995, rtol=1e-14)
  TrialFESpace!(Uu, Du, 0.4)
  @test isapprox(norm(Uu.dirichlet_values), 0.12396128427860398, rtol=1e-14)
  TrialFESpace!(Uu, Du, 1.0)
  @test isapprox(norm(Uu.dirichlet_values), 0.30990321069650995, rtol=1e-14)

end


@testset "StaggeredModel restores the load map" begin

  # A staggered model whose loop throws half-way must not leave the conditions with the ramp of its last step
  model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (4, 4))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "bottom", [1, 2, 5])
  add_tag_from_tags!(labels, "top", [3, 4, 6])
  dΩ = Measure(Triangulation(model), 2)
  res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u))dΩ
  jac(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du))dΩ
  nls = NewtonSolver(LUSolver(); maxiter=5, rtol=1e-10, verbose=false)
  function poisson(top)                                           # two independent Laplace problems
    bcp = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("top", top, Λ -> Λ))
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), bcp)
    Up = TrialFESpace(Vp, bcp, 0.0)
    up⁺ = FEFunction(Up, zero_free_values(Up))
    up⁻ = FEFunction(TrialFESpace(Vp, bcp, 0.0), zero_free_values(Up))
    StaticNonlinearModel(res, jac, Up, Vp, bcp; nls=nls, xh=up⁺), up⁺, up⁻, bcp, Up
  end
  m1, uh1⁺, uh1⁻, bc, U = poisson(1.0)
  m2, uh2⁺, uh2⁻, bc2, _ = poisson(2.0)
  sm = StaggeredModel((m1, m2), (uh1⁺, uh2⁺), (uh1⁻, uh2⁻))
  args = (Dict(:stepping => (nsteps=1, maxbisec=1)), Dict(:stepping => (nsteps=1, maxbisec=1)))
  load_map = HyperFEM.ComputationalModels.load_map

  @test_throws ErrorException solve!(sm; stepping=(nsteps=4, nsubsteps=1, maxbisec=1), kargsolve=args, verbose=false,
                                     presolver=(τ, ∆τ) -> τ == 2 && error("inner failure"))
  @test load_map(bc, 0.3) == 0.3 && load_map(bc2, 0.3) == 0.3     # identity again, not the ramp of step 1
  solve!(sm; stepping=(nsteps=4, nsubsteps=1, maxbisec=1), kargsolve=args, verbose=false)
  @test load_map(bc, 0.3) == 0.3
  @test maximum(get_dirichlet_dof_values(U)) ≈ 1.0                # full load at the end of the staggered loop

end
