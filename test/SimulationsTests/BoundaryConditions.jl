using Gridap
using Gridap.FESpaces

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
  dir_u_tags_air = ["uair_fixed", "Interface"]
  dir_u_values_air = [[0.0, 0.0], Λ -> (x -> Mapping(x, Λ))]
  dir_u_timesteps_air = [evolu, nothing]
  Du = DirichletBC(dir_u_tags_air, dir_u_values_air, dir_u_timesteps_air)

  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)

  Vu = TestFESpace(Ω, reffeu, Du, conformity=:H1)
  Uu = TrialFESpace(Vu, Du, 1.0)

@test isapprox(norm(Uu.dirichlet_values) ,  0.30990321069650995,rtol=1e-14)
  TrialFESpace!(Uu, Du, 0.4)
@test isapprox(norm(Uu.dirichlet_values) ,  0.12396128427860398,rtol=1e-14)
  TrialFESpace!(Uu, Du, 1.0)
@test isapprox(norm(Uu.dirichlet_values) ,  0.30990321069650995,rtol=1e-14)

 
end


@testset "Mesh movement stabilization" begin

  meshfile = "test_BC2.msh"
  geomodel = GmshDiscreteModel(projdir("test/models/" * meshfile))

  Params = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]

  model_vacuum_mech_ = NonlinearMooneyRivlin2D_CV(λ=1 * Params[1], μ1=Params[1], μ2=0.0, α1=6.0, α2=1.0, γ=6.0)
  model_vacuum_mech = HessianRegularization(mechano=model_vacuum_mech_, δ=1e-6 * Params[1])


  # Domains
  order = 1
  degree = 1 * order
  bdegree = 1 * order
  Ωair = Interior(geomodel, tags=["Air"])
  dΩair = Measure(Ωair, degree)
  Ωsolid = Interior(geomodel, tags=["Solid"])

  Γair_int = BoundaryTriangulation(Ωair, tags="Interface")
  nair_int = get_normal_vector(Γair_int)
  dΓair_int = Measure(Γair_int, bdegree)

  Γsf = InterfaceTriangulation(Ωsolid, Ωair)
  nΓsf = get_normal_vector(Γsf)
  dΓsf = Measure(Γsf, bdegree)


  L = 0.1
  function Mapping(x, Λ)
    θmax = -1.0 * π / 2
    #θmax   =  -2.0*π/2*0.001     
    A = 0.3 * L * Λ
    #     xd     =  [x[1],  x[2]+A*((x[1]+L/2)/L)^2]
    xd = [x[1], x[2] + A * sin(π * (x[1] + L / 2) / L)]

    θ = θmax * Λ
    R = [[cos(θ) -sin(θ)]; [sin(θ) cos(θ)]]
    xd2 = R * (xd + [L / 2, 0]) - [L / 2, 0.0]
    u = (xd2 - [x[1], x[2]])
    return VectorValue(u)
  end


  # FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  reffeJ = ReferenceFE(lagrangian, Float64, order)
  reffeJx = ReferenceFE(lagrangian, Float64, order - 1)

  # Test FE Spaces
  Vu_⁺ = TestFESpace(Ωair, reffeu, dirichlet_tags=["uair_fixed", "Interface"], conformity=:H1)
  Vu_⁻ = TestFESpace(Ωair, reffeu, dirichlet_tags=["uair_fixed", "Interface"], conformity=:H1)

  uh⁺ = interpolate_everywhere(x -> Mapping(x, 1.0), Vu_⁺)
  uh⁻ = interpolate_everywhere(x -> Mapping(x, 0.0), Vu_⁻)

  dir(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ

  evolu(Λ) = Λ
  dir_u_tags_air = ["uair_fixed", "Interface"]
  dir_u_values_air = [[0.0, 0.0], Λ -> dir(Λ)]
  dir_u_timesteps_air = [evolu, nothing]
  Du_air = DirichletBC(dir_u_tags_air, dir_u_values_air, dir_u_timesteps_air)

  Vu = TestFESpace(Ωair, reffeu, Du_air, conformity=:H1)
  Uu = TrialFESpace(Vu, Du_air, 1.0)
  TrialFESpace!(Uu, Du_air, 0.0)

  DΨvacuum_mech = model_vacuum_mech(1.0)
  k = Kinematics(Mechano, Solid)
  F, H, J = get_Kinematics(k; Λ=1.0)

  # Vacuum mechanics
  res_vacmech(Λ) = (u, v) -> ∫((∇(v)' ⊙ (DΨvacuum_mech[2] ∘ (F ∘ (∇(u)')))))dΩair

  jac_vacmech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((DΨvacuum_mech[3] ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩair

  α = CellState(1.0, dΩair)
  linesearch = Injectivity_Preserving_LS(α, Uu, Vu; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95)
  nls_vacmech = Newton_RaphsonSolver(LUSolver(); maxiter=10, rtol=2, verbose=false, linesearch=linesearch)

  xh = FEFunction(Uu, zero_free_values(Uu))
  comp_model_vacmech = StaticNonlinearModel(res_vacmech, jac_vacmech, Uu, Vu, Du_air; nls=nls_vacmech, xh=xh)
  args_vacmech = Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true)

  nsteps = 10
  flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
  Δβ = 1.0 / nsteps
  nbisect = 0
  for t in 0:nsteps-1
    interpolate_everywhere!(x -> Mapping(x, Δβ * (1 + t)), get_free_dof_values(uh⁺), uh⁺.dirichlet_values, Vu_⁺)
    interpolate_everywhere!(x -> Mapping(x, Δβ * (t)), get_free_dof_values(uh⁻), uh⁻.dirichlet_values, Vu_⁻)
    solve!(comp_model_vacmech; args_vacmech...)
  end
 @test isapprox( norm(xh.free_values) ,   2.8132015601158087,rtol=1e-14)


end
