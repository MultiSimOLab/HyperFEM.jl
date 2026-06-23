using Gridap, Gridap.FESpaces, GridapSolvers, GridapSolvers.NonlinearSolvers
using HyperFEM
using HyperFEM: jacobian


function staggered_electro_mechanical_simulation(; is_vtk=true, verbose=true)
  
  # Problem name
  pname = stem(@__FILE__)
  folder = projdir("data", "sims", pname)
  outpath = joinpath(folder, pname)
  setupfolder(folder)
  
  # Geometry and discrete model
  domain = (0.0, 0.1, 0.0, 0.01, 0.0, 0.002)
  partition = (8, 2, 2)
  geometry = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "fixedu", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
  add_tag_from_vertex_filter!(labels, "midsuf", geometry, x -> x[3] ≈ 0.001)

  # Constitutive model
  physmodel_mec = NeoHookean3D(λ=10.0, μ=1.0)
  physmodel_elec = IdealDielectric(ε=1.0)
  physmodel = ElectroMechModel(physmodel_elec, physmodel_mec)

  # Setup integration
  order = 2
  degree = 2 * order
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)

  # Dirichlet conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedu"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolφ(Λ) = Λ
  dir_φ_tags = ["midsuf", "topsuf"]
  dir_φ_values = [0.0, 0.00005]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  # FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  reffeφ = ReferenceFE(lagrangian, Float64, order)

  # Test FE Spaces
  Vu = TestFESpace(geometry, reffeu, Du, conformity=:H1)
  Vφ = TestFESpace(geometry, reffeφ, Dφ, conformity=:H1)

  # Trial FE Spaces and state variables
  Uu = TrialFESpace(Vu, Du, 1.0)
  uh⁺ = FEFunction(Uu, zero_free_values(Uu))

  Uu⁻ = TrialFESpace(Vu, Du, 1.0)
  uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu⁻))

  Uφ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))

  Uφ⁻ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))

  # Kinematics
  k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))

  # Electro
  Mechano_coupling(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ
  res_elec(Λ) = (φ, vφ) -> residual(physmodel, Electro, k, (Mechano_coupling(Λ), φ), vφ, dΩ)
  jac_elec(Λ) = (φ, dφ, vφ) -> jacobian(physmodel, Electro, k, (Mechano_coupling(Λ), φ), dφ, vφ, dΩ)

  # Mechano
  Electro_coupling(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
  res_mec(Λ) = (u, v) -> residual(physmodel, Mechano, k, (u, Electro_coupling(Λ)), v, dΩ)
  jac_mec(Λ) = (u, du, v) -> jacobian(physmodel, Mechano, k, (u, Electro_coupling(Λ)), du, v, dΩ)

  # nonlinear solver
  ls = LUSolver()
  nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=verbose)
  comp_model_elec = StaticNonlinearModel(res_elec, jac_elec, Uφ, Vφ, Dφ; nls=nls_, xh=φh⁺)
  comp_model_mec = StaticNonlinearModel(res_mec, jac_mec, Uu, Vu, Du; nls=nls_, xh=uh⁺)
  comp_model= StaggeredModel((comp_model_elec,comp_model_mec), (φh⁺,uh⁺), (φh⁻,uh⁻))

  args_elec = Dict(:stepping => (nsteps=1, maxbisec=1))
  args_mec  = Dict(:stepping => (nsteps=5, maxbisec=1))
  args=(args_elec,args_mec)

  x = solve!(comp_model; stepping=(nsteps=5, nsubsteps=1, maxbisec=1), kargsolve=args)

  if is_vtk
    writevtk(Ω, outpath, cellfields=["φh" => φh⁺, "uh" => uh⁺])
  end

  return x
end


# staggered_electro_mechanical_simulation()
