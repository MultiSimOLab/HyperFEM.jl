using HyperFEM
using HyperFEM: jacobian, solve!
using HyperFEM.ComputationalModels.PostMetrics
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels.EvolutionFunctions
using Gridap, GridapGmsh, GridapSolvers
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces


function staggered_visco_electric_simulation(; t_end=2, writevtk=true, verbose=true)

  pname = stem(@__FILE__)
  folder = projdir("data", "sims", pname)
  outpath = joinpath(folder, pname)
  setupfolder(folder; remove=".vtu")

  long  = 0.050  # m
  width = 0.005  # m
  thick = 0.001  # m
  domain = (0.0, long, 0.0, width, 0.0, thick)
  partition = (8, 2, 2)
  geometry = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
  add_tag_from_vertex_filter!(labels, geometry, "mid", x -> x[3] ≈ 0.5thick)

  # Constitutive model
  hyper_elastic_model = NeoHookean3D(λ=1e6, μ=1.4e4)
  viscous_branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=5.6e4); τ=0.82)
  viscous_branch_2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=3.4e4); τ=10.7)
  visco_elastic_model = GeneralizedMaxwell(hyper_elastic_model, viscous_branch_1, viscous_branch_2)
  elec_model = IdealDielectric(ε=1.0)
  cons_model = ElectroMechModel(mechano=visco_elastic_model, electro=elec_model)
  ku = Kinematics(Mechano, Solid)
  ke = Kinematics(Electro, Solid)
  F, _... = get_Kinematics(ku)

  # Setup integration
  order = 1
  degree = 2 * order
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)
  Δt = 0.05   # s
  update_time_step!(cons_model, Δt)

  # Dirichlet boundary conditions 
  dir_u_tags = ["fixed"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [constant()]
  dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  dir_φ_tags = ["bottom", "mid"]
  dir_φ_values = [0.0, 0.1]
  dir_φ_timesteps = [constant(), ramp(1.0)]
  dirichlet_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  # Finite Elements
  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  reffeφ = ReferenceFE(lagrangian, Float64, order)

  # Test FE Spaces
  Vu = TestFESpace(geometry, reffeu, dirichlet_u, conformity=:H1)
  Vφ = TestFESpace(geometry, reffeφ, dirichlet_φ, conformity=:H1)

  # Trial FE Spaces and state variables
  Uu = TrialFESpace(Vu, dirichlet_u, 1.0)
  uh⁺ = FEFunction(Uu, zero_free_values(Uu))

  Uu⁻ = TrialFESpace(Vu, dirichlet_u, 1.0)
  uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu⁻))

  Uφ = TrialFESpace(Vφ, dirichlet_φ, 1.0)
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))

  Uφ⁻ = TrialFESpace(Vφ, dirichlet_φ, 1.0)
  φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))

  Fh  = F∘∇(uh⁺)'
  Fh⁻ = F∘∇(uh⁻)'
  A   = CellState(cons_model, dΩ)

  # Electrical staggered step
  res_elec(Λ) = (φ, vφ) -> residual(cons_model, Electro, (ku, ke), (uh⁺, φ), vφ, dΩ, 0.0, Fh⁻, A...)
  jac_elec(Λ) = (φ, dφ, vφ) -> jacobian(cons_model, Electro, (ku, ke), (uh⁺, φ), dφ, vφ, dΩ, 0.0, Fh⁻, A...)

  # Mechanical staggered step
  res_mec(Λ) = (u, v) -> residual(cons_model, Mechano, (ku, ke), (u, φh⁺), v, dΩ, 0.0, Fh⁻, A...)
  jac_mec(Λ) = (u, du, v) -> jacobian(cons_model, Mechano, (ku, ke), (u, φh⁺), du, v, dΩ, 0.0, Fh⁻, A...)

  # nonlinear solver electro
  ls = LUSolver()
  nls = NewtonSolver(ls; maxiter=20, atol=1e-6, rtol=1e-6, verbose=verbose)
  solver = FESolver(nls)

  # Postprocessor to save results
  function driverpost(pvd, step, time)
    if writevtk && mod(step, 5) == 0
      pvd[time] = createvtk(Ω, outpath * "_$(lpad(step, 3, "0"))", cellfields=["u" => uh⁺, "φ" => φh⁺])
    end
    push!(uz, component_LInf(uh⁺, :z, Ω))
  end

  t = 0:Δt:t_end-Δt
  uz = Float64[]

  pvdstrategy = writevtk ? createpvd : mockpvd
  pvdstrategy(outpath) do pvd  
    u⁻ = get_free_dof_values(uh⁻)
    φ⁻ = get_free_dof_values(φh⁻)

    step = 0
    time = 0
    while time < t_end
      step += 1
      time += Δt
      println("Step: $step")
      println("Time: $time")

      TrialFESpace!(Uφ, dirichlet_φ, time)
      TrialFESpace!(Uu, dirichlet_u, time)
      
      println("Electric staggered step")
      op_elec = FEOperator(res_elec(time), jac_elec(time), Uφ, Vφ)
      solve!(φh⁺, solver, op_elec)

      println("Mechanical staggered step")
      op_mec = FEOperator(res_mec(time), jac_mec(time), Uu, Vu)
      solve!(uh⁺, solver, op_mec)
      
      driverpost(pvd, step, time)

      update_state!(cons_model, A, Fh, Fh⁻)
      TrialFESpace!(Uφ⁻, dirichlet_φ, time)
      TrialFESpace!(Uu⁻, dirichlet_u, time)
      φ⁻ .= get_free_dof_values(φh⁺)
      u⁻ .= get_free_dof_values(uh⁺)
    end
  end
  (t, uz)
end

# t, uz = staggered_visco_electric_simulation(; t_end=2, writevtk=true, verbose=true)
# using Plots
# p = plot(t, uz)
