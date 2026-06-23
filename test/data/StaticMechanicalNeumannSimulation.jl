using Gridap, GridapSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using TimerOutputs
using Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels.EvolutionFunctions


function static_mechanical_neumann_simulation(;writevtk=true, verbose=true)

  pname = "StaticMechanical"
  simdir = projdir("data", "sims", pname)
  setupfolder(simdir)

  long   = 0.05   # m
  width  = 0.005  # m
  thick  = 0.002  # m
  geometry = CartesianDiscreteModel((0, long, 0, width, 0, thick), (5,2,2))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "force", CartesianTags.face1YZ⁺)
  
  physmodel = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)

  # Setup integration
  order = 1
  degree = 2 * order + 1
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)

  # Dirichlet conditions 
  dir_u_tags = ["fixed"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [constant()]
  D_bc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  # Neumann conditions 
  neu_F_tags = ["force"]
  neu_F_values = [[0.0, 0.0, -1e-3]]
  neu_F_timesteps = [ramp()]
  N_bc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)
  dΓ = get_Neumann_dΓ(geometry, N_bc, degree)

  #  FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  V = TestFESpace(Ω, reffeu, D_bc, conformity=:H1)
  U = TrialFESpace(V, D_bc, 0.0)

  #  residual and jacobian function of load factor
  k = Kinematics(Mechano, Solid)
  res(Λ) = (u, v) -> residual(physmodel, k, u, v, dΩ) + residual_Neumann(N_bc, v, dΓ, Λ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, k, u, du, v, dΩ)

  ls = LUSolver()
  nls = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=verbose)

  comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls)

  function driverpost_mech(post)
    if writevtk
      state = post.comp_model.caches[3]
      Λ_ = post.iter
      Λ = post.Λ[Λ_]
      xh = FEFunction(U, state)
      pvd = post.cachevtk[3]
      filePath = post.cachevtk[2]
      if post.cachevtk[1]
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(Ω,
          filePath * "/" * pname * "_Λ_" * Λstring * ".vtu",
          cellfields=["u" => xh])
      end
    end
  end

  post_model = PostProcessor(comp_model, driverpost_mech; is_vtk=writevtk, filepath=simdir)

  @timeit pname begin
    x = solve!(comp_model; stepping=(nsteps=8, maxbisec=0), post=post_model)
  end
  return x
end


if abspath(PROGRAM_FILE) == @__FILE__
  reset_timer!()
  static_mechanical_neumann_simulation()
  print_timer()
end
