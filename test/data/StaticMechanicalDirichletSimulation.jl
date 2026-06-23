using Gridap, GridapSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using TimerOutputs
using Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels.EvolutionFunctions


function static_mechanical_dirichlet_simulation(;writevtk=true, verbose=true)

  pname = "Stretch"
  simdir = projdir("data", "sims", pname)
  setupfolder(simdir)

  long   = 0.05   # m
  width  = 0.005  # m
  thick  = 0.002  # m
  geometry = CartesianDiscreteModel((0, long, 0, width, 0, thick), (5,2,2))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "moving", CartesianTags.face1YZ⁺)

  physmodel = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=0.0, ρ=1.0)

  # Setup integration
  order = 1
  degree = 2 * order
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)

  # Dirichlet boundary conditions
  dir_u_tags = ["fixed", "moving"]
  dir_u_values = [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]
  dir_u_timesteps = [constant(), ramp()]
  D_bc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  #  FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

  V = TestFESpace(Ω, reffeu, D_bc, conformity=:H1)
  U = TrialFESpace(V, D_bc, 0.0)

  #  residual and jacobian function of load factor
  k = Kinematics(Mechano, Solid)
  res(Λ) = (u, v) -> residual(physmodel, k, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, k, u, du, v, dΩ)

  #computational model
  ls = LUSolver()
  nls = NewtonSolver(ls; maxiter=15, rtol=1.e-12, verbose=verbose)

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
          filePath *  "/" * pname * "_Λ_" * Λstring * ".vtu",
          cellfields=["u" => xh])
      end
    end
  end

  post_model = PostProcessor(comp_model, driverpost_mech; is_vtk=writevtk, filepath=simdir)

  @timeit pname begin
    x, flag = solve!(comp_model; stepping=(nsteps=10, maxbisec=10), post=post_model,ProjectDirichlet=true)
  end
  return x
end


if abspath(PROGRAM_FILE) == @__FILE__
  reset_timer!()
  static_mechanical_dirichlet_simulation()
  print_timer()
end
