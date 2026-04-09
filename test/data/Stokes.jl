using Gridap, GridapSolvers
using GridapSolvers.LinearSolvers
using TimerOutputs
using Gridap.FESpaces
using HyperFEM
 

function Stokes(; writevtk=true, verbose=true)

  pname = "Stokes"
  simdir = projdir("data", "sims", pname)
  setupfolder(simdir)

  geomodel  = GmshDiscreteModel(projdir() * "/test/models/test_stokes.msh"; has_affine_map=false)


  # Setup integration
  orderu = 1
  orderp = 1

  Ω = Triangulation(geomodel)
  dΩ = Measure(Ω, 2 * orderu)

  # DirichletBC velocity
  evolu(Λ) = 1.0
  dir_u_tags = ["Dirichlet_Inlet", "Dirichlet_Wall", "Dirichlet_Circle"]
  dir_u_values = [x -> VectorValue([16 * 1.0 * (0.0625 - x[2]^2), 0.0]), [0.0, 0.0], [0.0, 0.0]]
  dir_u_timesteps = [evolu, evolu, evolu]
  Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  D_bc = MultiFieldBC([Du, NothingBC()])

  # Finite Elements
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, orderu)
  reffep = ReferenceFE(lagrangian, Float64, orderp)

  # Test FE Spaces
  Vu = TestFESpace(geomodel, reffeu, D_bc[1], conformity=:H1)
  Vp = TestFESpace(geomodel, reffep, D_bc[2], conformity=:H1)

  # Trial FE Spaces
  Uu = TrialFESpace(Vu, D_bc[1], 1.0)
  Up = TrialFESpace(Vp, D_bc[2], 1.0)

  # Multifield FE Spaces
  V = MultiFieldFESpace([Vu, Vp])
  U = MultiFieldFESpace([Uu, Up])

  I = TensorValue(1.0, 0.0, 0.0, 1.0)
  ε(∇u) = 0.5 * (∇u + ∇u')
  μ = 1.0
  a((u, p), (v, q)) = ∫((μ * (ε ∘ (∇(u))) - p * I) ⊙ (ε ∘ (∇(v))))dΩ - ∫(q * (I ⊙ (ε ∘ (∇(u)))))dΩ
  l((v, q)) = 0.0
  compmodel = StaticLinearModel(l, a, U, V, D_bc)

  xh = get_state(compmodel)
  # Postprocessor to save results
  function driverpost(post)
    if writevtk
      Λ_ = post.iter
      uh = xh[1]
      ph = xh[2]
      pvd = post.cachevtk[3]
      filePath = post.cachevtk[2]
      if post.cachevtk[1]
        pvd[Λ_] = createvtk(Ω, filePath * "/Stokes" * ".vtu", cellfields=["u" => uh, "p" => ph])
      end
    end
  end

  post_model = PostProcessor(compmodel, driverpost; is_vtk=true, filepath=simdir)

  @timeit pname begin
    solve!(compmodel; Assembly=false, post=post_model)
  end
  return get_free_dof_values(xh)
end


if abspath(PROGRAM_FILE) == @__FILE__
  reset_timer!()
  Stokes()
  print_timer()
end
