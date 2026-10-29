# Third-medium contact: constrained buckling of a double-clamped beam pushed by a rigid piston inside a box
# (Faltus, Amato, Horák 2026, Sec. 3.3, "piston" benchmark). Test version of Piston_clean.jl with the same weak
# forms, models and solver, written with BoundaryConditions. Mesh test/models/Piston_1.msh (P1 triangles).
#
#   Solid           : NonlinearMooneyRivlin2D (μ1 = μ2 = μ, λ = 2μ)
#   Third medium    : ARAP2D with stiffness fc·μ
#   Regularisation  : jump of ∇u on the interior facets of the third medium, σβ/h_F ∫ mean(w) ⟦∇u⟧:⟦∇v⟧, with the
#                     weight w(F) = wvol e^{-kvol J} + wdist e^{kdist (F:F - 2J)} - (wvol + wdist - 1)
#                     (Bluhm, Sigmund, Poulios 2021). Residual w·q(u,v); consistent tangent
#                     w·q(du,v) + (∂w/∂F : ∇du)·q(u,v) (not symmetric). P1 only: the Hessian terms of the original
#                     script (volume and jump of ∇∇u) vanish for P1 and are not included.
#   Loading         : vertical displacement of the piston, uy_max in nsteps increments
#   Solver          : Newton_RaphsonSolver with Injectivity_Preserving_LS
#
# The test is reduced: uy_max = 0.15 (1.5d) in 15 increments, past the buckling of the beam at u_y ≈ 1.05d (the
# full benchmark goes to 15d in 1000 increments, `thirdmedium_contact_simulation(uy_max=1.5, nsteps=1000)`).
# Reactions in the piston and in the bottom clamp are returned normalised by the double-clamped Euler load of the
# linearised solid (as in the original script; the plateau comes out at ≈ 7 f_cr with that normalisation).
# With the kernel backend the skeleton term is assembled by facet kernels (HyperFEMKernels skeleton integrals).
using Gridap, GridapGmsh, GridapSolvers
using Gridap.FESpaces
using Gridap.CellData
using HyperFEM
using LinearAlgebra


function thirdmedium_contact_simulation(; backend=:lazy, uy_max=0.15, nsteps=15, maxbisec=5, verbose=false,
                                          writevtk=false, vtkevery=10)
  geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "Piston_1.msh"))

  # Parameters
  order = 1
  μ     = 1e5                 # solid: μ1 = μ2 = μ, λ = 2μ
  fc    = 1e-6                # third-medium stiffness relative to μ
  β     = 1e-5 * μ            # regularisation parameter
  σ     = 1.0                 # jump of ∇u: σβ/h_F
  wvol, wdist = 1.0, 1.0      # weight coefficients (wvol = wdist = 0 -> w ≡ 1)
  kvol, kdist = 5.0, 1.0      # weight exponents

  # Constitutive models
  model_solid   = NonlinearMooneyRivlin2D(λ=2μ, μ1=μ, μ2=μ, α1=1.0, α2=1.0)
  model_contact = ARAP2D(μ=μ * fc)
  Ψs, ∂Ψs∂F, ∂Ψs∂FF = model_solid()
  Ψc, ∂Ψc∂F, ∂Ψc∂FF = model_contact()
  F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))

  # Discretisation
  reffe     = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  Ωdomain   = Triangulation(geomodel, tags=["domain", "third"])
  Ωsolid    = Triangulation(geomodel, tags=["domain"])
  Ωcontact  = Triangulation(geomodel, tags=["third"])
  dΩdomain  = Measure(Ωdomain, 2order)
  dΩsolid   = Measure(Ωsolid, 2order)
  dΩcontact = Measure(Ωcontact, 2order)
  Γsk       = SkeletonTriangulation(geomodel, tags=["third"])     # interior facets of the third medium
  dΓsk      = Measure(Γsk, 2order)
  hF        = CellField(get_array(∫(1)dΓsk), Γsk)                # facet length
  cJ        = (σ * β) / hF

  # Boundary conditions: piston (rigid top moving down), beam clamped at both ends, third medium sliding on the box
  uimp = [0.0, -uy_max]
  D_bc = BoundaryConditions(
    BoundaryCondition("Fixed_XY", [0.0, 0.0]),
    BoundaryCondition("Fixed_Y", [0.0, 0.0]; mask=(false, true)),
    BoundaryCondition("Fixed_X", [0.0, 0.0]; mask=(true, false)),
    BoundaryCondition("Imposed_Y_free_X", uimp, EvolutionFunctions.ramp(); mask=(false, true)),
    BoundaryCondition("Imposed_Y_fixed_X", uimp, EvolutionFunctions.ramp()))
  Vu = TestFESpace(Ωdomain, reffe, D_bc, conformity=:H1)
  Uu = TrialFESpace(Vu, D_bc, 0.0)
  uh = FEFunction(Uu, zero_free_values(Uu))

  # Weight w(F) and its derivative
  cof(A) = TensorValue(A[2, 2], -A[1, 2], -A[2, 1], A[1, 1])        # ∂J/∂F in 2D
  w(A)   = wvol * exp(-kvol * det(A)) + wdist * exp(kdist * (A ⊙ A - 2det(A))) - (wvol + wdist - 1)
  dw(A)  = -kvol * wvol * exp(-kvol * det(A)) * cof(A) + 2kdist * wdist * exp(kdist * (A ⊙ A - 2det(A))) * (A - cof(A))
  Fu(u)  = F ∘ (∇(u)')
  qJ(a, b) = jump(∇(a)) ⊙ jump(∇(b))                               # jump of ∇u (skeleton)
  wres(u, v)     = mean(w ∘ Fu(u)) * qJ(u, v)
  wjac(u, du, v) = mean(w ∘ Fu(u)) * qJ(du, v) + mean((dw ∘ Fu(u)) ⊙ ∇(du)') * qJ(u, v)

  # Weak forms
  res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψs∂F ∘ Fu(u)))dΩsolid +
                     ∫(∇(v)' ⊙ (∂Ψc∂F ∘ Fu(u)))dΩcontact +
                     ∫(cJ * wres(u, v))dΓsk
  jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ Fu(u)) ⊙ ∇(du)'))dΩsolid +
                         ∫(∇(v)' ⊙ ((∂Ψc∂FF ∘ Fu(u)) ⊙ ∇(du)'))dΩcontact +
                         ∫(cJ * wjac(u, du, v))dΓsk

  # Reactions in piston and clamp (vertical component), normalised by the double-clamped Euler load
  function reaction_test(rtags)
    Vr = TestFESpace(Ωdomain, reffe, conformity=:H1, dirichlet_tags=rtags, dirichlet_masks=[(false, true) for _ in rtags])
    FEFunction(TrialFESpace(Vr, [VectorValue(0.0, 1.0) for _ in rtags]), zero_free_values(Vr))
  end
  φpiston = reaction_test(["Imposed_Y_fixed_X", "Imposed_Y_free_X"])
  φclamp  = reaction_test(["Fixed_XY"])
  d, Lb   = 0.1, 1.8
  μl, λl  = 2μ, 4μ                                                  # linearisation of the Mooney-Rivlin solid
  E       = μl * (3λl + 2μl) / (λl + μl)
  fcr     = π^2 * E * (d^3 / 12) / (0.5Lb)^2

  simdir = projdir("data", "sims", "ThirdMediumContact")
  writevtk && setupfolder(simdir)
  Λs, fpiston, fclamp = Float64[], Float64[], Float64[]
  function driverpost(post)
    Λ = post.Λ[post.iter]
    push!(Λs, Λ)
    push!(fpiston, -sum(res(Λ)(uh, φpiston)) / fcr)                 # compression > 0
    push!(fclamp, sum(res(Λ)(uh, φclamp)) / fcr)
    if writevtk && post.iter % vtkevery == 0
      Gridap.writevtk(Ωsolid, joinpath(simdir, "solid_$(post.iter).vtu"), cellfields=["uh" => uh])
      Gridap.writevtk(Ωcontact, joinpath(simdir, "thirdmedium_$(post.iter).vtu"), cellfields=["uh" => uh])
    end
  end

  # Solver
  α_state    = CellState(1.0, dΩdomain)
  linesearch = Injectivity_Preserving_LS(α_state, Uu, Vu; maxiter=20, αmin=1e-16, ρ=0.5, c=0.95, β=1 - 1e-6)
  nls        = Newton_RaphsonSolver(LUSolver(); maxiter=20, atol=1e-8, rtol=1e-4, ctol=1e-10, verbose=verbose, linesearch=linesearch)
  comp_model = StaticNonlinearModel(res, jac, Uu, Vu, D_bc; nls=nls, xh=uh, backend=backend)
  post_model = PostProcessor(comp_model, driverpost; is_vtk=false, filepath=simdir)
  solve!(comp_model; stepping=(nsteps=nsteps, maxbisec=maxbisec), post=post_model)

  (; u=copy(get_free_dof_values(uh)), Λ=Λs, uy_over_d=uy_max .* Λs ./ d, fpiston, fclamp)
end


if abspath(PROGRAM_FILE) == @__FILE__
  s = thirdmedium_contact_simulation(verbose=true)
  println("u_y/d = ", s.uy_over_d[end], "  f_piston/f_cr = ", s.fpiston[end], "  f_clamp/f_cr = ", s.fclamp[end])
end
