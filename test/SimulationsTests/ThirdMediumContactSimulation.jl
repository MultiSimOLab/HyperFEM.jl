# Third-medium contact: constrained buckling of a double-clamped beam pushed by a rigid piston inside a box
# (Faltus, Amato, Horák 2026, Sec. 3.3, "piston" benchmark). Test version of Piston_clean.jl with the same weak
# forms, models and solver, written with BoundaryConditions. Mesh test/models/Piston_1.msh (P1 triangles).
#
#   Solid           : NonlinearMooneyRivlin2D (μ1 = μ2 = μ, λ = λsolid·μ; 2μ by default)
#   Third medium    : ARAP2D with stiffness fc·μ, plus the barrier κ/(2p)(ln J)^{2p} (LogBarrier, κ = kb·μ·fc)
#                     when kb > 0
#   Regularisation  : jump of ∇u on the interior facets of the third medium, σβ/h_F ∫ mean(w) ⟦∇u⟧:⟦∇v⟧, with the
#                     weight w(F) = wvol e^{-kvol J} + wdist e^{kdist (F:F - 2J)} - (wvol + wdist - 1)
#                     (Bluhm, Sigmund, Poulios 2021). Residual w·q(u,v); consistent tangent
#                     w·q(du,v) + (∂w/∂F : ∇du)·q(u,v) (not symmetric). With weight = false, w ≡ 1: the residual
#                     derives from an energy (Armijo). P1 only: the Hessian terms of the original script (volume
#                     and jump of ∇∇u) vanish for P1 and are not included.
#   Loading         : vertical displacement of the piston, uy_max in nsteps increments
#   Solver          : NewtonRaphson with InjectivityLS (exact injectivity step, backtracking on |R·dx| or Armijo
#                     on the energy); with hybrid = true the hybrid Newton (HybridTangent) with the projected
#                     tangents of both media (HessianRegularization sharing one TangentSwitch)
#
# The test is reduced: uy_max = 0.15 (1.5d) in 15 increments, past the buckling of the beam at u_y ≈ 1.05d (the
# full benchmark goes to 15d in 1000 increments, `thirdmedium_contact_simulation(uy_max=1.5, nsteps=1000)`).
# Reactions in the piston and in the bottom clamp are returned normalised by the double-clamped Euler load of the
# linearised solid (as in the original script; the plateau comes out at ≈ 7 f_cr with that normalisation) and the
# plane-strain one (`fcr_plane_strain`, E' = E/(1 - ν²), Horák's normalisation) is returned as well.
# With the kernel backend the skeleton term is assembled by facet kernels (HyperFEMKernels skeleton integrals).
#
# The configuration of the note "Solver improvements for third-medium contact" (2026-10-25) is
# `weight = false, kb = 1.0, pvol = 2, λsolid = 6.0, hybrid = true, armijo = true, jmin = 1e-6, maxiter = 500`
# (local long tests in runtests.jl, HYPERFEM_TEST_LONG = 1: to 3.5 d with a fixed increment, and the full
# benchmark to 15 d with the growth of the increment, `grow = 1.5`, against Horák's curve
# test/models/horak2026_piston_f_vs_uy.json).
using Gridap, GridapGmsh
using Gridap.FESpaces
using Gridap.CellData
using HyperFEM
using LinearAlgebra


function thirdmedium_contact_simulation(; backend=:lazy, uy_max=0.15, nsteps=15, maxbisec=5, grow=1.0, easy=10, verbose=false,
                                          writevtk=false, vtkevery=10, ProjectDirichlet=false,
                                          weight=true, kb=0.0, pvol=2, λsolid=2.0,
                                          hybrid=false, armijo=false, jmin=0.0, hrs=1e-6, hybback=1e-2, atolproj=1e-3,
                                          maxiter=20, atol=1e-8, rtol=1e-4)
  geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "Piston_1.msh"))

  # Parameters
  order = 1
  μ     = 1e5                 # solid: μ1 = μ2 = μ, λ = λsolid μ
  fc    = 1e-6                # third-medium stiffness relative to μ
  β     = 1e-5 * μ            # regularisation parameter
  σ     = 1.0                 # jump of ∇u: σβ/h_F
  wvol, wdist = weight ? (1.0, 1.0) : (0.0, 0.0)   # weight coefficients (wvol = wdist = 0 -> w ≡ 1 exactly)
  kvol, kdist = 5.0, 1.0      # weight exponents
  armijo && weight && error("thirdmedium_contact_simulation: the Armijo line search needs an energy, i.e. w ≡ 1 (weight = false)")

  # Constitutive models (with the shared switch of the hybrid Newton: consistent tangents while it is off)
  switch = hybrid ? TangentSwitch() : nothing
  wrap(model, δ) = hybrid ? HessianRegularization(model; δ=δ, switch=switch) : model
  model_solid_   = NonlinearMooneyRivlin2D(λ=λsolid * μ, μ1=μ, μ2=μ, α1=1.0, α2=1.0)
  model_contact_ = kb > 0 ? ARAP2D(μ=μ * fc) + LogBarrier(κ=kb * μ * fc, p=pvol) : ARAP2D(μ=μ * fc)
  model_solid    = wrap(model_solid_, hrs * μ)
  model_contact  = wrap(model_contact_, hrs * μ * fc)
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
  # Energy whose first variation is the residual (w ≡ 1): solid, third medium (with the barrier), regularisation
  en(Λ) = u -> ∫(Ψs ∘ Fu(u))dΩsolid + ∫(Ψc ∘ Fu(u))dΩcontact + ∫(0.5 * cJ * wres(u, u))dΓsk

  # Reactions in piston and clamp (vertical component), normalised by the double-clamped Euler load
  function reaction_test(rtags)
    Vr = TestFESpace(Ωdomain, reffe, conformity=:H1, dirichlet_tags=rtags, dirichlet_masks=[(false, true) for _ in rtags])
    FEFunction(TrialFESpace(Vr, [VectorValue(0.0, 1.0) for _ in rtags]), zero_free_values(Vr))
  end
  φpiston = reaction_test(["Imposed_Y_fixed_X", "Imposed_Y_free_X"])
  φclamp  = reaction_test(["Fixed_XY"])
  d, Lb   = 0.1, 1.8
  μl, λl  = 2μ, λsolid * μ + 2μ                                     # linearisation of the Mooney-Rivlin solid
  E       = μl * (3λl + 2μl) / (λl + μl)
  Eps     = 4μl * (λl + μl) / (λl + 2μl)                            # plane strain, E/(1 - ν²)
  fcr     = π^2 * E * (d^3 / 12) / (0.5Lb)^2
  fcr_ps  = π^2 * Eps * (d^3 / 12) / (0.5Lb)^2

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

  # Solver: injectivity-preserving line search with the exact root of det(F + α ∇δu) over the quadrature
  # points of dΩdomain (solid and third medium), then backtracking on |R·dx| (or Armijo on the energy)
  linesearch = InjectivityLS(maxiter=20, αmin=1e-16, ρ=0.5, c=0.95, β=1 - 1e-6, jmin=jmin, armijo=armijo)
  tangent    = hybrid ? HybridTangent(hybback=hybback, atol_projected=atolproj * fcr_ps) : ConsistentTangent()
  nls        = NewtonRaphson(LUSolver(); maxiter=maxiter, atol=atol, rtol=rtol, verbose=verbose, linesearch=linesearch, tangent=tangent)
  comp_model = StaticNonlinearModel(res, jac, Uu, Vu, D_bc; nls=nls, xh=uh, backend=backend, injectivity=dΩdomain,
                                    energy=armijo ? en : nothing, tangent_switch=switch)
  post_model = PostProcessor(comp_model, driverpost; is_vtk=false, filepath=simdir)
  stepping = LoadStepping(; nsteps, maxbisec, grow, easy)
  nbisect = Ref(0)
  x, flag = solve!(comp_model; stepping, ProjectDirichlet=ProjectDirichlet, post=post_model, on_bisection=(Λ, ΔΛ) -> nbisect[] += 1)

  (; u=copy(get_free_dof_values(uh)), Λ=Λs, uy_over_d=uy_max .* Λs ./ d, fpiston, fclamp, fcr, fcr_plane_strain=fcr_ps,
     flag, nls, nbisect=nbisect[])
end


if abspath(PROGRAM_FILE) == @__FILE__
  s = thirdmedium_contact_simulation(verbose=true)
  println("u_y/d = ", s.uy_over_d[end], "  f_piston/f_cr = ", s.fpiston[end], "  f_clamp/f_cr = ", s.fclamp[end])
end
