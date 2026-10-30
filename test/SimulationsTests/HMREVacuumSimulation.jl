# hMRE beam with explicit surrounding vacuum, 2D, staggered magneto-mechanics: local version of
# Adv_Mater_Ultra_Soft_Magnet src/hMRE_Vacuum_2D.jl (Perez-Garcia, Ortigosa, Martinez-Frutos, Garcia-Gonzalez,
# Advanced Materials) with the same weak forms, models, boundary conditions and solvers, written with
# BoundaryConditions and the Coupling prescription. Mesh test/models/Short_magnet.msh (25 375 dofs).
# The test is reduced: half of the applied potential and four staggered steps. The example of HyperFEMKernels
# (examples/hmre_vacuum_2d) builds the full problem with `hmre_vacuum_model`.
using Gridap, GridapGmsh
using Gridap.FESpaces
using Gridap.CellData
using HyperFEM
using LinearAlgebra

"""
    hmre_vacuum_model(backend; φbot=0.2 * 5200, verbose=false)

The three staggered problems of the hMRE + vacuum example built for `backend`: magnetic potential on
solid + vacuum, solid mechanics with the Maxwell traction of the vacuum on the interface, fictitious mechanics
of the vacuum driven by the solid through a `Coupling` prescription. `φbot` is the potential at the bottom of
the box (0.2 · 5200 in the original script). Returns the `StaggeredModel` (`comp_model`), the three
problems (weak forms, spaces and states) and the triangulations and states for post-processing.
"""
function hmre_vacuum_model(backend::Symbol; φbot=0.2 * 5200, verbose=false)
  geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "Short_magnet.msh"))

  # Constitutive models
  μ0 = 4π * 1e-7
  model_vacuum_mag = IdealMagnetic2D(μ0=μ0, χe=0.0)
  params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
  modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
              TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
  modelhard = HardMagnetic2D(μ0=μ0, αr=10e-3 / μ0, χe=0.0, χr=8.0)
  model_solid = MagnetoMechModel(magneto=modelhard, mechano=modelmech)
  model_vacuum_mech_ = NonlinearMooneyRivlin2D_CV(λ=1.0, μ1=1.0, μ2=0.0, α1=6.0, α2=1.0, γ=6.0)
  model_vacuum_mech = HessianRegularization(mechano=model_vacuum_mech_, δ=1e-6)

  # Finite elements
  order_solid = 1
  order_vacuum = 1
  reffeφ = ReferenceFE(lagrangian, Float64, order_solid)
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_solid)
  reffeu_vacuum = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_vacuum)

  # Domains
  Ωdomain = Triangulation(geomodel, tags=["vacuumb", "Beam"])
  Ωvacuum = Triangulation(geomodel, tags=["vacuumb"])
  Ωsolid = Triangulation(geomodel, tags=["Beam"])
  dΩsolid = Measure(Ωsolid, 2 * order_solid)
  dΩvacuum_mag = Measure(Ωvacuum, 2 * order_solid)
  dΩvacuum_mec = Measure(Ωvacuum, 2 * order_vacuum)
  Γair_int = BoundaryTriangulation(Ωvacuum, tags="Interface")
  Γsolid_int = BoundaryTriangulation(Ωsolid, tags="Interface")
  dΓs = Measure(Γsolid_int, 4 * order_solid)

  # Dirichlet boundary conditions
  evolφ(Λ) = Λ
  evolu(Λ) = 1.0
  Dφ = BoundaryConditions(BoundaryCondition("phibot", φbot, evolφ), BoundaryCondition("phitop", 0.0, evolφ))
  Du_solid = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]), BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))
  # Du_vacuum below: its coupling needs the interface spaces, built from the solid spaces

  # FE spaces and state variables ((⁺) current stage, (⁻) previous stage)
  Vφ = TestFESpace(Ωdomain, reffeφ, Dφ, conformity=:H1)
  Uφ⁺ = TrialFESpace(Vφ, Dφ); Uφ⁻ = TrialFESpace(Vφ, Dφ)
  φh⁺ = FEFunction(Uφ⁺, zero_free_values(Uφ⁺)); φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))
  Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1)
  Uu_solid⁺ = TrialFESpace(Vu_solid, Du_solid); Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid)
  uh_solid⁺ = FEFunction(Uu_solid⁺, zero_free_values(Uu_solid⁺)); uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))
  # Solid–vacuum coupling: staggered solid displacement interpolated to the vacuum interface
  Uair_int = FESpace(Γair_int, reffeu_vacuum)
  Usolid_int = FESpace(Γsolid_int, reffeu)
  uhsolid(Λ) = uh_solid⁻ + (uh_solid⁺ - uh_solid⁻) * Λ
  uhsolid_int = interpolate_everywhere(uh_solid⁺, Usolid_int)
  uhair_int = interpolate_everywhere(Interpolable(uhsolid_int), Uair_int)
  uhsolid_int_(Λ) = Interpolable(interpolate_everywhere!(uhsolid(Λ), get_free_dof_values(uhsolid_int), uhsolid_int.dirichlet_values, Usolid_int))
  uhair_int_(Λ) = interpolate_everywhere!(uhsolid_int_(Λ), get_free_dof_values(uhair_int), uhair_int.dirichlet_values, Uair_int)
  Du_vacuum = BoundaryConditions(BoundaryCondition("Fixed_Air", [0.0, 0.0]), BoundaryCondition("Fixed_Air_x", [0.0, 0.0]; mask=(true, false)),
                                 BoundaryCondition("Interface", Coupling(uhair_int_)))         # the state of the solid, a function of the stage parameter
  Vu_vacuum = TestFESpace(Ωvacuum, reffeu_vacuum, Du_vacuum, conformity=:H1)
  Uu_vacuum⁺ = TrialFESpace(Vu_vacuum, Du_vacuum); Uu_vacuum⁻ = TrialFESpace(Vu_vacuum, Du_vacuum)
  uh_vacuum⁺ = FEFunction(Uu_vacuum⁺, zero_free_values(Uu_vacuum⁺)); uh_vacuum⁻ = FEFunction(Uu_vacuum⁻, zero_free_values(Uu_vacuum⁻))

  # Maxwell stress in the vacuum and traction on the interface
  V_maxwell = TestFESpace(Ωvacuum, reffeu_vacuum)
  Pmaxw = FEFunction(V_maxwell, zero_free_values(V_maxwell))
  Pmaxw_ = get_free_dof_values(Pmaxw)
  V_traction = TestFESpace(Γsolid_int, reffeu)
  U_traction = TrialFESpace(V_traction)
  tmaxw = interpolate_everywhere(Pmaxw, Usolid_int)
  tmaxw_ = get_free_dof_values(tmaxw)
  Massoperator = StaticLinearModel((u, v) -> ∫(u ⋅ v)dΓs, U_traction, V_traction, NothingBC())

  # Weak forms
  Ψs, ∂Ψs∂F, ∂Ψs∂H0, ∂Ψs∂FF, ∂Ψs∂H0F, ∂Ψs∂H0H0 = model_solid()
  Ψv, ∂Ψv∂F, ∂Ψv∂H0, ∂Ψv∂FF, ∂Ψv∂H0F, ∂Ψv∂H0H0 = model_vacuum_mag()
  Ψvm, ∂Ψvm∂F, ∂Ψvm∂FF = model_vacuum_mech()
  F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
  ℋ₀ = get_Kinematics(Kinematics(Magneto, Solid))
  uhvacuum(Λ) = uh_vacuum⁻ + (uh_vacuum⁺ - uh_vacuum⁻) * Λ
  φh(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
  V_N = TestFESpace(Ωsolid, reffeu)
  Nh = interpolate_everywhere((x) -> VectorValue(1.0, 0.0), V_N)

  res_mag(Λ) = (φ, vφ) -> -1.0 * ∫((∇(vφ) ⋅ (∂Ψs∂H0 ∘ (F ∘ (∇(uhsolid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh))))dΩsolid -
                          ∫((∇(vφ) ⋅ (∂Ψv∂H0 ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φ))))))dΩvacuum_mag
  jac_mag(Λ) = (φ, dφ, vφ) -> ∫(∇(vφ)' ⋅ ((∂Ψs∂H0H0 ∘ (F ∘ (∇(uhsolid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩsolid +
                              ∫(∇(vφ)' ⋅ ((∂Ψv∂H0H0 ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φ)))) ⋅ ∇(dφ)))dΩvacuum_mag
  res_mech(Λ) = begin
    assemble_vector!((v) -> ∫(∇(v)' ⊙ (∂Ψv∂F ∘ (F ∘ (∇(uhvacuum(Λ))'), ℋ₀ ∘ (∇(φh(Λ))))))dΩvacuum_mec, Pmaxw_, V_maxwell)
    interpolate_everywhere!(Interpolable(Pmaxw), tmaxw_, tmaxw.dirichlet_values, Usolid_int)
    Massoperator(tmaxw_)
    (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(Λ))), Nh))))dΩsolid + ∫((v ⋅ tmaxw))dΓs
  end
  jac_mech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(Λ))), Nh)) ⊙ (∇(du)')))dΩsolid
  res_vacmech(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψvm∂F ∘ (F ∘ (∇(u)')))))dΩvacuum_mec
  jac_vacmech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψvm∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩvacuum_mec

  # Computational problems
  nls_mag = NewtonRaphson(LUSolver(); maxiter=20, atol=1.e-9, rtol=1.e-8, verbose=verbose)
  cm_mag = StaticNonlinearModel(res_mag, jac_mag, Uφ⁺, Vφ, Dφ; nls=nls_mag, xh=φh⁺, backend=backend)
  nls_mech = NewtonRaphson(LUSolver(); maxiter=20, atol=1.e-6, rtol=1.e-2, verbose=verbose)
  cm_mech = StaticNonlinearModel(res_mech, jac_mech, Uu_solid⁺, Vu_solid, Du_solid; nls=nls_mech, xh=uh_solid⁺, backend=backend)
  nls_vacmech = NewtonRaphson(LUSolver(); maxiter=10, rtol=2, verbose=verbose)
  cm_vacmech = StaticNonlinearModel(res_vacmech, jac_vacmech, Uu_vacuum⁺, Vu_vacuum, Du_vacuum; nls=nls_vacmech, xh=uh_vacuum⁺, backend=backend)
  comp_model = StaggeredModel((cm_mag, cm_mech, cm_vacmech), (φh⁺, uh_solid⁺, uh_vacuum⁺), (φh⁻, uh_solid⁻, uh_vacuum⁻))

  problems = ((name="magnetic", res=res_mag, jac=jac_mag, U=Uφ⁺, V=Vφ, xh=φh⁺),
              (name="solid mechanics", res=res_mech, jac=jac_mech, U=Uu_solid⁺, V=Vu_solid, xh=uh_solid⁺),
              (name="vacuum mechanics", res=res_vacmech, jac=jac_vacmech, U=Uu_vacuum⁺, V=Vu_vacuum, xh=uh_vacuum⁺))
  (; comp_model, problems, Ωsolid, Ωvacuum, φh⁺, uh_solid⁺, uh_vacuum⁺)
end


function hmre_vacuum_simulation(; backend=:lazy, φbot=0.2 * 5200 / 2, nsteps=4, nsubsteps=1, verbose=false)
  s = hmre_vacuum_model(backend; φbot, verbose)
  args = (Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true),
          Dict(:stepping => (nsteps=1, maxbisec=5)),
          Dict(:stepping => (nsteps=1, maxbisec=5), :ProjectDirichlet => true))
  solve!(s.comp_model; stepping=(nsteps=nsteps, nsubsteps=nsubsteps, maxbisec=1), kargsolve=args, verbose=verbose)
  (; φ=copy(get_free_dof_values(s.φh⁺)), u_solid=copy(get_free_dof_values(s.uh_solid⁺)),
     u_vacuum=copy(get_free_dof_values(s.uh_vacuum⁺)))
end
