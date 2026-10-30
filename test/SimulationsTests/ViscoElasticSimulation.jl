using Gridap
using Gridap.FESpaces
using HyperFEM
using HyperFEM.DiscreteModeling.EvolutionFunctions
using HyperFEM.ComputationalModels.PostMetrics

"""
    visco_elastic_simulation(; backend=:lazy, driver=:transient, t_end=2, writevtk=false, verbose=false, nls=nothing, stepping=nothing)

Uniaxial loading–unloading of a generalized Maxwell bar (three viscous branches), the internal variables
in `CellState`s (`backend = :compiled` stores them as `KernelState`s, updated by the engine). `driver`
selects the time loop: `:transient` — `TransientModel` with the `QuasiStatic` scheme, the viscous
constitutive update through the `physics`/`internal` hooks and the state of the previous step through
`ctx.u⁻`; `:static` — the load-step loop of `StaticNonlinearModel` with the load factor as time and the
state update in the post-processor (the historical setup). `nls` replaces the Newton solver and `stepping`
the time stepping of the transient driver (tests force failed attempts with them). Returns the axial
stretch and the mean first Piola stress on the moving face per step.
"""
function visco_elastic_simulation(; backend=:lazy, driver=:transient, t_end=2, writevtk=false, verbose=false, nls=nothing, stepping=nothing)
  simdir = projdir("data", "sims", stem(@__FILE__))
  writevtk && setupfolder(simdir)

  # Domain and tessellation
  long   = 0.05   # m
  width  = 0.005  # m
  thick  = 0.001  # m
  domain = (0.0, long, 0.0, width, 0.0, thick)
  partition = (3, 1, 1)
  model = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "corner1", CartesianTags.corner000)
  add_tag_from_tags!(labels, "corner2", CartesianTags.corner010)
  add_tag_from_tags!(labels, "fixed",  CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "moving", CartesianTags.face1YZ⁺)

  # Constitutive model
  μ = 1.37e4   # Pa
  λ = 100μ     # Pa
  μ₁ = 5.64e4  # Pa
  τ₁ = 0.82    # s
  μ₂ = 3.15e4  # Pa
  τ₂ = 10.7    # s
  μ₃ = 1.98e4  # Pa
  τ₃ = 500.0   # s
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
  viscous_branch_1 = ViscousPolyconvex(μ=μ₁, τ=τ₁)
  viscous_branch_2 = ViscousPolyconvex(μ=μ₂, τ=τ₂)
  viscous_branch_3 = ViscousPolyconvex(μ=μ₃, τ=τ₃)
  cons_model = GeneralizedMaxwell(hyper_elastic_model, viscous_branch_1, viscous_branch_2, viscous_branch_3)
  k=Kinematics(Mechano,Solid)

  # Dirichlet boundary conditions: the stretch peaks at t = 10 s (the static driver evaluates the evolution
  # laws at the load factor Λ = t / t_end, the transient one at the time)
  strain = 0.5
  law = driver === :transient ? triangular(10.0) : triangular(10/t_end)
  D_bc = BoundaryConditions(
    BoundaryCondition("corner1", [0.0, 0.0, 0.0], constant(); mask=(true, true, true)),
    BoundaryCondition("corner2", [0.0, 0.0, 0.0], constant(); mask=(true, false, true)),
    BoundaryCondition("fixed", [0.0, 0.0, 0.0], constant(); mask=(true, false, false)),
    BoundaryCondition("moving", [long * strain, 0.0, 0.0], law; mask=(true, false, false)))

  # Setup integration
  order = 1
  degree = 2 * order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dΓ = get_Neumann_dΓ(model, NothingBC(), degree)
  Γ1  = BoundaryTriangulation(model, tags=get_tags(D_bc)[4])
  dΓ1 = Measure(Γ1, degree)
  Δt = 0.05
  update_time_step!(cons_model, Δt)

  # FE spaces
  reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  Vu = TestFESpace(model, reffe, D_bc, conformity=:H1)
  VL2 = FESpace(Ω, reffe, conformity=:L2)
  Uu = TrialFESpace(Vu, D_bc, 0.0)

  uh = FEFunction(Uu, zero_free_values(Uu))
  state_vars = CellState(cons_model, dΩ; backend)      # :compiled → KernelStates, updated by the engine
  F,_,_ = get_Kinematics(k)

  ls = LUSolver()
  nls === nothing && (nls = NewtonRaphson(ls; maxiter=20, atol=1.e-6, rtol=1.e-6, verbose=verbose))

  λx = Float64[]
  σΓ = Float64[]

  # the stress of the step from the state of the previous step and the internal variables the residual was
  # evaluated with (the post-processing runs before the update of the internal variables)
  function stress!(uh, unh)
    _, ∂Ψu, _ = cons_model()
    Ph = interpolate_L2_field(∂Ψu ∘ (F∘∇(uh)', F∘∇(unh)', state_vars...), dΩ)
    σh11 = VectorValue(1.0, 0.0, 0.0) ⋅ Ph ⋅ VectorValue(1.0, 0.0, 0.0)
    σΓ1 = sum(∫(σh11)dΓ1) / sum(∫(1.0)dΓ1)
    push!(σΓ, σΓ1)
    push!(λx, 1.0 + component_LInf(uh, :x, Ω) / long)
  end

  if driver === :transient
    # general level of the stage contract: the field of the stage, the time and the state of the previous
    # step come from the context (QuasiStatic: field(ctx, u) = u, ctx.t = tₙ₊₁)
    res_t(ctx) = (u,v)->residual(cons_model, k, field(ctx, u), v, dΩ, ctx.t, F∘∇(ctx.u⁻)', state_vars...)
    jac_t(ctx) = (u,du,v)->jacobian(cons_model, k, field(ctx, u), δfield(ctx, du), v, dΩ, ctx.t, F∘∇(ctx.u⁻)', state_vars...)
    internal(uh, unh) = update_state!(cons_model, state_vars, F∘(∇(uh)'), F∘(∇(unh)'); backend)
    comp_model = TransientModel(res_t, jac_t, Uu, Vu, D_bc; scheme=QuasiStatic(), nls, xh=uh, physics=cons_model, internal, backend)
    post_model = PostProcessor(comp_model, post -> stress!(uh, get_previous_state(comp_model)); is_vtk=writevtk, filepath=simdir)
    stepping === nothing && (stepping = TimeStepping(; Δt, t_end=Float64(t_end), maxbisec=1))
    solve!(comp_model; stepping, post=post_model, ProjectDirichlet=true)
  else
    Uun = TrialFESpace(Vu, D_bc, 0.0)
    unh = FEFunction(Uun, zero_free_values(Uun))
    Fnh = F∘∇(unh)'
    res_s(Λ) = (u,v)->residual(cons_model, k, u, v, dΩ, t_end * Λ, Fnh, state_vars...)
    jac_s(Λ) = (u,du,v)->jacobian(cons_model, k, u, du, v, dΩ, t_end * Λ, Fnh, state_vars...)
    comp_model = StaticNonlinearModel(res_s, jac_s, Uu, Vu, D_bc; nls=nls, xh=uh, xh⁻=unh, backend=backend)
    function driverpost(post)
      stress!(uh, unh)
      update_state!(cons_model, state_vars, F∘(∇(uh)'), F∘(∇(unh)'); backend)
    end
    post_model = PostProcessor(comp_model, driverpost; is_vtk=writevtk, filepath=simdir)
    solve!(comp_model; stepping=(nsteps=Int(t_end/Δt), maxbisec=1), post=post_model, ProjectDirichlet=true)
  end
  (λx, σΓ)
end


if abspath(PROGRAM_FILE) == @__FILE__
  using Plots
  @time λx, σΓ = visco_elastic_simulation(t_end=15, writevtk=true, verbose=true)
  plot(λx, σΓ)
end
