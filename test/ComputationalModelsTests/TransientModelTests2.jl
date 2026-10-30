# TransientModel: the time loop itself — moving Dirichlet values (and their inertia), the restore after a
# failed solve and after a rejected step, the Dirichlet projection as predictor, exact output times, the
# allocations of the stage loop.
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: TimeStep, initialize!, exhausted, _build_stage!
using Gridap.Algebra: residual_and_jacobian!
using LinearAlgebra
using Logging
using Test

# heat with a time-dependent Dirichlet value on the left side, u = g(t) sin(πy) ... the exact semi-discrete
# problem is the same for both levels, so the structured level (Dirichlet inertia as a load vector) must give
# the general level (the contract carries the Dirichlet velocity through the FE functions) to round-off
function heat_dirichlet_problem(; n=4, order=1)
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [7]); add_tag_from_tags!(labels, "rest", [1, 2, 3, 4, 5, 6, 8])
  reffe = ReferenceFE(lagrangian, Float64, order)
  bc = BoundaryConditions(BoundaryCondition("left", 1.0, EvolutionFunctions.ramp(1.0)),
                          BoundaryCondition("rest", 0.0, EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
  κ = 0.3
  (; U, V, dΩ, κ, bc)
end

@testset "TransientModel: moving Dirichlet values" begin
  p = heat_dirichlet_problem()
  (; U, V, dΩ, κ, bc) = p
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-8, verbose=0)
  t_end = 0.4
  for scheme in (BDF(1), BDF(2), Theta(0.5))
    res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ
    jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
    mass = (u, v) -> ∫(u * v)dΩ
    ms = TransientModel(res, jac, U, V, bc; scheme, mass, nls)
    xs, flag = solve!(ms; stepping=TimeStepping(; Δt=t_end / 8, t_end))
    @test flag == 1
    @test ms.caches.op.use_bd                     # the Dirichlet inertia vector is in use
    resg(ctx) = (u, v) -> ∫(∂t(ctx, u) * v)dΩ + ∫(κ * (∇(field(ctx, u)) ⋅ ∇(v)))dΩ
    jacg(ctx) = (u, du, v) -> ∫(δ∂t(ctx, du) * v)dΩ + ∫(κ * (∇(δfield(ctx, du)) ⋅ ∇(v)))dΩ
    mg = TransientModel(resg, jacg, U, V, bc; scheme, nls)
    xg, flag = solve!(mg; stepping=TimeStepping(; Δt=t_end / 8, t_end))
    @test flag == 1
    @test norm(copy(xs) - xg) <= 1e-10 * norm(xg)
    # the Dirichlet values of the trial space are those of t_end after the run, the history those of the last step
    @test maximum(get_dirichlet_dof_values(U)) ≈ min(t_end, 1.0) atol = 1e-12
  end
end

# A solver that fails at chosen attempts (as LoadSteppingTests' FailAt)
mutable struct FailAtT{N<:NewtonRaphson,L} <: Gridap.Algebra.NonlinearSolver
  nls::N
  fails::Vector{Int}
  calls::Int
  ls::L
  linesearch::Any
  log::Any
end
FailAtT(nls, fails) = FailAtT(nls, fails, 0, nls.ls, nls.linesearch, nls.log)
HyperFEM.ComputationalModels.instantiate_caches(x, s::FailAtT, op) = HyperFEM.ComputationalModels.instantiate_caches(x, s.nls, op)
function Gridap.Algebra.solve!(x::AbstractVector, s::FailAtT, op::Gridap.Algebra.NonlinearOperator, cache; r_reference=0.0)
  s.calls += 1
  s.calls in s.fails && return cache
  solve!(x, s.nls, op, cache; r_reference)
end
HyperFEM.ComputationalModels.nls_converged(s::FailAtT) = !(s.calls in s.fails) && HyperFEM.Solvers.converged(s.nls)
HyperFEM.ComputationalModels.nls_message(s::FailAtT, msg) = nothing
HyperFEM.ComputationalModels.nls_verbosity(s::FailAtT) = 0
HyperFEM.ComputationalModels.nls_reset!(s::FailAtT) = HyperFEM.ComputationalModels.nls_reset!(s.nls)
HyperFEM.ComputationalModels.nls_iterations(s::FailAtT) = HyperFEM.ComputationalModels.nls_iterations(s.nls)

function bar_dirichlet_problem(; n=(4, 2), order=1, law=EvolutionFunctions.ramp(1.0))
  model = CartesianDiscreteModel((0, 1, 0, 0.2), n)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [7]); add_tag_from_tags!(labels, "right", [8])
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  bc = BoundaryConditions(BoundaryCondition("left", [0.0, 0.0], EvolutionFunctions.constant()),
                          BoundaryCondition("right", [0.02, 0.0], law; mask=(true, false)))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
  E, ν, ρ = 1.0, 0.3, 1.0
  λ = E * ν / ((1 + ν) * (1 - 2ν)); μ = E / (2(1 + ν))
  ε(u) = 0.5 * (∇(u) + ∇(u)')
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  res(t) = (u, v) -> ∫((σ ∘ ε(u)) ⊙ ε(v))dΩ
  jac(t) = (u, du, v) -> ∫((σ ∘ ε(du)) ⊙ ε(v))dΩ
  mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ
  (; U, V, dΩ, bc, res, jac, mass)
end

@testset "TransientModel: restore after a failed solve and after a rejection" begin
  p = bar_dirichlet_problem()
  (; U, V, dΩ, bc, res, jac, mass) = p
  nls() = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  t_end = 0.5
  # a run without failures
  m0 = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls())
  x0, flag = solve!(m0; stepping=TimeStepping(; Δt=t_end / 5, t_end))
  @test flag == 1
  # the same run with the third attempt failing: the state after the failure is the one of step 2, the run
  # continues with the halved increment and the trajectory is a legitimate one (different steps)
  snap = Dict{Symbol,Any}()
  # the internal-variable hook: called once per accepted step (never on a failed attempt), after the
  # post-processing, with the new state and the state of the previous step
  calls = Ref(0); prev = Ref(0.0); seen_by_post = Ref(0)
  internal = (uh, unh) -> (calls[] += 1; prev[] = norm(get_free_dof_values(unh)); nothing)
  m = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=FailAtT(nls(), [3]), internal)
  post = PostProcessor(m, _ -> (seen_by_post[] = calls[]); is_vtk=false, filepath=mktempdir())
  st = TimeStep(TimeStepping(; Δt=t_end / 5, t_end))
  stepping = TimeStepping(; Δt=t_end / 5, t_end, maxbisec=3)
  initialize!(m, st)
  ok1 = step!(m, st, stepping; post)
  x1 = norm(m.caches.x)
  ok2 = step!(m, st, stepping; post)
  @test ok1 && ok2
  @test calls[] == 2 && prev[] == x1 && seen_by_post[] == 1
  snap[:x] = copy(m.caches.x); snap[:v] = copy(m.hist.vn); snap[:a] = copy(m.hist.an); snap[:t] = st.t
  snap[:gd] = copy(get_dirichlet_dof_values(U)); snap[:gdn] = deepcopy(m.hist.gdn); snap[:vdn] = deepcopy(m.hist.vdn)
  ok = @test_logs (:warn, "Bisection performed!") step!(m, st, stepping; post)
  @test !ok && calls[] == 2
  @test m.caches.x == snap[:x] && m.hist.vn == snap[:v] && m.hist.an == snap[:a] && st.t == snap[:t]
  @test get_dirichlet_dof_values(U) == snap[:gd] && m.hist.gdn == snap[:gdn] && m.hist.vdn == snap[:vdn]
  @test st.Δt == t_end / 10 && st.nbisect == 1
  ok3 = step!(m, st, stepping)
  @test ok3
  @test st.t ≈ 2t_end / 5 + t_end / 10
  # a rejection by the error estimate restores the same things; a smooth start (the ramp has a velocity
  # jump at t = 0 whose Zienkiewicz–Xie estimate does not decrease with Δt: every attempt would be rejected)
  p = bar_dirichlet_problem(; law=EvolutionFunctions.smoothstep(0.5))
  (; U, V, dΩ, bc, res, jac, mass) = p
  m = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls())
  stepping = AdaptiveStepping(; Δt=t_end / 5, t_end, rtol=1e-2, atol=1e-12)
  st = TimeStep(stepping)
  initialize!(m, st)
  n = 0
  while !step!(m, st, stepping) && !exhausted(st, stepping) && (n += 1) < 10 end
  @test st.nsteps == 1 && st.nreject == 0
  snap[:x] = copy(m.caches.x); snap[:v] = copy(m.hist.vn); snap[:a] = copy(m.hist.an); snap[:t] = st.t
  snap[:gd] = copy(get_dirichlet_dof_values(U)); snap[:gdn] = deepcopy(m.hist.gdn)
  # force a rejection: an absurd increment
  st.Δt = 50.0
  rejected = Ref(false)
  ok = step!(m, st, stepping; on_reject=(t, Δt, err) -> (rejected[] = true))
  @test !ok && rejected[] && st.nreject == 1
  @test m.caches.x == snap[:x] && m.hist.vn == snap[:v] && m.hist.an == snap[:a] && st.t == snap[:t]
  @test get_dirichlet_dof_values(U) == snap[:gd] && m.hist.gdn == snap[:gdn]
  @test st.Δt < 50.0
end

@testset "TransientModel: Dirichlet projection as predictor, tstops, allocations" begin
  p = bar_dirichlet_problem()
  (; U, V, dΩ, bc, res, jac, mass) = p
  nls() = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  t_end = 0.5
  # moving Dirichlet values: the projection gives the same solution (the problem is linear: it is exact up
  # to the inertia, which Newton corrects)
  m1 = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls(), injectivity=dΩ)
  x1, flag1 = solve!(m1; stepping=TimeStepping(; Δt=t_end / 5, t_end))
  m2 = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls(), injectivity=dΩ)
  x2, flag2 = solve!(m2; stepping=TimeStepping(; Δt=t_end / 5, t_end), ProjectDirichlet=true)
  @test flag1 == 1 && flag2 == 1
  @test norm(copy(x1) - x2) <= 1e-8 * norm(x1)
  # exact output times
  ts = Float64[]
  m3 = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls())
  post = PostProcessor(m3, post -> push!(ts, post.Λ[post.iter]); is_vtk=false, filepath=mktempdir())
  solve!(m3; stepping=TimeStepping(; Δt=0.3, t_end=1.0, tstops=[0.5, 0.75]), post)
  @test 0.5 in ts && 0.75 in ts && ts[end] == 1.0
  # allocations of a stage (the forms built, the residual and the tangent assembled): the capture of the
  # stage forms by the compiled backend (the same per-step cost as the static drivers), the FE functions of
  # the stage, and the backend's materialisation of the auxiliary FE functions per cell (a few hundred bytes
  # per cell) — nothing proportional to the number of dofs in the driver itself (no vector or matrix copies).
  # The linear solve is left out (Gridap's LU allocates a factorisation per solve) and so is the lazy backend
  # (Gridap's cell-wise assembly allocates per cell).
  if :compiled in TEST_BACKENDS_T
    function stage_allocations(n)
      p = bar_dirichlet_problem(; n)
      (; U, V, bc, res, jac, mass) = p
      m = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nls(), backend=:compiled)
      stepping = TimeStepping(; Δt=t_end / 5, t_end)
      st = TimeStep(stepping)
      initialize!(m, st)
      step!(m, st, stepping); step!(m, st, stepping)
      c = m.caches
      b, A, y = c.nls_cache.b, c.nls_cache.A, c.y
      tn, Δt = st.t, st.Δt
      _build_stage!(m, st, 1, tn, Δt, identity); residual_and_jacobian!(b, A, c.op, y)
      @allocated begin
        _build_stage!(m, st, 1, tn, Δt, identity)
        residual_and_jacobian!(b, A, c.op, y)
      end
    end
    a_coarse = stage_allocations((4, 2))
    a_fine = stage_allocations((40, 20))
    @test a_fine <= a_coarse + 400 * 800         # per-cell materialisation only; a copy of the tangent is ≈ 250 kB
  end
end

@testset "TransientModel: the context of a field of a multi-field problem" begin
  # two copies of the heat problem as the two fields of a multi-field space: `ctx[i]` gives each field its
  # β functions and history; the solution of each field is the single-field one
  p = heat_dirichlet_problem()
  (; U, V, dΩ, κ, bc) = p
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-10, verbose=0)
  t_end = 0.4
  res1(ctx) = (u, v) -> ∫(∂t(ctx, u) * v)dΩ + ∫(κ * (∇(field(ctx, u)) ⋅ ∇(v)))dΩ
  jac1(ctx) = (u, du, v) -> ∫(δ∂t(ctx, du) * v)dΩ + ∫(κ * (∇(δfield(ctx, du)) ⋅ ∇(v)))dΩ
  m1 = TransientModel(res1, jac1, U, V, bc; scheme=BDF(2), nls)
  x1, flag1 = solve!(m1; stepping=TimeStepping(; Δt=t_end / 8, t_end))
  x1 = copy(x1)
  V2 = TestFESpace(get_background_model(get_triangulation(U)), ReferenceFE(lagrangian, Float64, 1), bc; conformity=:H1)
  U2 = TrialFESpace(V2, bc, 0.0)
  UU = MultiFieldFESpace([U, U2]); VV = MultiFieldFESpace([V, V2])
  bc2 = MultiFieldBC([bc, bc])
  res2(ctx) = ((u, w), (v, q)) -> ∫(∂t(ctx[1], u) * v)dΩ + ∫(κ * (∇(field(ctx[1], u)) ⋅ ∇(v)))dΩ +
                                   ∫(2.0 * ∂t(ctx[2], w) * q)dΩ + ∫(2.0 * κ * (∇(field(ctx[2], w)) ⋅ ∇(q)))dΩ
  jac2(ctx) = ((u, w), (du, dw), (v, q)) -> ∫(δ∂t(ctx[1], du) * v)dΩ + ∫(κ * (∇(δfield(ctx[1], du)) ⋅ ∇(v)))dΩ +
                                             ∫(2.0 * δ∂t(ctx[2], dw) * q)dΩ + ∫(2.0 * κ * (∇(δfield(ctx[2], dw)) ⋅ ∇(q)))dΩ
  for backend in TEST_BACKENDS_T
    m2 = TransientModel(res2, jac2, UU, VV, bc2; scheme=BDF(2), nls, backend)
    x2, flag2 = solve!(m2; stepping=TimeStepping(; Δt=t_end / 8, t_end))
    @test flag1 == 1 && flag2 == 1
    xh = get_state(m2)
    @test norm(get_free_dof_values(xh[1]) - x1) <= 1e-10 * norm(x1)
    @test norm(get_free_dof_values(xh[2]) - x1) <= 1e-10 * norm(x1)
    # the history of a field: ctx[i].u⁻ is the field i of uₙ
    ctx = HyperFEM.ComputationalModels.TimeContext(0.0, 1.0, 1, 1, m2.caches.op.c, m2.hist, nothing)
    @test ctx[2].u⁻ === m2.hist.un[2] && ctx[1].u⁻ === m2.hist.un[1] && ctx.u⁻ === m2.hist.un
  end
end

@testset "TransientModel: inertia mask of a multi-field mass form" begin
  # the mass form touches the first field only: the error estimate and the initial acceleration are
  # restricted to its dofs (the compiled backend stores the whole pattern of the space with zeros)
  p = bar_dirichlet_problem()
  (; U, V, dΩ, bc, res, jac, mass) = p
  model = get_background_model(get_triangulation(U))
  bcp = BoundaryConditions(BoundaryCondition("left", 0.0, EvolutionFunctions.constant()))
  Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), bcp; conformity=:H1)
  Up = TrialFESpace(Vp, bcp, 0.0)
  UU = MultiFieldFESpace([U, Up]); VV = MultiFieldFESpace([V, Vp])
  bc2 = MultiFieldBC([bc, bcp])
  res2(t) = ((u, w), (v, q)) -> res(t)(u, v) + ∫(∇(w) ⋅ ∇(q))dΩ
  jac2(t) = ((u, w), (du, dw), (v, q)) -> jac(t)(u, du, v) + ∫(∇(dw) ⋅ ∇(q))dΩ
  mass2 = ((a, aw), (v, q)) -> mass(a, v)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  for backend in TEST_BACKENDS_T
    m = TransientModel(res2, jac2, UU, VV, bc2; scheme=Newmark(), mass=mass2, nls, backend, initial_acceleration=:solve)
    mask = HyperFEM.ComputationalModels._inertia_mask(m)
    nu = num_free_dofs(U)
    @test all(mask[1:nu]) && !any(mask[nu+1:end])
    x, flag = solve!(m; stepping=TimeStepping(; Δt=0.1, t_end=0.3))
    @test flag == 1
  end
end

@testset "TransientModel: moving Dirichlet values on every scheme and both levels" begin
  # the Dirichlet kinematics of every scheme (the algorithmic derivatives of the prescribed values, the
  # Dirichlet values of the stage unknown and of the stage field) on both levels: general = structured to
  # round-off; the explicit scheme against the trapezoidal rule at a fine step (order 2 with a moving boundary)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-10, verbose=0)
  # first order: heat with a Dirichlet ramp
  p = heat_dirichlet_problem()
  (; U, V, dΩ, κ, bc) = p
  t_end = 0.4
  res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ
  jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  mass = (u, v) -> ∫(u * v)dΩ
  resg(ctx) = (u, v) -> ∫(∂t(ctx, u) * v)dΩ + ∫(κ * (∇(field(ctx, u)) ⋅ ∇(v)))dΩ
  jacg(ctx) = (u, du, v) -> ∫(δ∂t(ctx, du) * v)dΩ + ∫(κ * (∇(δfield(ctx, du)) ⋅ ∇(v)))dΩ
  xfine, ffine = solve!(TransientModel(res, jac, U, V, bc; scheme=BDF(2), mass, nls); stepping=TimeStepping(; Δt=t_end / 512, t_end))
  @test ffine == 1
  xfine = copy(xfine)
  for scheme in (BDF(1), BDF(2), Theta(1.0), Theta(0.5), Theta(0.7), SDIRK(2)), backend in TEST_BACKENDS_T
    xs, fs = solve!(TransientModel(res, jac, U, V, bc; scheme, mass, nls, backend); stepping=TimeStepping(; Δt=t_end / 8, t_end))
    xs = copy(xs)
    xg, fg = solve!(TransientModel(resg, jacg, U, V, bc; scheme, nls, backend); stepping=TimeStepping(; Δt=t_end / 8, t_end))
    @test fs == 1 && fg == 1
    @test norm(xs - xg) <= 1e-9 * norm(xg)
    @test norm(xs - xfine) <= 0.2 * norm(xfine)         # a sane solution (a missing Dirichlet inertia is an O(1) error)
  end
  # second order: the bar with a smooth Dirichlet ramp (the Newton tolerances of the analytical tests: at
  # the fine step of the reference the residual of the linear problem is at the round-off of the inertia)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  b = bar_dirichlet_problem(; law=EvolutionFunctions.smoothstep(0.5))
  (; U, V, dΩ, bc, res, jac, mass) = b
  ρ = 1.0
  resb(ctx) = (u, v) -> ∫(ρ * (∂tt(ctx, u) ⋅ v))dΩ + res(ctx.t)(field(ctx, u), v)
  jacb(ctx) = (u, du, v) -> ∫(ρ * (δ∂tt(ctx, du) ⋅ v))dΩ + jac(ctx.t)(field(ctx, u), δfield(ctx, du), v)
  t_end = 1.0
  xref, fref = solve!(TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls, initial_acceleration=:solve); stepping=TimeStepping(; Δt=t_end / 2048, t_end))
  @test fref == 1
  xref = copy(xref)
  for scheme in (Newmark(), GeneralizedAlpha(0.8), HHT(0.1), Midpoint()), backend in TEST_BACKENDS_T
    xs, fs = solve!(TransientModel(res, jac, U, V, bc; scheme, mass, nls, backend); stepping=TimeStepping(; Δt=t_end / 16, t_end))
    xs = copy(xs)
    xg, fg = solve!(TransientModel(resb, jacb, U, V, bc; scheme, nls, backend); stepping=TimeStepping(; Δt=t_end / 16, t_end))
    @test fs == 1 && fg == 1
    @test norm(xs - xg) <= 1e-9 * norm(xg)
    @test norm(xs - xref) <= 0.2 * norm(xref)
  end
  # central differences with the moving boundary: second order towards the fine trapezoidal solution
  errs = Float64[]
  Δts = t_end ./ [128, 256, 512]
  for Δt in Δts
    xc, fc = solve!(TransientModel(res, jac, U, V, bc; scheme=CentralDifference(), mass, nls, initial_acceleration=:solve); stepping=TimeStepping(; Δt, t_end))
    @test fc == 1
    push!(errs, norm(xc - xref))
  end
  @test errs[end] <= 1e-3 * norm(xref)
  @test abs(log(errs[1] / errs[end]) / log(Δts[1] / Δts[end]) - 2) < 0.4
end

@testset "TransientModel: Dirichlet projection with the linear tangent policies; a₀ with damping" begin
  # the projection factorises the Newton cache with the static tangent: the :Δt policy must re-assemble
  # the stage tangent afterwards (the same solution as :iteration with projection)
  p = heat_dirichlet_problem()
  (; U, V, dΩ, κ, bc) = p
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-10, verbose=0)
  t_end = 0.4
  res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ
  jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  mass = (u, v) -> ∫(u * v)dΩ
  run(policy; project) = copy(solve!(TransientModel(res, jac, U, V, bc; scheme=BDF(1), mass, nls, tangent_update=policy);
                                     stepping=TimeStepping(; Δt=t_end / 8, t_end), ProjectDirichlet=project)[1])
  x0 = run(:iteration; project=false)
  for policy in (:iteration, :step, :Δt)
    @test norm(run(policy; project=true) - x0) <= 1e-9 * norm(x0)
  end
  # the projection is skipped for a Runge–Kutta stage (the stage unknown is a derivative)
  xk = copy(solve!(TransientModel(res, jac, U, V, bc; scheme=SDIRK(2), mass, nls); stepping=TimeStepping(; Δt=t_end / 8, t_end), ProjectDirichlet=true)[1])
  xk0 = copy(solve!(TransientModel(res, jac, U, V, bc; scheme=SDIRK(2), mass, nls); stepping=TimeStepping(; Δt=t_end / 8, t_end))[1])
  @test xk == xk0
  # initial acceleration with damping: M a₀ = −r(u₀) − C v₀
  b = bar_dirichlet_problem()
  (; U, V, dΩ, bc, res, jac, mass) = b
  damping = (v, w) -> ∫(0.3 * (v ⋅ w))dΩ
  v0 = interpolate(x -> VectorValue(0.1 * x[1], 0.0), U)
  nlsb = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  m = TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, damping, nls=nlsb, vh=v0, initial_acceleration=:solve)
  st = TimeStep(TimeStepping(; Δt=0.1, t_end=0.3))
  initialize!(m, st)
  M = m.caches.op.M
  r = assemble_vector(v -> res(0.0)(FEFunction(U, m.caches.x), v), V)
  Cv = assemble_vector(v -> damping(v0, v), V)           # C v₀ with the Dirichlet values of v₀
  a0 = M \ (-r - Cv)                                     # the Dirichlet ramp has g̈₀ = 0
  @test norm(m.hist.an - a0) <= 1e-10 * norm(a0)
  # the start of the explicit scheme on an accelerating boundary: a₀ with the Dirichlet inertia keeps the
  # second order (a₀ = 0 costs one order: the velocity error of the first step persists)
  bs = bar_dirichlet_problem(; law=EvolutionFunctions.smoothstep(0.5))
  (; U, V, dΩ, bc, res, jac, mass) = bs
  t_end = 1.0
  xref, _ = solve!(TransientModel(res, jac, U, V, bc; scheme=Newmark(), mass, nls=nlsb, initial_acceleration=:solve); stepping=TimeStepping(; Δt=t_end / 4096, t_end))
  xref = copy(xref)
  Δts = t_end ./ [128, 256, 512]
  errs = [norm(copy(solve!(TransientModel(res, jac, U, V, bc; scheme=CentralDifference(), mass, nls=nlsb, initial_acceleration=:solve);
                           stepping=TimeStepping(; Δt, t_end))[1]) - xref) for Δt in Δts]
  @test abs(log(errs[1] / errs[end]) / log(Δts[1] / Δts[end]) - 2) < 0.3
end
