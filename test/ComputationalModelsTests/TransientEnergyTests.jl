# TransientModel: the energies — the incremental functional of the stage (its gradient is the stage
# residual, checked by finite differences for every implicit structured scheme), the Armijo line search
# on it, and the energies reported by the verbose output (kinetic, stored, work of the loads, balance).
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: TimeStep, initialize!, _build_stage!, _model_operator
using Gridap.Algebra: residual
using LinearAlgebra
using Random
using Test

# a dynamic Mooney–Rivlin plate: moving Dirichlet values, a Neumann traction ramp, the mass form
function dynamic_mooney_rivlin(; n=(6, 3), stretch=0.2, ρ=1.0)
  geometry = CartesianDiscreteModel((0, 2, 0, 1), n)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  add_tag_from_tags!(labels, "top", [6])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  km = Kinematics(Mechano, Solid)
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [2stretch, 0.3], EvolutionFunctions.ramp()))
  N_bc = BoundaryConditions(BoundaryCondition("top", [0.05, -0.3], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(t) = (u, v) -> residual(physmodel, km, u, v, dΩ)
  jac(t) = (u, du, v) -> jacobian(physmodel, km, u, du, v, dΩ)
  en(t) = u -> energy(physmodel, km, u, dΩ)
  mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ
  neumann = NeumannConditions(geometry, N_bc, 2)
  (; res, jac, en, mass, U, V, bc, dΩ, neumann)
end

# the heat equation: Π = ½ κ ∫ ∇u ⋅ ∇u, a moving Dirichlet value
function heat_energy_problem(; n=4, κ=0.3)
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [7]); add_tag_from_tags!(labels, "rest", [1, 2, 3, 4, 5, 6, 8])
  reffe = ReferenceFE(lagrangian, Float64, 1)
  bc = BoundaryConditions(BoundaryCondition("left", 1.0, EvolutionFunctions.ramp(1.0)),
                          BoundaryCondition("rest", 0.0, EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ
  jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  en(t) = u -> ∫(0.5 * κ * (∇(u) ⋅ ∇(u)))dΩ
  mass = (u, v) -> ∫(u * v)dΩ
  (; res, jac, en, mass, U, V, bc, dΩ, neumann=nothing)
end

# the standard output of `f()` as a string
function captured(f)
  path = tempname()
  open(path, "w") do io
    redirect_stdout(io) do
      f()
    end
  end
  read(path, String)
end

# central finite difference of Φ along d against R ⋅ d at y
function fd_consistency(mop, y, d; ε=1e-6)
  Φp = energy(mop, y .+ ε .* d)
  Φm = energy(mop, y .- ε .* d)
  (Φp - Φm) / 2ε, dot(residual(mop, y), d)
end

@testset "Stage energy: finite-difference consistency with the stage residual" begin
  rng = MersenneTwister(7)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, verbose=0)
  t_end = 0.4
  cases = ((dynamic_mooney_rivlin, (Newmark(), GeneralizedAlpha(0.8), HHT(0.1), Midpoint(), Newmark(β=0.3, γ=0.6))),
           (heat_energy_problem, (BDF(1), BDF(2), Theta(0.5), Theta(0.7), SDIRK(2))))
  for (problem, schemes) in cases, scheme in schemes
    p = problem()
    m = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme, mass=p.mass, nls, neumann=p.neumann, energy=p.en,
                       initial_acceleration=:solve)
    stepping = TimeStepping(; Δt=t_end / 8, t_end)
    st = TimeStep(stepping)
    initialize!(m, st)
    step!(m, st, stepping); step!(m, st, stepping)       # a history (BDF2 past the start-up, velocities)
    c = m.caches
    _build_stage!(m, st, 1, st.t, st.Δt, identity)
    mop = _model_operator(m.hooks, m.spaces, c.op, true)
    y = copy(c.y)
    y .= m.hist.xn .+ 0.01 .* randn(rng, length(y))
    for _ in 1:3
      d = randn(rng, length(y))
      fd, rd = fd_consistency(mop, y, d)
      @test abs(fd - rd) <= 1e-6 * max(abs(rd), 1e-8)
    end
  end
  # an explicit stage has no energy
  p = dynamic_mooney_rivlin()
  m = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme=CentralDifference(), mass=p.mass, nls, neumann=p.neumann, energy=p.en)
  st = TimeStep(TimeStepping(; Δt=1e-3, t_end))
  initialize!(m, st)
  _build_stage!(m, st, 1, st.t, st.Δt, identity)
  mop = _model_operator(m.hooks, m.spaces, m.caches.op, true)
  @test_throws ErrorException energy(mop, m.caches.y)
  # the general level reports the stored energy but has no stage energy without `stage_energy`
  ph = heat_energy_problem()
  dΩ = ph.dΩ
  resh(ctx) = (u, v) -> ∫(∂t(ctx, u) * v)dΩ + ph.res(ctx.t)(field(ctx, u), v)
  jach(ctx) = (u, du, v) -> ∫(δ∂t(ctx, du) * v)dΩ + ph.jac(ctx.t)(field(ctx, u), δfield(ctx, du), v)
  mg = @test_logs (:warn, r"stage_energy") TransientModel(resh, jach, ph.U, ph.V, ph.bc; scheme=BDF(1), nls, energy=ph.en)
  mop = _model_operator(mg.hooks, mg.spaces, mg.caches.op, false)
  @test_throws ErrorException energy(mop, mg.caches.y)
  @test_throws ErrorException TransientModel(ph.res, ph.jac, ph.U, ph.V, ph.bc; scheme=BDF(1), mass=ph.mass, nls, energy=ph.en,
                                             stage_energy=ctx -> (u -> ∫(u * u)dΩ))
end

@testset "Armijo line search on the stage energy" begin
  p = dynamic_mooney_rivlin()
  t_end = 0.5
  nls0 = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, verbose=0)
  m0 = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme=Newmark(), mass=p.mass, nls=nls0, neumann=p.neumann,
                      initial_acceleration=:solve)
  x0, flag0 = solve!(m0; stepping=TimeStepping(; Δt=t_end / 5, t_end))
  x0 = copy(x0)
  p = dynamic_mooney_rivlin()
  ls = InjectivityLS(armijo=true, c1=1e-4, maxiter=20)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=ls, verbose=0)
  m = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme=Newmark(), mass=p.mass, nls, neumann=p.neumann, energy=p.en,
                     injectivity=p.dΩ, initial_acceleration=:solve)
  x, flag = solve!(m; stepping=TimeStepping(; Δt=t_end / 5, t_end), ProjectDirichlet=true)
  @test flag0 == 1 && flag == 1 && converged(nls)
  @test all(0 .< step_history(nls) .<= 1)
  @test norm(x - x0) <= 1e-7 * norm(x0)
  # the model's energy is needed
  p = dynamic_mooney_rivlin()
  nls = NewtonRaphson(LUSolver(); maxiter=30, linesearch=InjectivityLS(armijo=true), verbose=0)
  m = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme=Newmark(), mass=p.mass, nls, neumann=p.neumann, injectivity=p.dΩ)
  @test_throws ErrorException solve!(m; stepping=TimeStepping(; Δt=t_end / 5, t_end))
end

@testset "Verbose output and the energy balance" begin
  p = dynamic_mooney_rivlin()
  t_end = 0.4
  # with a load ramp and moving Dirichlet values: the lines carry K, Π, K+Π and the balance; the balance
  # is not closed (the work of the Dirichlet reactions is not tracked)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, verbose=1)
  m = TransientModel(p.res, p.jac, p.U, p.V, p.bc; scheme=Newmark(), mass=p.mass, nls, neumann=p.neumann, energy=p.en,
                     initial_acceleration=:solve)
  text = captured(() -> solve!(m; stepping=TimeStepping(; Δt=t_end / 4, t_end)))
  lines = split(text, '\n')
  steplines = filter(l -> startswith(l, "step"), lines)
  @test length(steplines) == 4
  @test all(l -> occursin("K = ", l) && occursin("Π = ", l) && occursin("K+Π = ", l) && occursin("balance = ", l) && occursin("✓", l), steplines)
  @test any(l -> startswith(l, "t = 4.0000e-01 reached: 4 steps, 0 bisections, 0 rejections"), lines)
  @test any(l -> startswith(l, "energy balance"), lines)
  e = get_energies(m)
  @test isfinite(e.K) && isfinite(e.Π) && isfinite(e.W) && isfinite(e.balance)
  # free vibration of a linear bar with the trapezoidal rule: the discrete energy is conserved, the
  # balance is round-off; with a traction ramp the trapezoidal work closes it as well
  function linear_bar(; traction=false)
    model = CartesianDiscreteModel((0, 1, 0, 0.2), (6, 2))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left", [7]); add_tag_from_tags!(labels, "right", [8])
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    bc = BoundaryConditions(BoundaryCondition("left", [0.0, 0.0], EvolutionFunctions.constant()))
    V = TestFESpace(model, reffe, bc; conformity=:H1)
    U = TrialFESpace(V, bc, 0.0)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)
    λ, μ, ρ = 1.0, 0.5, 1.0
    ε(u) = 0.5 * (∇(u) + ∇(u)')
    σ(e) = λ * tr(e) * one(e) + 2μ * e
    res(t) = (u, v) -> ∫((σ ∘ ε(u)) ⊙ ε(v))dΩ
    jac(t) = (u, du, v) -> ∫((σ ∘ ε(du)) ⊙ ε(v))dΩ
    en(t) = u -> ∫(0.5 * ((σ ∘ ε(u)) ⊙ ε(u)))dΩ
    mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ
    v0 = interpolate(x -> VectorValue(0.1 * x[1], 0.0), U)
    N_bc = BoundaryConditions(BoundaryCondition("right", [0.02, 0.0], EvolutionFunctions.ramp()))
    neumann = traction ? NeumannConditions(model, N_bc, 2) : nothing
    (; res, jac, en, mass, U, V, bc, v0, neumann)
  end
  for traction in (false, true)
    b = linear_bar(; traction)
    nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-12, rtol=1e-12, verbose=1)
    m = TransientModel(b.res, b.jac, b.U, b.V, b.bc; scheme=Newmark(), mass=b.mass, nls, energy=b.en, neumann=b.neumann,
                       vh=b.v0, initial_acceleration=:solve)
    text = captured(() -> solve!(m; stepping=TimeStepping(; Δt=0.05, t_end=1.0)))
    e = get_energies(m)
    @test abs(e.balance) <= 1e-10 * (e.K + e.Π)
    traction || @test e.W == 0.0
    @test count(l -> startswith(l, "step"), split(text, '\n')) == 20
  end
  # not verbose: nothing tracked, nothing printed
  b = linear_bar()
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-12, rtol=1e-12, verbose=0)
  m = TransientModel(b.res, b.jac, b.U, b.V, b.bc; scheme=Newmark(), mass=b.mass, nls, energy=b.en, vh=b.v0)
  @test isempty(captured(() -> solve!(m; stepping=TimeStepping(; Δt=0.1, t_end=0.3))))
  @test isnan(get_energies(m).K)
end
