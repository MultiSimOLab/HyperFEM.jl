# TransientModel: the stage contract on analytical problems. Orders in Δt against a fine reference run of
# the same semi-discrete problem (the spatial error cancels), the general level against the structured one,
# both backends, energy of the non-dissipative schemes, the adaptive control.
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: TimeContext, field, ∂t, ∂tt, at, δfield, δ∂t, δ∂tt, δat
using LinearAlgebra
using SparseArrays
using Logging
using Test

# both backends, as the simulation tests (HYPERFEM_TEST_BACKENDS, default "lazy,compiled")
const TEST_BACKENDS_T = isdefined(Main, :TEST_BACKENDS) ? TEST_BACKENDS :
  Tuple(Symbol.(strip.(split(get(ENV, "HYPERFEM_TEST_BACKENDS", "lazy,compiled"), ","))))
:compiled in TEST_BACKENDS_T && @eval using HyperFEMKernels

# observed order from errors e(Δt) on halved steps
function observed_order(Δts, errs)
  p = [log(errs[i] / errs[i+1]) / log(Δts[i] / Δts[i+1]) for i in 1:length(errs)-1]
  p[end]
end

# ---------------------------------------------------------------------------------------------------
# First order: heat equation ∂t u = κ Δu + f on the unit square, u = 0 on the boundary, u₀ = sin(πx) sin(πy)
# ---------------------------------------------------------------------------------------------------
function heat_problem(; n=4, order=1)
  model = CartesianDiscreteModel((0, 1, 0, 1), (n, n))
  reffe = ReferenceFE(lagrangian, Float64, order)
  bc = BoundaryConditions(BoundaryCondition("boundary", 0.0, EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
  κ = 0.1
  f(t) = x -> 2π^2 * κ * sin(π * x[1]) * sin(π * x[2]) * exp(-t)
  u0 = interpolate(x -> sin(π * x[1]) * sin(π * x[2]), U)
  (; model, U, V, dΩ, κ, f, u0, bc)
end
function heat_structured(p; scheme, stepping, backend=:lazy, tangent_update=:iteration, kwargs...)
  (; U, V, dΩ, κ, f, u0, bc) = p
  res(t) = (u, v) -> ∫(κ * (∇(u) ⋅ ∇(v)))dΩ - ∫(f(t) * v)dΩ
  jac(t) = (u, du, v) -> ∫(κ * (∇(du) ⋅ ∇(v)))dΩ
  mass = (u, v) -> ∫(u * v)dΩ
  xh = FEFunction(U, copy(get_free_dof_values(u0)))
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-8, verbose=0)
  m = TransientModel(res, jac, U, V, bc; scheme, mass, xh, nls, backend, tangent_update, kwargs...)
  x, flag = solve!(m; stepping)
  copy(x), flag, m
end
function heat_general(p; scheme, stepping, backend=:lazy, tangent_update=:iteration)
  (; U, V, dΩ, κ, f, u0, bc) = p
  res(ctx) = (u, v) -> ∫(∂t(ctx, u) * v)dΩ + ∫(κ * (∇(field(ctx, u)) ⋅ ∇(v)))dΩ - ∫(f(ctx.t) * v)dΩ
  jac(ctx) = (u, du, v) -> ∫(δ∂t(ctx, du) * v)dΩ + ∫(κ * (∇(δfield(ctx, du)) ⋅ ∇(v)))dΩ
  xh = FEFunction(U, copy(get_free_dof_values(u0)))
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-8, verbose=0)
  m = TransientModel(res, jac, U, V, bc; scheme, xh, nls, backend, tangent_update)
  x, flag = solve!(m; stepping)
  copy(x), flag, m
end

@testset "TransientModel: first order (heat)" begin
  p = heat_problem()
  t_end = 0.5
  xref, flag = heat_structured(p; scheme=Theta(0.5), stepping=TimeStepping(; Δt=t_end / 2048, t_end))
  @test flag == 1
  Δts = t_end ./ [8, 16, 32, 64]
  for (scheme, pexp, tol) in ((BDF(1), 1, 0.15), (BDF(2), 2, 0.2), (Theta(1.0), 1, 0.15), (Theta(0.5), 2, 0.15), (SDIRK(2), 2, 0.25))
    errs = [norm(heat_structured(p; scheme, stepping=TimeStepping(; Δt, t_end))[1] - xref) for Δt in Δts]
    @test abs(observed_order(Δts, errs) - pexp) < tol
  end
  # the general level (the contract in the forms) gives the structured result to round-off
  for scheme in (BDF(1), BDF(2), Theta(0.5), SDIRK(2))
    xs, _ = heat_structured(p; scheme, stepping=TimeStepping(; Δt=t_end / 16, t_end))
    xg, _ = heat_general(p; scheme, stepping=TimeStepping(; Δt=t_end / 16, t_end))
    @test norm(xs - xg) <= 1e-12 * norm(xs)
  end
  # tangent policies: one linear solve per step gives the Newton result (the problem is linear)
  xi, _ = heat_structured(p; scheme=BDF(2), stepping=TimeStepping(; Δt=t_end / 16, t_end))
  for tu in (:step, :Δt)
    xl, _ = heat_structured(p; scheme=BDF(2), stepping=TimeStepping(; Δt=t_end / 16, t_end), tangent_update=tu)
    @test norm(xl - xi) <= 1e-10 * norm(xi)
    xl, _ = heat_general(p; scheme=BDF(2), stepping=TimeStepping(; Δt=t_end / 16, t_end), tangent_update=tu)
    @test norm(xl - xi) <= 1e-10 * norm(xi)
  end
  # BDF2 with alternating steps keeps the order (tstops force the cut every other step)
  errs = Float64[]
  for n in (8, 16, 32)
    Δt = t_end / n
    stops = [k * Δt + 0.3Δt for k in 0:n-1]
    push!(errs, norm(heat_structured(p; scheme=BDF(2), stepping=TimeStepping(; Δt, t_end, tstops=stops))[1] - xref))
  end
  @test abs(observed_order(t_end ./ [8, 16, 32], errs) - 2) < 0.3
  # backends
  for b in TEST_BACKENDS_T
    b === :lazy && continue
    xs, _ = heat_structured(p; scheme=BDF(2), stepping=TimeStepping(; Δt=t_end / 16, t_end), backend=b)
    xg, _ = heat_general(p; scheme=Theta(0.5), stepping=TimeStepping(; Δt=t_end / 16, t_end), backend=b)
    xsl, _ = heat_structured(p; scheme=BDF(2), stepping=TimeStepping(; Δt=t_end / 16, t_end))
    xgl, _ = heat_general(p; scheme=Theta(0.5), stepping=TimeStepping(; Δt=t_end / 16, t_end))
    @test norm(xs - xsl) <= 1e-11 * norm(xsl)
    @test norm(xg - xgl) <= 1e-11 * norm(xgl)
  end
end

# ---------------------------------------------------------------------------------------------------
# Second order: linear elastic bar, fixed at x = 0, free vibration from an initial stretch (plane strain)
# ---------------------------------------------------------------------------------------------------
function bar_problem(; n=(6, 2), order=1)
  model = CartesianDiscreteModel((0, 1, 0, 0.2), n)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", [7])
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  bc = BoundaryConditions(BoundaryCondition("left", [0.0, 0.0], EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
  E, ν, ρ = 1.0, 0.3, 1.0
  λ = E * ν / ((1 + ν) * (1 - 2ν)); μ = E / (2(1 + ν))
  ε(u) = 0.5 * (∇(u) + ∇(u)')
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  u0 = interpolate(x -> VectorValue(0.01 * x[1], 0.0), U)
  (; model, U, V, dΩ, ε, σ, ρ, u0, bc)
end
function bar_run(p; scheme, stepping, backend=:lazy, vh=nothing, initial_acceleration=:solve, tangent_update=:step, kwargs...)
  (; U, V, dΩ, ε, σ, ρ, u0, bc) = p
  res(t) = (u, v) -> ∫((σ ∘ ε(u)) ⊙ ε(v))dΩ
  jac(t) = (u, du, v) -> ∫((σ ∘ ε(du)) ⊙ ε(v))dΩ
  mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ
  xh = FEFunction(U, copy(get_free_dof_values(u0)))
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=0)
  m = TransientModel(res, jac, U, V, bc; scheme, mass, xh, vh, nls, backend, initial_acceleration, tangent_update, kwargs...)
  x, flag = solve!(m; stepping)
  copy(x), flag, m
end

@testset "TransientModel: second order (bar)" begin
  p = bar_problem()
  t_end = 2.0
  xref, flag, mref = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 8192, t_end))
  @test flag == 1
  # orders in the asymptotic regime of the highest mode (ω_max Δt ≲ 1)
  Δts = t_end ./ [64, 128, 256, 512]
  for (scheme, tol) in ((Newmark(), 0.15), (GeneralizedAlpha(0.8), 0.25), (HHT(0.1), 0.25), (Midpoint(), 0.15))
    errs = [norm(bar_run(p; scheme, stepping=TimeStepping(; Δt, t_end))[1] - xref) for Δt in Δts]
    @test abs(observed_order(Δts, errs) - 2) < tol
  end
  # explicit central differences below the stability limit: consistent mass against the reference, lumped
  # mass (another semi-discrete problem) against its own finer run; the lumped matrix is diagonal
  Δts_e = t_end ./ [512, 1024, 2048]
  errs = [norm(bar_run(p; scheme=CentralDifference(), stepping=TimeStepping(; Δt, t_end))[1] - xref) for Δt in Δts_e]
  @test errs[end] < 1e-3 * norm(xref)
  @test abs(observed_order(Δts_e, errs) - 2) < 0.3
  xl1, _, ml = bar_run(p; scheme=CentralDifference(), stepping=TimeStepping(; Δt=t_end / 2048, t_end), mass_lumping=true)
  xl2, _ = bar_run(p; scheme=CentralDifference(), stepping=TimeStepping(; Δt=t_end / 8192, t_end), mass_lumping=true)
  @test nnz(ml.caches.op.M) == length(xl1)
  @test norm(xl1 - xl2) < 1e-3 * norm(xl2)
  @test norm(xl1 - xref) > 1e-3 * norm(xref)       # the lumping error is the larger one
  # energy of the trapezoidal rule and the midpoint rule: K + Π constant to round-off (linear problem)
  (; U, V, dΩ, ε, σ) = p
  K = assemble_matrix((du, v) -> ∫((σ ∘ ε(du)) ⊙ ε(v))dΩ, U, V)
  for scheme in (Newmark(), Midpoint())
    Es = Float64[]
    x, _, m = bar_run(p; scheme, stepping=TimeStepping(; Δt=t_end / 32, t_end))
    M = m.caches.op.M
    v = get_free_dof_values(HyperFEM.ComputationalModels.get_velocity(m))
    E1 = 0.5 * dot(v, M * v) + 0.5 * dot(x, K * x)
    x0 = get_free_dof_values(p.u0)
    E0 = 0.5 * dot(x0, K * x0)
    @test abs(E1 - E0) <= 1e-10 * E0
  end
  # generalized-α dissipates
  x, _, m = bar_run(p; scheme=GeneralizedAlpha(0.5), stepping=TimeStepping(; Δt=t_end / 32, t_end))
  M = m.caches.op.M
  v = get_free_dof_values(HyperFEM.ComputationalModels.get_velocity(m))
  x0 = get_free_dof_values(p.u0)
  @test 0.5 * dot(v, M * v) + 0.5 * dot(x, K * x) < 0.5 * dot(x0, K * x0)
  # the Newton path (consistent tangent at every iteration) gives the one-solve result (linear problem)
  xn_, _ = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 32, t_end), tangent_update=:iteration)
  xs_, _ = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 32, t_end))
  @test norm(xn_ - xs_) <= 1e-8 * norm(xs_)
  # a zero initial acceleration (inconsistent with the initial stretch) costs accuracy
  x1, _ = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 64, t_end))
  x2, _ = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 64, t_end), initial_acceleration=:zero)
  @test x1 != x2 && norm(x1 - xref) < norm(x2 - xref)
  # backends
  for b in TEST_BACKENDS_T
    b === :lazy && continue
    xb, _ = bar_run(p; scheme=GeneralizedAlpha(0.9), stepping=TimeStepping(; Δt=t_end / 32, t_end), backend=b)
    xl, _ = bar_run(p; scheme=GeneralizedAlpha(0.9), stepping=TimeStepping(; Δt=t_end / 32, t_end))
    @test norm(xb - xl) <= 1e-11 * norm(xl)
  end
end

@testset "TransientModel: adaptive stepping" begin
  p = bar_problem()
  t_end = 2.0
  xref, _ = bar_run(p; scheme=Newmark(), stepping=TimeStepping(; Δt=t_end / 4096, t_end))
  errs = Float64[]
  for rtol in (1e-2, 1e-3, 1e-4)
    x, flag, m = bar_run(p; scheme=Newmark(), stepping=AdaptiveStepping(; Δt=t_end / 16, t_end, rtol, atol=1e-8 * rtol))
    @test flag == 1
    push!(errs, norm(x - xref))
  end
  @test issorted(errs; rev=true)
  # BDF2 adaptive on the heat problem: the error decreases with rtol
  ph = heat_problem()
  th = 0.5
  href, _ = heat_structured(ph; scheme=Theta(0.5), stepping=TimeStepping(; Δt=th / 2048, t_end=th))
  e = [norm(heat_structured(ph; scheme=BDF(2), stepping=AdaptiveStepping(; Δt=th / 8, t_end=th, rtol, atol=1e-12))[1] - href) for rtol in (1e-3, 1e-5, 1e-6)]
  @test issorted(e; rev=true) && e[end] < 1e-1 * e[1]
  # schemes without an estimate are refused with AdaptiveStepping
  @test_throws ErrorException bar_run(p; scheme=CentralDifference(), stepping=AdaptiveStepping(; Δt=1e-3, t_end))
end
