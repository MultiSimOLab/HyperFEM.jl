# Line searches: exact injectivity step (2D quadratic, 3D cubic, J_min) against sampling, the step of a
# field over a measure, InjectivityLS and BacktrackingLS inside the Newton, failures, deprecated names.
using Gridap, Gridap.Algebra, Gridap.FESpaces, Gridap.CellData
using HyperFEM
using HyperFEM.Solvers: _det_coefficients, _smallest_root
using LinearAlgebra
using Random
using Test

const D2 = TensorValue{2,2,Float64}
const D3 = TensorValue{3,3,Float64}

# det(F + s G) sampled on (0, α): never below the bound; at α the bound is reached (if α < 1)
function check_step(∇u, ∇du, α, jmin; n=400)
  F = ∇u + one(∇u)
  J = det(F)
  jm = J > jmin ? jmin : 0.0
  ok = all(det(F + s * ∇du) > jm for s in range(0, α; length=n + 2)[2:end-1])
  if α < 1
    ok &= abs(det(F + α * ∇du) - jm) ≤ 1e-10 * max(1.0, abs(J))
  else
    ok &= det(F + ∇du) >= jm - 1e-10 * max(1.0, abs(J))
  end
  ok
end

@testset "Exact injectivity step at a point" begin
  rng = MersenneTwister(7)
  for (T, nd) in ((D2, 4), (D3, 9))
    nroot, none = 0, 0
    for _ in 1:400
      ∇u = T(ntuple(i -> 0.4 * (rand(rng) - 0.5), nd)...)
      ∇du = T(ntuple(i -> 3.0 * (rand(rng) - 0.5), nd)...)
      for jmin in (0.0, 1e-3)
        α = injectivity_step(∇u, ∇du, 1.0, jmin)
        @test 0 < α <= 1
        @test check_step(∇u, ∇du, α, jmin)
        α < 1 ? (nroot += 1) : (none += 1)
        # β scales the root
        @test injectivity_step(∇u, ∇du, 0.9, jmin) ≈ (α < 1 ? 0.9α : 1.0)
      end
    end
    @test nroot > 50 && none > 50          # both branches exercised
  end
  # 3D polynomial = det(F + s G) sampled (the α² and α³ coefficients of the cubic)
  for _ in 1:50
    ∇u = D3(ntuple(i -> 0.4 * (rand(rng) - 0.5), 9)...)
    ∇du = D3(ntuple(i -> 3.0 * (rand(rng) - 0.5), 9)...)
    F = ∇u + one(∇u)
    (c, t, Dc, E), J = _det_coefficients(F, ∇du, 0.0)
    @test J ≈ det(F)
    for s in (0.3, 0.7, 1.0)
      @test c + s * (t + s * (Dc + s * E)) ≈ det(F + s * ∇du) rtol = 1e-12
    end
  end
  # quadratic and cubic root finders on known polynomials
  @test _smallest_root(1.0, -3.0, 2.0) ≈ 0.5            # (1 - α)(1 - 2α): roots 1/2, 1
  @test _smallest_root(1.0, -3.0, 2.0, 0.0) ≈ 0.5
  @test _smallest_root(1.0, 1.0, 1.0) == Inf              # no real roots
  @test _smallest_root(1.0, -2.0, 0.0) ≈ 0.5              # linear
  @test _smallest_root(-0.6, 5.0, -11.6, 8.0) ≈ 0.2       # 8(α - 0.2)(α - 0.5)(α - 0.75): smallest root 0.2
  @test _smallest_root(0.6, -5.0, 11.6, -8.0) ≈ 0.2       # the same with the opposite sign
  @test _smallest_root(1.0, 0.0, 0.0, -1.0) ≈ 1.0         # 1 - α³: root at 1
  @test _smallest_root(1.0, 0.0, 0.0, -0.5) == Inf        # 1 - α³/2 > 0 on (0, 1]
  @test _smallest_root(1.0, -2.0, 1.0) ≈ 1.0              # (1 - α)²: double root at 1
  # already below J_min: the bound is J > 0
  ∇u = D2(-0.9995, 0.0, 0.0, 0.0)                         # J = 5e-4 < jmin = 1e-3
  ∇du = D2(-1.0, 0.0, 0.0, 0.0)
  @test injectivity_step(∇u, ∇du, 1.0, 1e-3) ≈ 5e-4
end

@testset "Injectivity step of a field" begin
  model = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 3, 3))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
  V = TestFESpace(Ω, reffe)
  uh = interpolate_everywhere(x -> VectorValue(0.1x[1]^2, 0.2x[2] * x[3], -0.1x[3]), V)
  duh = interpolate_everywhere(x -> VectorValue(-0.8x[1], -1.5x[2]^2, 0.3x[1] * x[3]), V)
  β, jmin = 1 - 1e-6, 1e-4
  α = injectivity_step(dΩ, uh, duh, β, jmin)
  # reference: every quadrature point through the pointwise function
  pts = get_cell_points(dΩ.quad)
  gu = (∇(uh)')(pts)
  gdu = (∇(duh)')(pts)
  αref = minimum(minimum(injectivity_step(a, b, β, jmin) for (a, b) in zip(gu[c], gdu[c])) for c in 1:num_cells(Ω))
  @test α == αref
  @test 0 < α < 1
  # multi-field: the displacement is the first field
  Vφ = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1))
  X = MultiFieldFESpace([V, Vφ])
  xh = interpolate_everywhere([x -> VectorValue(0.1x[1]^2, 0.2x[2] * x[3], -0.1x[3]), x -> x[1]], X)
  dxh = interpolate_everywhere([x -> VectorValue(-0.8x[1], -1.5x[2]^2, 0.3x[1] * x[3]), x -> 0.0], X)
  @test injectivity_step(dΩ, xh[1], dxh[1], β, jmin) == α
end

# Small 2D hyperelastic block compressed by a Dirichlet condition (large step: the full Newton step
# would invert elements)
function linesearch_test_problem(; stretch=-0.5)
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  _, ∂Ψ∂F, ∂Ψ∂FF = physmodel()
  F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [2stretch, 0.3], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F ∘ ∇(u)')))dΩ
  jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψ∂FF ∘ (F ∘ ∇(u)')) ⊙ ∇(du)'))dΩ
  (; res, jac, U, V, bc, dΩ)
end

@testset "InjectivityLS in the Newton" begin
  # reference: small steps with the full Newton step
  p2 = linesearch_test_problem()
  nls2 = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10)
  m2 = StaticNonlinearModel(p2.res, p2.jac, p2.U, p2.V, p2.bc; nls=nls2)
  x2, _ = solve!(m2; stepping=(nsteps=10, maxbisec=0))
  x2 = copy(x2)
  # two large increments with the injectivity line search and the Dirichlet projection
  p = linesearch_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=InjectivityLS())
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=p.dΩ)
  x, flag = solve!(m; stepping=(nsteps=2, maxbisec=0), ProjectDirichlet=true)
  @test flag == 1 && converged(nls)
  @test all(0 .< step_history(nls) .<= 1)
  uh = FEFunction(p.U, x)
  Jq = (A -> det(A + one(A))) ∘ (∇(uh)')
  @test minimum(minimum.(Jq(get_cell_points(p.dΩ.quad)))) > 0  # no inverted element
  @test isapprox(x, x2; rtol=1e-6)
  # the injectivity limit: three times the converged deformation inverts elements
  u0 = FEFunction(p.U, zeros(length(x)))
  du = FEFunction(p.V, 3x)
  αmax = injectivity_step(m, u0, du, 1 - 1e-6, 0.0)
  @test 0 < αmax < 1
  @test max_step(HyperFEM.ComputationalModels.model_operator(m, m.caches.op), zeros(length(x)), 3x, 1 - 1e-6, 0.0) == αmax
  # jmin and β are honoured (safe load factor of the Dirichlet projection and the search)
  p3 = linesearch_test_problem()
  ls3 = InjectivityLS(jmin=1e-3, β=0.5)
  nls3 = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=ls3)
  m3 = StaticNonlinearModel(p3.res, p3.jac, p3.U, p3.V, p3.bc; nls=nls3, injectivity=p3.dΩ)
  x3, flag3 = solve!(m3; stepping=(nsteps=2, maxbisec=0), ProjectDirichlet=true)
  @test flag3 == 1
  @test isapprox(x3, x2; rtol=1e-6)
  # without `injectivity` in the model the line search cannot work: clear error
  p4 = linesearch_test_problem()
  nls4 = NewtonRaphson(LUSolver(); maxiter=30, linesearch=InjectivityLS())
  m4 = StaticNonlinearModel(p4.res, p4.jac, p4.U, p4.V, p4.bc; nls=nls4)
  @test_throws ErrorException solve!(m4; stepping=(nsteps=1, maxbisec=0))
  @test_throws ErrorException StaticNonlinearModel(p4.res, p4.jac, p4.U, p4.V, p4.bc; nls=nls4, injectivity=1.0)
  # Armijo needs an energy
  nls5 = NewtonRaphson(LUSolver(); maxiter=30, linesearch=InjectivityLS(armijo=true))
  m5 = StaticNonlinearModel(p4.res, p4.jac, p4.U, p4.V, p4.bc; nls=nls5, injectivity=p4.dΩ)
  op5 = HyperFEM.ComputationalModels.model_operator(m5, m5.caches.op)
  @test_throws ErrorException energy(op5, zeros(num_free_dofs(p4.U)))
  @test_throws ErrorException energy(m5.caches.op, zeros(num_free_dofs(p4.U)))
end

@testset "Line-search failure and bisection" begin
  p = linesearch_test_problem(stretch=0.3)
  # c = 0: the backtracking of InjectivityLS can never accept a step -> failure, the driver bisects
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=InjectivityLS(c=0.0, maxiter=3))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=p.dΩ)
  x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && stop_reason(nls) === :linesearch_failed
  @test step_history(nls) == [0.0]
  @test residual_history(nls)[2] == residual_history(nls)[1]       # the residual is left at x
  @test x == zeros(length(x))
  # BacktrackingLS takes the last α when the condition is never met (no failure)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=BacktrackingLS(c=-1.0, maxiter=4))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test step_history(nls)[1] ≈ 0.5^4
end

@testset "Deprecated line-search names" begin
  @test Roman_LS(maxiter=3) isa BacktrackingLS
  @test HyperFEM.Solvers.LineSearch() isa FullStepLS
  @test_throws ArgumentError InjectivityLS(β=1.5)
  @test_throws ArgumentError InjectivityLS(jmin=-1.0)
end
