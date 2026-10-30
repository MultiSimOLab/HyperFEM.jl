# Energy of the model: consistency with the residual (finite differences), Neumann work, the energy of
# WeakForms, and the Armijo line search.
using Gridap, Gridap.Algebra, Gridap.FESpaces, Gridap.CellData
using HyperFEM
using HyperFEM.ComputationalModels: model_operator
using LinearAlgebra
using Random
using Test

function energy_test_problem(; neumann=false, stretch=0.2)
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  add_tag_from_tags!(labels, "top", [6])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  Ψ, ∂Ψ∂F, ∂Ψ∂FF = physmodel()
  km = Kinematics(Mechano, Solid)
  F, _, _ = get_Kinematics(km)
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [2stretch, 0.3], EvolutionFunctions.ramp()))
  N_bc = BoundaryConditions(BoundaryCondition("top", [0.05, -0.3], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> residual(physmodel, km, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, km, u, du, v, dΩ)
  en(Λ) = u -> energy(physmodel, km, u, dΩ)
  nc = neumann ? NeumannConditions(geometry, N_bc, 2) : nothing
  (; res, jac, en, U, V, bc, dΩ, neumann=nc, Ψ, F)
end

@testset "Energy and residual" begin
  rng = MersenneTwister(3)
  for neumann in (false, true)
    p = energy_test_problem(; neumann)
    nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10)
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, neumann=p.neumann, energy=p.en, injectivity=p.dΩ)
    x, _ = solve!(m; stepping=(nsteps=2, maxbisec=0))
    # operator with the hooks at Λ = 1 (load 1): E and R = ∇E
    op = model_operator(m, m.caches[7][], 1.0, 1.0)
    x0 = x .+ 0.01 .* randn(rng, length(x))
    E0 = energy(op, x0)
    @test isfinite(E0)
    R = residual(op, x0)
    for _ in 1:3
      δ = randn(rng, length(x))
      h = 1e-5
      dE = (energy(op, x0 .+ h .* δ) - energy(op, x0 .- h .* δ)) / 2h
      @test isapprox(dE, dot(R, δ); rtol=1e-6)
    end
    # the energy of WeakForms is the integral of Ψ
    uh = FEFunction(p.U, x0)
    Eint = sum(∫(p.Ψ ∘ (p.F ∘ ∇(uh)'))p.dΩ)
    if neumann
      @test !isapprox(energy(op, x0), Eint)   # plus the work of the load
    else
      @test energy(op, x0) ≈ Eint
    end
  end
end

@testset "Armijo line search" begin
  # reference: small steps, full Newton step
  p2 = energy_test_problem()
  nls2 = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10)
  m2 = StaticNonlinearModel(p2.res, p2.jac, p2.U, p2.V, p2.bc; nls=nls2)
  x2, _ = solve!(m2; stepping=(nsteps=10, maxbisec=0))
  x2 = copy(x2)
  p = energy_test_problem()
  ls = InjectivityLS(armijo=true, c1=1e-4, maxiter=20)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=ls)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=p.dΩ, energy=p.en)
  x, flag = solve!(m; stepping=(nsteps=2, maxbisec=0), ProjectDirichlet=true)
  @test flag == 1 && converged(nls)
  @test all(0 .< step_history(nls) .<= 1)
  @test isapprox(x, x2; rtol=1e-6)
  # a sufficient-decrease constant that no step can meet: failure, the driver bisects
  p = energy_test_problem()
  ls = InjectivityLS(armijo=true, c1=1e6, maxiter=5)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=ls)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=p.dΩ, energy=p.en)
  x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && stop_reason(nls) === :linesearch_failed && step_history(nls) == [0.0]
  # the model's energy is needed
  p = energy_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, linesearch=InjectivityLS(armijo=true))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=p.dΩ)
  @test_throws ErrorException solve!(m; stepping=(nsteps=1, maxbisec=0))
end
