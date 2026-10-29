# NewtonRaphson: tolerances and log, parity with GridapSolvers' NewtonSolver (temporary, while the
# compatibility extension exists), and the curvature criterion.
using Gridap, Gridap.Algebra, Gridap.FESpaces
using HyperFEM
using LinearAlgebra
using Test
using GridapSolvers: NonlinearSolvers
using GridapSolvers.NonlinearSolvers: NewtonSolver

# Small 2D hyperelastic block stretched by a Dirichlet condition: res, jac, spaces and conditions.
function newton_test_problem(; n=(4, 2))
  geometry = CartesianDiscreteModel((0, 2, 0, 1), n)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  Ψ, ∂Ψ∂F, ∂Ψ∂FF = physmodel()
  F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [0.6, 0.1], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F ∘ ∇(u)')))dΩ
  jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψ∂FF ∘ (F ∘ ∇(u)')) ⊙ ∇(du)'))dΩ
  (; res, jac, U, V, bc, dΩ)
end

@testset "NewtonTolerances and NewtonLog" begin
  tols = NewtonTolerances(maxiter=7, atol=1e-9, rtol=1e-4, ctol=1e-3)
  @test (tols.maxiter, tols.atol, tols.rtol, tols.ctol) == (7, 1e-9, 1e-4, 1e-3)
  @test_throws ArgumentError NewtonTolerances(maxiter=-1)
  log = NewtonLog(maxiter=3, verbose=true)
  @test verbosity(log) == 2
  @test verbosity(NewtonLog(verbose=false)) == 0
  @test verbosity(NewtonLog(verbose=3)) == 3
  @test length(log.residuals) == 4
  nls = NewtonRaphson(LUSolver(); maxiter=5, atol=1e-10, rtol=1e-6, verbose=0)
  @test get_tolerances(nls) == NewtonTolerances(maxiter=5, atol=1e-10, rtol=1e-6, ctol=0.0)
  @test stop_reason(nls) === :running
  @test !converged(nls)
end

@testset "Parity with GridapSolvers' NewtonSolver" begin
  p = newton_test_problem()
  histories = Dict{Symbol,Vector{Vector{Float64}}}()
  solutions = Dict{Symbol,Vector{Float64}}()
  for (name, nls) in (:hyperfem => NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8),
                      :gridapsolvers => NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8))
    h = Vector{Float64}[]
    xh = FEFunction(p.U, zero_free_values(p.U))
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, xh=xh)
    function record(post)
      push!(h, copy(nls.log.residuals[1:nls.log.num_iters+1]))
    end
    solve!(m; stepping=(nsteps=4, maxbisec=2), post=PostProcessor(m, record; is_vtk=false, filepath=mktempdir()))
    histories[name] = h
    solutions[name] = copy(get_free_dof_values(xh))
  end
  @test length(histories[:hyperfem]) == 4
  @test histories[:hyperfem] == histories[:gridapsolvers]        # bit-identical residual histories
  @test solutions[:hyperfem] == solutions[:gridapsolvers]
end

@testset "Stop reasons" begin
  p = newton_test_problem()
  # relative tolerance
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=0.0, rtol=1e-6)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = solve!(m; stepping=(nsteps=2, maxbisec=0))
  @test flag == 1 && converged(nls) && stop_reason(nls) === :converged_rtol
  r = residual_history(nls)
  @test length(r) == num_iterations(nls) + 1 && r[end] < 1e-6 * r[1] && r[end-1] >= 1e-6 * r[1]
  @test all(==(1.0), step_history(nls))
  # absolute tolerance
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-6, rtol=0.0)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  solve!(m; stepping=(nsteps=2, maxbisec=0))
  @test stop_reason(nls) === :converged_atol && residual_history(nls)[end] < 1e-6
  # maximum number of iterations: the driver bisects
  nls = NewtonRaphson(LUSolver(); maxiter=1, atol=1e-12, rtol=1e-12)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = @test_logs (:warn, r"Bisection") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && !converged(nls) && stop_reason(nls) === :maxiter && num_iterations(nls) == 1
  # maxiter = 0: no iteration, not converged
  nls = NewtonRaphson(LUSolver(); maxiter=0, atol=1e-12, rtol=1e-12)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  @test_logs (:warn, r"Bisection") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test stop_reason(nls) === :maxiter && num_iterations(nls) == 0
  # curvature criterion, relative to the first iteration: stops before the residual tolerances
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=0.0, rtol=1e-12, ctol=1e-6)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 1 && stop_reason(nls) === :converged_ctol
  @test residual_history(nls)[end] > 1e-12 * residual_history(nls)[1]
  nls0 = NewtonRaphson(LUSolver(); maxiter=20, atol=0.0, rtol=1e-12, ctol=0.0)
  m0 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls0)
  solve!(m0; stepping=(nsteps=1, maxbisec=0))
  @test num_iterations(nls0) > num_iterations(nls)
  # deprecated name still constructs the solver
  @test Newton_RaphsonSolver(LUSolver(); maxiter=3) isa NewtonRaphson
end

# stdout of `f()` (redirect_stdout needs a file)
function capture_stdout(f)
  path, io = mktemp()
  try
    redirect_stdout(f, io)
  finally
    close(io)
  end
  read(path, String)
end

@testset "Verbose output" begin
  p = newton_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=2)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  out = capture_stdout(() -> solve!(m; stepping=(nsteps=1, maxbisec=0)))
  @test occursin("‖R‖", out) && occursin("converged", out) && occursin("STEP: 0", out)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=0)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  out = capture_stdout(() -> solve!(m; stepping=(nsteps=1, maxbisec=0)))
  @test isempty(out)
end
