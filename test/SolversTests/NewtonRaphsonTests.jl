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
  @test occursin("‖R‖", out) && occursin("converged", out) && occursin("Step 1", out)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=0)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  out = capture_stdout(() -> solve!(m; stepping=(nsteps=1, maxbisec=0)))
  @test isempty(out)
end

# The Newton cache is lazy: building it (and a model) assembles and factorises nothing; the first solve
# does one assembly and one numerical setup, the next ones update it in place
mutable struct CountingLU <: Gridap.Algebra.LinearSolver
  setups::Int
  updates::Int
end
struct CountingSS <: Gridap.Algebra.SymbolicSetup
  ls::CountingLU
end
mutable struct CountingNS <: Gridap.Algebra.NumericalSetup
  ls::CountingLU
  ns::Gridap.Algebra.LUNumericalSetup
end
Gridap.Algebra.symbolic_setup(ls::CountingLU, A::AbstractMatrix) = CountingSS(ls)
Gridap.Algebra.numerical_setup(ss::CountingSS, A::AbstractMatrix) = (ss.ls.setups += 1; CountingNS(ss.ls, numerical_setup(symbolic_setup(LUSolver(), A), A)))
Gridap.Algebra.numerical_setup!(ns::CountingNS, A::AbstractMatrix) = (ns.ls.updates += 1; numerical_setup!(ns.ns, A); ns)
Gridap.Algebra.solve!(x::AbstractVector, ns::CountingNS, b::AbstractVector) = solve!(x, ns.ns, b)

@testset "Lazy numerical setup of the Newton cache" begin
  p = newton_test_problem()
  ls = CountingLU(0, 0)
  nls = NewtonRaphson(ls; maxiter=20, atol=1e-12, rtol=1e-12, verbose=0)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  c = m.caches.nls_cache
  @test c isa NewtonCache && c.ns === nothing && ls.setups == 0 && ls.updates == 0
  @test_throws ErrorException HyperFEM.Solvers.numerical_setup(c)
  x, flag = solve!(m; stepping=(nsteps=2, maxbisec=0))
  @test flag == 1 && c.ns isa CountingNS && ls.setups == 1
  @test ls.updates >= 2                       # one update per re-assembled tangent (two steps)
  # the same run with Gridap's LU gives the same solution (bit-identical: same assemblies, same factorisations)
  p = newton_test_problem()
  m2 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12, verbose=0))
  x2, _ = solve!(m2; stepping=(nsteps=2, maxbisec=0))
  @test x2 == x
end

@testset "EisenstatWalker forcing term" begin
  ft = HyperFEM.Solvers.forcing_term
  ew = EisenstatWalker()
  @test (ew.η0, ew.ηmax) == (1e-2, 1e-2)                              # defaults
  @test ft(ew, 0, 0.0, 1.0, 1.0, 1e-12) == 1e-2                       # η0
  @test ft(ew, 1, 1e-2, 0.1, 1.0, 1e-12) ≈ 0.9 * 0.1^2                # γ (‖R_k‖/‖R_{k-1}‖)^α
  @test ft(ew, 1, 1e-2, 0.5, 1.0, 1e-12) == 1e-2                      # bounded by ηmax (0.9·0.25 = 0.225)
  @test ft(ew, 1, 1e-2, 1e-9, 1.0, 1e-20) == 1e-10                    # floor ηmin (0.9·1e-18)
  @test ft(ew, 3, 1e-4, 1e-7, 1e-4, 1e-9) ≈ 0.005                     # no oversolving: 0.5 τ/‖R‖ = 0.005 > 0.9·1e-6
  @test ft(ew, 2, 1e-2, 1e-3, 1.0, 1e-20) ≈ 0.9e-6                    # the safeguard is inactive with the defaults (γ·1e-4 < 0.1)
  @test ft(ew, 2, 0.0, 1e-3, 1.0, 1e-20) == 1e-2                      # no previous forcing term (projected iteration): η0
  @test ft(ew, 2, 1e-2, 1e-3, 0.0, 1e-20) == 1e-2                     # rprev = 0: η0
  @test ft(ew, 2, 1e-2, 0.0, 1.0, 1e-12) == 1e-10                     # r = 0: no floor, η → ηmin
  @test ft(EisenstatWalker(ηmax=0.5), 2, 0.5, 1e-3, 1.0, 1e-20) ≈ 0.225   # active above (0.1/γ)^(1/α) ≈ 0.33
  ew2 = EisenstatWalker(ηmax=0.9)
  @test ft(ew2, 2, 0.5, 1e-3, 1.0, 1e-12) ≈ 0.9 * 0.5^2               # safeguard γ η_{k-1}^α = 0.225 > 0.1
  @test ft(ew2, 2, 0.3, 1e-3, 1.0, 1e-12) ≈ 0.9 * 1e-6                # no safeguard: γ·0.09 = 0.081 < 0.1
  @test_throws ArgumentError EisenstatWalker(η0=0.5, ηmax=0.1)
  @test_throws ArgumentError EisenstatWalker(γ=1.5)
  @test_throws ArgumentError EisenstatWalker(α=1.0)
  # with a direct solver the forcing term is recorded and ignored: the LU trajectory is unchanged
  p = newton_test_problem()
  hs = Vector{Float64}[]
  for forcing in (nothing, EisenstatWalker())
    nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, forcing)
    xh = FEFunction(p.U, zero_free_values(p.U))
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, xh=xh)
    solve!(m; stepping=(nsteps=4, maxbisec=2))
    push!(hs, copy(residual_history(nls)))
    forcing === nothing ? (@test all(==(0.0), forcing_history(nls))) : (@test forcing_history(nls)[1] == 1e-2)
  end
  @test hs[1] == hs[2]
end

# A direct solver with a tolerance (LU underneath): records what the Newton asks of set_tolerance!, and can throw
# at a given solve (TOLLU_THROW) to check the restore on an exception
struct TolLU <: LinearSolver end
struct TolLUSS <: SymbolicSetup end
mutable struct TolLUNS{T} <: NumericalSetup
  lu::T
  rtol::Float64
  seen::Vector{Float64}
  nsolve::Int
end
const TOLLU_OWN = 1e-5                                       # the "setup" tolerance that `nothing` restores
const TOLLU_START = 1e-6                                     # the tolerance the setup starts with (set by the caller)
const TOLLU_THROW = Ref(0)
Gridap.Algebra.symbolic_setup(::TolLU, A::AbstractMatrix) = TolLUSS()
Gridap.Algebra.numerical_setup(::TolLUSS, A::AbstractMatrix) =
  TolLUNS(numerical_setup(symbolic_setup(LUSolver(), A), A), TOLLU_START, Float64[], 0)
Gridap.Algebra.numerical_setup!(ns::TolLUNS, A::AbstractMatrix) = (numerical_setup!(ns.lu, A); ns)
function Gridap.Algebra.solve!(x::AbstractVector, ns::TolLUNS, b::AbstractVector)
  ns.nsolve += 1
  ns.nsolve == TOLLU_THROW[] && error("TolLU: failure requested by the test")
  solve!(x, ns.lu, b)
end
HyperFEM.Solvers.set_tolerance!(ns::TolLUNS, r) = (ns.rtol = r === nothing ? TOLLU_OWN : r; push!(ns.seen, ns.rtol); ns)
HyperFEM.Solvers.tolerance(ns::TolLUNS) = ns.rtol

struct NoHookNS <: NumericalSetup end

@testset "Forcing term: the tolerance in force before the Newton is restored" begin
  p = newton_test_problem()
  nls = NewtonRaphson(TolLU(); maxiter=20, atol=1e-10, rtol=1e-8, forcing=EisenstatWalker())
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, xh=FEFunction(p.U, zero_free_values(p.U)))
  _, flag = solve!(m; stepping=(nsteps=2, maxbisec=0))
  @test flag == 1
  ns = m.caches.nls_cache.ns
  @test ns.rtol == TOLLU_START                                 # not the solver's own (TOLLU_OWN): the caller's value
  @test 1e-2 in ns.seen && TOLLU_START in ns.seen && !(TOLLU_OWN in ns.seen)
  @test minimum(ns.seen) ≥ TOLLU_START                         # never tighter than the tolerance at the start
  # an exception inside a Newton solve (the linear solve of its second iteration) restores it as well
  m2 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, xh=FEFunction(p.U, zero_free_values(p.U)))
  TOLLU_THROW[] = 2                                            # the linear solve of the second Newton iteration
  try
    @test_throws ErrorException solve!(m2; stepping=(nsteps=2, maxbisec=0))
  finally
    TOLLU_THROW[] = 0
  end
  ns2 = m2.caches.nls_cache.ns
  @test ns2.nsolve == 2
  @test length(ns2.seen) ≥ 2 && ns2.seen[1] == 1e-2            # a forcing term was in force when it threw
  @test ns2.seen[end] == TOLLU_START && ns2.rtol == TOLLU_START  # and the finally put the caller's value back
end

@testset "set_tolerance!: warning for linear solvers without the hook" begin
  @test_logs (:warn, r"does not support a forcing term") set_tolerance!(NoHookNS(), 1e-2)
  @test tolerance(NoHookNS()) === nothing
  A = [4.0 1.0; 1.0 3.0]
  @test tolerance(numerical_setup(symbolic_setup(LUSolver(), A), A)) === nothing
  @test_logs set_tolerance!(numerical_setup(symbolic_setup(BackslashSolver(), A), A), 1e-2)   # silent for direct solvers
end
