# Hybrid Newton: switch to the projected tangent after a failed line search, absolute tolerance in
# projected mode, return to the consistent tangent, every solve starting consistent, errors.
using Gridap, Gridap.Algebra, Gridap.FESpaces, Gridap.CellData
using HyperFEM
using HyperFEM.Solvers: search!
using LinearAlgebra, SparseArrays
using Test

# A linear solver (LU underneath) that reports an indefinite operator on its first `nfail` solves (S5): the solve
# fails with `linear_indefinite`, the state is untouched
struct IndefLU <: Gridap.Algebra.LinearSolver
  nfail::Int
end
struct IndefLUSS <: Gridap.Algebra.SymbolicSetup
  nfail::Int
end
mutable struct IndefLUNS{T} <: Gridap.Algebra.NumericalSetup
  lu::T
  nfail::Int
  calls::Int
  failed::Bool
end
Gridap.Algebra.symbolic_setup(s::IndefLU, A::AbstractMatrix) = IndefLUSS(s.nfail)
Gridap.Algebra.numerical_setup(ss::IndefLUSS, A::AbstractMatrix) =
  IndefLUNS(numerical_setup(symbolic_setup(LUSolver(), A), A), ss.nfail, 0, false)
Gridap.Algebra.numerical_setup!(ns::IndefLUNS, A::AbstractMatrix) = (numerical_setup!(ns.lu, A); ns)
function Gridap.Algebra.solve!(x::AbstractVector, ns::IndefLUNS, b::AbstractVector)
  ns.calls += 1
  ns.failed = ns.calls <= ns.nfail
  ns.failed && (fill!(x, 0.0); return x)
  solve!(x, ns.lu, b)
end
HyperFEM.Solvers.linear_converged(ns::IndefLUNS) = !ns.failed
HyperFEM.Solvers.linear_indefinite(ns::IndefLUNS) = ns.failed

# A line search that fails on its first `nfail` calls (the residual left at x) and takes the full step afterwards
mutable struct FailFirstLS <: HyperFEM.Solvers.AbstractLineSearch
  nfail::Int
  calls::Int
end
function HyperFEM.Solvers.search!(ls::FailFirstLS, w, op, x, dx, b)
  ls.calls += 1
  if ls.calls <= ls.nfail
    residual!(b, op, x)
    return 0.0
  end
  x .+= dx
  residual!(b, op, x)
  return 1.0
end

function hybrid_test_problem(; stretch=0.2, switch=nothing)
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  model = switch === nothing ? physmodel : HessianRegularization(physmodel; δ=1e-6, switch=switch)
  km = Kinematics(Mechano, Solid)
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [2stretch, 0.3], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> residual(model, km, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(model, km, u, du, v, dΩ)
  en(Λ) = u -> energy(model, km, u, dΩ)
  (; res, jac, en, U, V, bc, dΩ)
end

@testset "HybridTangent" begin
  @test_throws ArgumentError HybridTangent(hybback=2.0, atol_projected=1.0)
  @test_throws ArgumentError HybridTangent(atol_projected=0.0)
  # reference solution
  pref = hybrid_test_problem()
  nref = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12)
  mref = StaticNonlinearModel(pref.res, pref.jac, pref.U, pref.V, pref.bc; nls=nref)
  xref, _ = solve!(mref; stepping=(nsteps=2, maxbisec=0))
  xref = copy(xref)
  r_ref = copy(residual_history(nref))

  # a failed first line search: switch to the projected tangent, absolute criterion there, back to the
  # consistent tangent once ‖R‖ < hybback ‖R‖_switch, convergence with the consistent criteria
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  ls = FailFirstLS(1, 0)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=ls,
                      tangent=HybridTangent(hybback=1e-2, atol_projected=1e-20))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 1 && converged(nls)
  modes = mode_history(nls)
  steps = step_history(nls)
  @test steps[1] == 0.0 && modes[1] === :consistent                 # the failed iteration, α = 0
  @test residual_history(nls)[2] == residual_history(nls)[1]        # residual unchanged by it
  @test modes[2] === :projected                                     # the switch
  @test nls.log.nswitches == 1
  @test modes[end] === :consistent                                  # back to consistent to finish
  @test stop_reason(nls) in (:converged_atol, :converged_rtol)
  @test !is_projected(switch)                                       # left consistent for the next solve
  @test isapprox(x, xref; rtol=1e-8)
  # the return happens exactly when ‖R‖ < hybback ‖R‖_switch
  r = residual_history(nls)
  r_sw = r[2]
  k = findfirst(i -> r[i+1] < 1e-2 * r_sw, 2:length(r)-1)
  @test k !== nothing && modes[k+2] === :consistent && all(==(:projected), modes[2:k+1])

  # with a loose absolute tolerance the solve converges in projected mode
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=FailFirstLS(1, 0),
                      tangent=HybridTangent(hybback=1e-6, atol_projected=1e-3))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 1 && stop_reason(nls) === :converged_projected
  @test residual_history(nls)[end] < 1e-3 && all(==(:projected), mode_history(nls)[2:end])
  @test !is_projected(switch)

  # a second failure in projected mode fails the solve; the driver bisects and the next attempt starts consistent
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  ls = FailFirstLS(2, 0)
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=ls,
                      tangent=HybridTangent(hybback=1e-2, atol_projected=1e-20))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = @test_logs (:warn, r"Bisection") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=3))
  @test flag == 1 && converged(nls)
  @test ls.calls > 2
  @test isapprox(x, xref; rtol=1e-8)

  # with a forcing term: no η on the projected iterations (the solver keeps the tolerance it had), η0 again on the
  # first consistent iteration after them (TolLU of NewtonRaphsonTests.jl records the tolerances it is given)
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  nls = NewtonRaphson(TolLU(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=FailFirstLS(1, 0),
                      tangent=HybridTangent(hybback=1e-2, atol_projected=1e-20), forcing=EisenstatWalker())
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 1 && converged(nls)
  etas, modes = forcing_history(nls), mode_history(nls)
  @test etas[1] == 1e-2                                             # the failed iteration, consistent
  @test all(==(0.0), etas[modes .=== :projected])
  lastproj = findlast(==(:projected), modes)
  k = findfirst(>(0.0), etas[lastproj+1:end])
  @test k !== nothing && etas[lastproj+k] == 1e-2                   # restart from η0
  @test m.caches.nls_cache.ns.rtol == TOLLU_START
  @test isapprox(x, xref; rtol=1e-8)

  # ConsistentTangent: a failed line search stops the solve
  p = hybrid_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=FailFirstLS(1, 0))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && stop_reason(nls) === :linesearch_failed

  # S5: an indefinite tangent reported by the linear solver is the signal of a failed line search — the solve switches
  # to the projected tangent at the same state (no bisection) and finishes consistent, at the LU solution
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  nls = NewtonRaphson(IndefLU(1); maxiter=30, atol=1e-10, rtol=1e-12,
                      tangent=HybridTangent(hybback=1e-2, atol_projected=1e-20))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = solve!(m; stepping=(nsteps=1, maxbisec=0))             # maxbisec = 0: a bisection would fail the run
  @test flag == 1 && converged(nls)
  modes = mode_history(nls)
  @test step_history(nls)[1] == 0.0 && modes[1] === :consistent     # the failed solve, logged with α = 0
  @test residual_history(nls)[2] == residual_history(nls)[1]        # the state is untouched by it
  @test modes[2] === :projected && nls.log.nswitches == 1
  @test modes[end] === :consistent
  @test !is_projected(switch)
  @test isapprox(x, xref; rtol=1e-8)
  @test linear_indefinite(LUSolver() |> s -> numerical_setup(symbolic_setup(s, sparse(1.0I, 2, 2)), sparse(1.0I, 2, 2))) == false

  # a second indefinite report, in projected mode, fails the solve (the driver bisects)
  switch = TangentSwitch()
  p = hybrid_test_problem(; switch)
  nls = NewtonRaphson(IndefLU(2); maxiter=30, atol=1e-10, rtol=1e-12,
                      tangent=HybridTangent(hybback=1e-2, atol_projected=1e-20))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, tangent_switch=switch)
  x, flag = @test_logs (:warn, r"Bisection") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=3))
  @test flag == 1 && converged(nls)
  @test isapprox(x, xref; rtol=1e-8)

  # ConsistentTangent: the indefinite report fails the solve as any other linear failure
  p = hybrid_test_problem()
  nls = NewtonRaphson(IndefLU(1); maxiter=30, atol=1e-10, rtol=1e-12)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && stop_reason(nls) === :linear_failed

  # HybridTangent needs the switch of the model
  p = hybrid_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, linesearch=FailFirstLS(1, 0), tangent=HybridTangent(atol_projected=1.0))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  @test_throws ErrorException solve!(m; stepping=(nsteps=1, maxbisec=0))

  # no allocations added to the Newton loop by the strategy (consistent tangent, diagonal model problem)
  nls = NewtonRaphson(DiagonalSolver(); maxiter=30, atol=1e-13, rtol=1e-14, tangent=HybridTangent(atol_projected=1e-20))
  @test solve_twice(nls, op) == 0
end
