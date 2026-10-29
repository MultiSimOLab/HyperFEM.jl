# Hybrid Newton: switch to the projected tangent after a failed line search, absolute tolerance in
# projected mode, return to the consistent tangent, every solve starting consistent, errors.
using Gridap, Gridap.Algebra, Gridap.FESpaces, Gridap.CellData
using HyperFEM
using HyperFEM.Solvers: search!
using LinearAlgebra
using Test

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
  w .= x .+ dx
  residual!(b, op, w)
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

  # ConsistentTangent: a failed line search stops the solve
  p = hybrid_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-12, linesearch=FailFirstLS(1, 0))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any solve!(m; stepping=(nsteps=1, maxbisec=0))
  @test flag == 0 && stop_reason(nls) === :linesearch_failed

  # HybridTangent needs the switch of the model
  p = hybrid_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=30, linesearch=FailFirstLS(1, 0), tangent=HybridTangent(atol_projected=1.0))
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  @test_throws ErrorException solve!(m; stepping=(nsteps=1, maxbisec=0))

  # no allocations added to the Newton loop by the strategy (consistent tangent, diagonal model problem)
  nls = NewtonRaphson(DiagonalSolver(); maxiter=30, atol=1e-13, rtol=1e-14, tangent=HybridTangent(atol_projected=1e-20))
  @test solve_twice(nls, op) == 0
end
