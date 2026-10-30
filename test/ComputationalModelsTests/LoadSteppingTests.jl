# LoadStepping and the incremental loop step! of StaticNonlinearModel.
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: as_stepping, LoadStep, finished
using LinearAlgebra
using Logging
using Test

@testset "LoadStepping" begin
  s = LoadStepping()
  @test (s.nsteps, s.maxbisec, s.grow, s.easy) == (20, 15, 1.0, 10)
  @test as_stepping((nsteps=4, maxbisec=2)) === LoadStepping(nsteps=4, maxbisec=2)
  @test as_stepping(s) === s
  @test_throws MethodError as_stepping((nsteps=4, maxbisect=2))
  @test_throws ErrorException as_stepping(4)
  @test_throws ErrorException LoadStepping(nsteps=0)
  @test_throws ErrorException LoadStepping(grow=0.5)

  # the NamedTuple and the LoadStepping give the same run; step! driven by hand gives the same run
  nls() = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12, verbose=0)
  p = drivers_test_problem()
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls())
  xref, flag = solve!(m; stepping=(nsteps=4, maxbisec=0), ProjectDirichlet=true)
  xref = copy(xref)
  @test flag == 1
  p = drivers_test_problem()
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls())
  x, flag = solve!(m; stepping=LoadStepping(nsteps=4, maxbisec=0), ProjectDirichlet=true)
  @test flag == 1 && x == xref
  p = drivers_test_problem()
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls())
  stepping = LoadStepping(nsteps=4, maxbisec=0)
  st = LoadStep(stepping)
  @test (st.Λ, st.ΔΛ, st.ΔΛ0, st.nsteps, st.nbisect, st.nbisect_total) == (0.0, 0.25, 0.25, 0, 0, 0)
  Λs = Float64[]
  while !finished(st)
    @test step!(m, st, stepping; ProjectDirichlet=true)
    push!(Λs, st.Λ)
  end
  @test Λs == [0.25, 0.5, 0.75, 1.0] && st.nsteps == 4 && st.nbisect_total == 0
  @test m.caches.x == xref
end

# A solver whose attempts listed in `fails` (1-based count of calls) fail without touching the state
mutable struct FailAt{N<:NewtonRaphson,L} <: Gridap.Algebra.NonlinearSolver
  nls::N
  fails::Vector{Int}
  calls::Int
  ls::L
end
FailAt(nls, fails) = FailAt(nls, fails, 0, nls.ls)
HyperFEM.ComputationalModels.instantiate_caches(x, s::FailAt, op) = HyperFEM.ComputationalModels.instantiate_caches(x, s.nls, op)
function Gridap.Algebra.solve!(x::AbstractVector, s::FailAt, op::Gridap.Algebra.NonlinearOperator, cache)
  s.calls += 1
  s.calls in s.fails && return cache
  solve!(x, s.nls, op, cache)
end
HyperFEM.ComputationalModels.nls_converged(s::FailAt) = !(s.calls in s.fails) && converged(s.nls)
HyperFEM.ComputationalModels.nls_message(s::FailAt, msg) = nothing
HyperFEM.ComputationalModels.nls_iterations(s::FailAt) = HyperFEM.ComputationalModels.nls_iterations(s.nls)

@testset "Growth of the increment and consecutive bisections" begin
  nls() = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12, verbose=0)
  function runcase(stepping, fails)
    p = drivers_test_problem()
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=FailAt(nls(), fails))
    Λs = Float64[]
    post = PostProcessor(m, post -> push!(Λs, post.Λ[post.iter]); is_vtk=false, filepath=mktempdir())
    x, flag = @test_logs min_level = Logging.Error solve!(m; stepping, post)
    Λs, flag
  end
  # no failure: the increment never exceeds ΔΛ0, so growth does not act
  Λs, flag = runcase(LoadStepping(nsteps=4, grow=2.0, easy=50), Int[])
  @test flag == 1 && Λs == [0.25, 0.5, 0.75, 1.0]
  # a first failed attempt: the increment is halved, then grows back to ΔΛ0 after the easy step
  Λs, flag = runcase(LoadStepping(nsteps=10, maxbisec=1, grow=2.0, easy=50), [1])
  @test flag == 1 && Λs ≈ [0.05; 0.15:0.1:0.95; 1.0]
  # the same without growth (grow = 1, as before): the halved increment stays
  Λs, flag = runcase(LoadStepping(nsteps=10, maxbisec=1), [1])
  @test flag == 1 && Λs ≈ 0.05:0.05:1.0
  # the same with growth but a step that is not easy (easy = 0): no growth either
  Λs, flag = runcase(LoadStepping(nsteps=10, maxbisec=1, grow=2.0, easy=0), [1])
  @test flag == 1 && Λs ≈ 0.05:0.05:1.0
  # growth by 1.5 is capped at ΔΛ0: 0.05 → 0.075 → 0.1
  Λs, flag = runcase(LoadStepping(nsteps=10, maxbisec=1, grow=1.5, easy=50), [1])
  @test flag == 1 && Λs[1:3] ≈ [0.05, 0.125, 0.225] && all(diff(Λs[3:end-1]) .≈ 0.1)
  # bisections are counted consecutively: failures at attempts 1 and 3 (one bisection each, a converged
  # step in between) pass with maxbisec = 1; two failures in a row do not
  Λs, flag = runcase(LoadStepping(nsteps=4, maxbisec=1), [1, 3])
  @test flag == 1 && Λs ≈ [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
  Λs, flag = runcase(LoadStepping(nsteps=4, maxbisec=1), [1, 2])
  @test flag == 0 && isempty(Λs)
end

@testset "Hooks of the incremental loop" begin
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12, verbose=0)
  p = drivers_test_problem()
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=FailAt(nls, [2]))
  pre, bis = Tuple{Float64,Float64}[], Tuple{Float64,Float64}[]
  x, flag = @test_logs min_level = Logging.Error begin
    solve!(m; stepping=LoadStepping(nsteps=2, maxbisec=1), presolver=(Λ, ΔΛ) -> push!(pre, (Λ, ΔΛ)),
           on_bisection=(Λ, ΔΛ) -> push!(bis, (Λ, ΔΛ)))
  end
  @test flag == 1
  # attempts: 0 → 0.5 ✓, 0.5 → 1 ✗ (bisected to 0.25), 0.5 → 0.75 ✓, 0.75 → 1 ✓
  @test pre == [(0.0, 0.5), (0.5, 0.5), (0.5, 0.25), (0.75, 0.25)]
  @test bis == [(0.5, 0.25)]
  # the model-level hooks accept nothing (default) and the loop is the same without them
  p = drivers_test_problem()
  m2 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=FailAt(nls, [2]))
  x2, _ = @test_logs min_level = Logging.Error solve!(m2; stepping=LoadStepping(nsteps=2, maxbisec=1))
  @test x2 == x
end
