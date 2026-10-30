# LoadStepping and the incremental loop step! of StaticNonlinearModel.
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: as_stepping, LoadStep, finished
using LinearAlgebra
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
