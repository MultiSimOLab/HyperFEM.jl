# Memory across load steps (requirement of 2026-10-31): the live heap after every load step does not grow once the
# caches exist — no buffer, closure or history grows with the number of steps. Nonlinear mechanics problem of
# krylov_common.jl (included by KrylovSolversTests.jl), both assembly backends, a direct and an iterative linear
# solver (the AMG hierarchy is rebuilt at every setup: the likeliest place for a leak). The whole heap of the test
# session is hundreds of MiB, so a relative threshold would hide a per-step copy of a vector; the test measures the
# trend over the last steps instead: a leak of one vector per step (≈3 KiB here) is a growth of tens of KiB over
# them. `maxbisec = 0`: no bisection or late line-search failure compiles new methods after the first steps.
isdefined(@__MODULE__, :s1_mech_problem) || include("krylov_common.jl")

@testset "Memory flat across load steps" begin
  p = s1_mech_problem()
  N = near_nullspace(p.U)
  nsteps = 20
  for backend in S1_BACKENDS, (lname, ls) in (("LU", LUSolver()), ("AMG", AMGSolver(:cg; rtol=1e-10, nullspace=N)))
    @testset "$backend, $lname" begin
      nls = NewtonRaphson(ls; maxiter=20, atol=1e-14, rtol=1e-6)
      m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend)
      live = Int[]
      sizehint!(live, nsteps)
      record(post) = (GC.gc(true); GC.gc(true); push!(live, Base.gc_live_bytes()))
      _, flag = solve!(m; stepping=(nsteps=nsteps, maxbisec=0), post=PostProcessor(m, record; is_vtk=false, filepath=mktempdir()))
      @test flag == 1
      @test length(live) == nsteps
      k = nsteps ÷ 2
      growth = live[end] - minimum(live[k:end-1])             # over the second half of the run
      @info "Memory across load steps ($backend, $lname): live heap $(round(live[k] / 2^20, digits=1)) MiB at step $k, growth $(round(growth / 2^10, digits=1)) KiB over the last $(nsteps - k) steps"
      @test growth ≤ 64 * 2^10
    end
  end
end
