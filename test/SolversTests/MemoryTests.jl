# Memory across load steps (requirement of 2026-10-31): the live heap after every load step stays flat once the caches
# exist — no buffer, closure or history grows with the number of steps. Uses the nonlinear mechanics problem and the
# backends of krylov_common.jl (included by KrylovSolversTests.jl).
isdefined(@__MODULE__, :s1_mech_problem) || include("krylov_common.jl")

@testset "Memory flat across load steps" begin
  p = s1_mech_problem()
  for backend in S1_BACKENDS
    @testset "$backend" begin
      nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-14, rtol=1e-6)
      m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend)
      live = Int[]
      sizehint!(live, 16)
      record(post) = (GC.gc(true); push!(live, Base.gc_live_bytes()))
      _, flag = solve!(m; stepping=(nsteps=10, maxbisec=0), post=PostProcessor(m, record; is_vtk=false, filepath=mktempdir()))
      @test flag == 1
      @test length(live) ≥ 10
      growth = live[end] - live[3]                 # after two steps every cache and compiled method exists
      @info "Memory across load steps ($backend): live heap $(round(live[3] / 2^20, digits=1)) MiB at step 3, $(round(growth / 2^10, digits=1)) KiB growth to step $(length(live))"
      @test growth ≤ 0.01 * live[3] + 2^20
    end
  end
end
