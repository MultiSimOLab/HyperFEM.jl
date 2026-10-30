using Gridap
using HyperFEM
using LinearAlgebra
using Test

# Each simulation file defines a function whose default arguments are the CI configuration and whose `backend`
# keyword selects who assembles: `:lazy` (Gridap's lazy evaluation, FEOperator) or `:compiled` (the weak form
# compiled into fused kernels by HyperFEMKernels, loaded below).
# Every simulation runs with every backend of TEST_BACKENDS; the reference values are checked for each backend,
# and the results of the other backends are compared with those of :lazy.
#
# TEST_BACKENDS comes from the environment variable HYPERFEM_TEST_BACKENDS (comma separated; default
# "lazy,compiled"), e.g. HYPERFEM_TEST_BACKENDS=lazy for a quick run without HyperFEMKernels.
# The same functions are benchmarked in benchmark/SimulationsBenchmarks/benchmarks.jl
const TEST_BACKENDS = Tuple(Symbol.(strip.(split(get(ENV, "HYPERFEM_TEST_BACKENDS", "lazy,compiled"), ","))))
all(b -> b in (:lazy, :compiled), TEST_BACKENDS) || error("HYPERFEM_TEST_BACKENDS: unknown backend in $TEST_BACKENDS")
:compiled in TEST_BACKENDS && @eval using HyperFEMKernels      # activates the extension HyperFEMKernelsExt

include("ViscoElasticSimulation.jl")
include("StaticMechanicalDirichletSimulation.jl")
include("StaticMechanicalNeumannSimulation.jl")
include("StaggeredElectroMechanicalSimulation.jl")
include("StaggeredViscoElectricSimulation.jl")
include("StokesSimulation.jl")
include("MeshMovementSimulation.jl")
include("TurekNavierStokesSimulation.jl")
include("HMREVacuumSimulation.jl")
include("ThirdMediumContactSimulation.jl")

"Runs `simulate(backend)` for every backend of TEST_BACKENDS: Dict backend => result."
run_backends(simulate) = Dict(b => simulate(b) for b in TEST_BACKENDS)

"""
    agree(results, f; rtol=1e-10)

Each backend against :lazy: `‖f(r_backend) - f(r_lazy)‖ ≤ rtol ‖f(r_lazy)‖`. Same problem, same Newton
histories; what differs is the rounding of the assembly and of the LU.
"""
function agree(results, f; rtol=1e-10)
  haskey(results, :lazy) || return
  ref = f(results[:lazy])
  for (b, r) in results
    b === :lazy && continue
    @testset "$b vs lazy" begin
      @test norm(f(r) - ref) ≤ rtol * norm(ref)
    end
  end
end

# A Newton solver that fails at chosen attempts (the time loop must bisect and go on from the last accepted
# step, the internal variables untouched)
mutable struct FailingNewton{N<:NewtonRaphson,L} <: Gridap.Algebra.NonlinearSolver
  nls::N
  fails::Vector{Int}
  calls::Int
  ls::L
  linesearch::Any
  log::Any
end
FailingNewton(nls, fails) = FailingNewton(nls, fails, 0, nls.ls, nls.linesearch, nls.log)
HyperFEM.ComputationalModels.instantiate_caches(x, s::FailingNewton, op) = HyperFEM.ComputationalModels.instantiate_caches(x, s.nls, op)
function Gridap.Algebra.solve!(x::AbstractVector, s::FailingNewton, op::Gridap.Algebra.NonlinearOperator, cache; r_reference=0.0)
  s.calls += 1
  s.calls in s.fails && return cache
  solve!(x, s.nls, op, cache; r_reference)
end
HyperFEM.ComputationalModels.nls_converged(s::FailingNewton) = !(s.calls in s.fails) && HyperFEM.Solvers.converged(s.nls)
HyperFEM.ComputationalModels.nls_message(s::FailingNewton, msg) = nothing
HyperFEM.ComputationalModels.nls_verbosity(s::FailingNewton) = 0
HyperFEM.ComputationalModels.nls_reset!(s::FailingNewton) = HyperFEM.ComputationalModels.nls_reset!(s.nls)
HyperFEM.ComputationalModels.nls_iterations(s::FailingNewton) = HyperFEM.ComputationalModels.nls_iterations(s.nls)

@testset "SimulationsTests (backends: $(join(TEST_BACKENDS, ", ")))" begin

  @testset "ViscoElastic" begin
    r = run_backends(b -> visco_elastic_simulation(; backend=b))
    for (b, (λx, σΓ)) in r
      @testset "$b" begin
        @test σΓ[end] ≈ 22152.0463
      end
    end
    agree(r, x -> x[2])
    # the transient driver (QuasiStatic scheme, the internal variables updated through the hooks, both
    # storages of the state variables) reproduces the load-step loop of the static driver
    rs = run_backends(b -> visco_elastic_simulation(; backend=b, driver=:static))
    for b in TEST_BACKENDS
      @testset "transient vs static ($b)" begin
        @test norm(r[b][2] - rs[b][2]) ≤ 1e-10 * norm(rs[b][2])
        @test norm(r[b][1] - rs[b][1]) ≤ 1e-10 * norm(rs[b][1])
      end
    end
    # a forced failure of the fifth attempt: the loop bisects from the fourth step, the internal variables
    # and the state of that step untouched, and goes on (the increment grows back); the response is that of
    # the uninterrupted run up to the time discretisation of the bisected step
    for b in TEST_BACKENDS
      @testset "forced bisection ($b)" begin
        nls = FailingNewton(NewtonRaphson(LUSolver(); maxiter=20, atol=1e-6, rtol=1e-6, verbose=0), [5])
        stepping = TimeStepping(; Δt=0.05, t_end=2.0, maxbisec=1, grow=2.0)
        λf, σf = visco_elastic_simulation(; backend=b, nls, stepping)
        @test length(σf) == length(r[b][2]) + 1
        @test σf[1:4] == r[b][2][1:4]
        @test abs(σf[end] - r[b][2][end]) ≤ 1e-3 * abs(r[b][2][end])
      end
    end
  end

  @testset "StaticMechanicalDirichlet" begin
    r = run_backends(b -> static_mechanical_dirichlet_simulation(; backend=b))
    for (b, x) in r
      @testset "$b" begin
        @test norm(x) ≈ 0.27148722276
      end
    end
    agree(r, identity)
  end

  @testset "StaticMechanicalNeumann" begin
    r = run_backends(b -> static_mechanical_neumann_simulation(; backend=b))
    for (b, x) in r
      @testset "$b" begin
        @test norm(x) ≈ 1.000148588846
      end
    end
    agree(r, x -> x[1])
  end

  @testset "StaggeredElectroMechanical" begin
    r = run_backends(b -> staggered_electro_mechanical_simulation(; backend=b))
    for (b, ((φh, uh), flag)) in r
      @testset "$b" begin
        @test flag == 1
        @test norm(uh) ≈ 0.01492384116
        @test norm(φh) ≈ 0.00023052926
      end
    end
    # thin Q2 plate, ill conditioned: identical Newton histories, the LU rounding shows at ~2e-10 in u
    agree(r, x -> x[1][2]; rtol=1e-8)
    agree(r, x -> x[1][1])
  end

  @testset "StaggeredViscoElectric" begin
    r = run_backends(b -> staggered_visco_electric_simulation(; backend=b))
    for (b, (t, uz)) in r
      @testset "$b" begin
        @test uz[end] ≈ 0.002914376983
        @test norm(uz) ≈ 0.004558785342
      end
    end
    agree(r, x -> x[2])
  end

  @testset "Stokes" begin
    r = run_backends(b -> stokes_simulation(; backend=b))
    for (b, x) in r
      @testset "$b" begin
        @test norm(x) ≈ 4199.925167995936
      end
    end
    # saddle point: the kernel's sparsity pattern stores the (empty) p-p block, which changes the pivoting of
    # the LU; matrix and vector agree to 1e-16, the solutions to ~1e-7
    agree(r, identity; rtol=1e-6)
  end

  @testset "MeshMovement" begin
    r = run_backends(b -> mesh_movement_simulation(; backend=b))
    ri = run_backends(b -> mesh_movement_simulation(; backend=b, interface_field=true))   # stages living only on the interface
    for b in TEST_BACKENDS
      @testset "$b" begin
        @test isapprox(norm(r[b]), 2.8132015601158087, rtol=b === :lazy ? 1e-14 : 1e-10)
        @test isapprox(ri[b], r[b], rtol=1e-12)
      end
    end
    agree(r, identity)
  end

  @testset "TurekNavierStokes" begin
    # DFG 2D-1 (Re = 20), P1-P1 SUPG/PSPG/LSIC, BDF2 Oseen on TransientModel (one linear solve per step)
    r = run_backends(b -> turek_navier_stokes_simulation(; backend=b))
    for (b, s) in r
      @testset "$b" begin
        @test isapprox(norm(s.x), 13.67687986714, rtol=1e-8)
        @test isapprox(s.cD, 5.5795, rtol=0.03)      # DFG reference values (fine meshes, P2-P1)
        @test isapprox(s.Δp, 0.1175, rtol=0.02)
      end
    end
    agree(r, s -> s.x)
    agree(r, s -> s.cD_history)
  end

  @testset "HMREVacuum" begin
    # staggered magneto-mechanics with the vacuum (three problems, Coupling, Maxwell traction), φbot = 520 V, 4 steps
    r = run_backends(b -> hmre_vacuum_simulation(; backend=b))
    for (b, s) in r
      @testset "$b" begin
        @test isapprox(norm(s.φ), 2.0940145799e4, rtol=1e-6)
        @test isapprox(norm(s.u_solid), 1.0426282896e-1, rtol=1e-6)
        @test isapprox(norm(s.u_vacuum), 2.4014448097e-1, rtol=1e-6)
      end
    end
    agree(r, s -> s.φ)
    agree(r, s -> s.u_solid)
    agree(r, s -> s.u_vacuum)
  end

  @testset "ThirdMediumContact" begin
    # piston benchmark (Faltus, Amato, Horák 2026): double-clamped beam buckling against a third medium, weighted jump
    # of ∇u on the skeleton of the third medium, injectivity-preserving line search; 15 increments up to u_y = 1.5 d
    r = run_backends(b -> thirdmedium_contact_simulation(; backend=b))
    for (b, s) in r
      @testset "$b" begin
        @test length(s.Λ) == 15 && s.Λ[end] ≈ 1.0
        @test isapprox(norm(s.u), 2.479372626098216, rtol=1e-8)
        @test isapprox(s.fpiston[end], 7.039427782245141, rtol=1e-8)
        @test isapprox(s.fclamp[end], 7.039054101423267, rtol=1e-8)
        @test s.fpiston[10] < 6.6 && s.fpiston[11] > 7.0        # buckling between u_y = 1.0 d and 1.1 d
        @test all(abs.(s.fpiston .- s.fclamp) .< 1e-3 .* s.fpiston)   # equilibrium of the beam
      end
    end
    agree(r, s -> s.u)
    agree(r, s -> s.fpiston)
  end

  # Piston with the configuration of the note "Solver improvements for third-medium contact" (2026-10-25):
  # barrier p = 2, w ≡ 1, λ = 6μ, exact injectivity step with J_min, Armijo, hybrid Newton; fixed ΔΛ0 (the step
  # growth belongs to the driver refactor, R2), up to u_y ≈ 3.5 d, past the first snap. Local only (several
  # minutes): HYPERFEM_TEST_LONG=1. The full loading to 15 d against Horák's curve is R2's acceptance test.
  if get(ENV, "HYPERFEM_TEST_LONG", "0") == "1"
    @testset "ThirdMediumContact, note configuration to 3.5 d (local)" begin
      s = thirdmedium_contact_simulation(; backend=:lazy, uy_max=0.35, nsteps=235, maxbisec=5, ProjectDirichlet=true,
                                           weight=false, kb=1.0, pvol=2, λsolid=6.0, hybrid=true, armijo=true,
                                           jmin=1e-6, maxiter=500, verbose=1)
      fps = s.fpiston .* (s.fcr / s.fcr_plane_strain)              # Horák's normalisation (plane strain)
      @test s.flag == 1 && s.Λ[end] ≈ 1.0
      imax = argmax(fps)
      @test s.uy_over_d[imax] < 3.4 && fps[end] < fps[imax]          # the first snap happened before 3.5 d
      @test all(abs.(s.fpiston .- s.fclamp) .< 1e-2 .* abs.(s.fpiston))   # equilibrium of the beam
      for (x, ref) in ((1.0, 1.077), (2.0, 1.306), (3.0, 3.434), (3.5, 2.676))   # Horák, Fig. 3b (P2 in the paper)
        i = argmin(abs.(s.uy_over_d .- x))
        println("  u_y/d = $(round(s.uy_over_d[i], digits=3))  f_piston/f_cr = $(round(fps[i], digits=3))  (Horák $ref)")
      end
      println("  Newton: $(s.nls.log.niters) iterations, $(s.nls.log.nswitches) switches, $(round(100 * s.nls.log.nprojected / max(s.nls.log.niters, 1), digits=1)) % projected")
    end

    # The full benchmark (Faltus, Amato, Horák 2026, Fig. 3b): piston to 15 d with the growth of the increment
    # (ΔΛ0 = 1/1000, ×1.5 after an easy step, 5 consecutive bisections), the acceptance run of the driver
    # refactor R2. P1 mesh 1: the beam locks in bending (plateaus above Horák's P2 curve), so the comparison
    # is printed, not asserted; what is checked is that the run reaches 15 d through the snaps.
    @testset "ThirdMediumContact, full piston to 15 d (local)" begin
      horak = JSON.parsefile(joinpath(@__DIR__, "..", "models", "horak2026_piston_f_vs_uy.json"))["piston"]
      hx, hy = first.(horak), last.(horak)
      t = @elapsed s = thirdmedium_contact_simulation(; backend=:lazy, uy_max=1.5, nsteps=1000, maxbisec=5, grow=1.5,
                                                      ProjectDirichlet=true, weight=false, kb=1.0, pvol=2, λsolid=6.0,
                                                      hybrid=true, armijo=true, jmin=1e-6, maxiter=500, verbose=1)
      fps = s.fpiston .* (s.fcr / s.fcr_plane_strain)
      @test s.flag == 1 && s.Λ[end] ≈ 1.0
      @test all(isfinite, fps)
      # equilibrium of the beam between the piston and the clamp while the beam is free of the box; later the
      # third medium transmits part of the load to the walls of the box
      eq = abs.(s.fpiston .- s.fclamp) .< 1e-2 .* abs.(s.fpiston)
      @test all(eq[s.uy_over_d .<= 3.5])
      ieq = findfirst(.!eq)
      println("  $(length(s.Λ)) increments, $(s.nbisect) bisections, $(round(t, digits=0)) s; piston and clamp forces " *
              (ieq === nothing ? "in equilibrium throughout" : "depart from each other from u_y/d = $(round(s.uy_over_d[ieq], digits=2))"))
      println("  Newton: $(s.nls.log.niters) iterations, $(s.nls.log.nswitches) switches, $(round(100 * s.nls.log.nprojected / max(s.nls.log.niters, 1), digits=1)) % projected")
      println("  u_y/d   f/f_cr (P1 mesh 1)   Horák (P2)")
      for x in 1.0:1.0:15.0
        i = argmin(abs.(s.uy_over_d .- x))
        j = argmin(abs.(hx .- x))
        println("  $(lpad(x, 4))    $(lpad(round(fps[i], digits=3), 8))             $(round(hy[j], digits=3))")
      end
    end
  end

end
