using Gridap
using HyperFEM
using LinearAlgebra
using Test

# Each simulation file defines a function whose default arguments are the CI configuration and whose `backend`
# keyword selects who assembles: `:gridap` (Gridap's FEOperator) or `:kernel` (HyperFEMKernels, loaded below).
# Every simulation runs with every backend of TEST_BACKENDS; the reference values are checked for each backend,
# and the results of the other backends are compared with those of :gridap.
#
# TEST_BACKENDS comes from the environment variable HYPERFEM_TEST_BACKENDS (comma separated; default
# "gridap,kernel"), e.g. HYPERFEM_TEST_BACKENDS=gridap for a quick run without the kernel backend.
# The same functions are benchmarked in benchmark/SimulationsBenchmarks/benchmarks.jl
const TEST_BACKENDS = Tuple(Symbol.(strip.(split(get(ENV, "HYPERFEM_TEST_BACKENDS", "gridap,kernel"), ","))))
all(b -> b in (:gridap, :kernel), TEST_BACKENDS) || error("HYPERFEM_TEST_BACKENDS: unknown backend in $TEST_BACKENDS")
:kernel in TEST_BACKENDS && @eval using HyperFEMKernels      # activates the extension HyperFEMKernelsExt

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

Each backend against :gridap: `‖f(r_backend) - f(r_gridap)‖ ≤ rtol ‖f(r_gridap)‖`. Same problem, same Newton
histories; what differs is the rounding of the assembly and of the LU.
"""
function agree(results, f; rtol=1e-10)
  haskey(results, :gridap) || return
  ref = f(results[:gridap])
  for (b, r) in results
    b === :gridap && continue
    @testset "$b vs gridap" begin
      @test norm(f(r) - ref) ≤ rtol * norm(ref)
    end
  end
end

@testset "SimulationsTests (backends: $(join(TEST_BACKENDS, ", ")))" begin

  @testset "ViscoElastic" begin
    r = run_backends(b -> visco_elastic_simulation(; backend=b))
    for (b, (λx, σΓ)) in r
      @testset "$b" begin
        @test σΓ[end] ≈ 22152.0463
      end
    end
    agree(r, x -> x[2])
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
    for (b, (φh, uh)) in r
      @testset "$b" begin
        @test norm(uh[1]) ≈ 0.01492384116
        @test norm(φh[1]) ≈ 0.00023052926
      end
    end
    # thin Q2 plate, ill conditioned: identical Newton histories, the LU rounding shows at ~2e-10 in u
    agree(r, x -> x[2][1]; rtol=1e-8)
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
        @test isapprox(norm(r[b]), 2.8132015601158087, rtol=b === :gridap ? 1e-14 : 1e-10)
        @test isapprox(ri[b], r[b], rtol=1e-12)
      end
    end
    agree(r, identity)
  end

  @testset "TurekNavierStokes" begin
    # DFG 2D-1 (Re = 20), P1-P1 SUPG/PSPG/LSIC, BDF2 Oseen, StaticLinearModel re-assembled at every step
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

end
