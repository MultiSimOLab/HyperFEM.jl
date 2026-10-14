using Gridap
using HyperFEM
using Test

# Each simulation file defines a function whose default arguments are the CI configuration.
# The same functions are benchmarked in benchmark/SimulationsBenchmarks/benchmarks.jl
include("ViscoElasticSimulation.jl")
include("StaticMechanicalDirichletSimulation.jl")
include("StaticMechanicalNeumannSimulation.jl")
include("StaggeredElectroMechanicalSimulation.jl")
include("StaggeredViscoElectricSimulation.jl")
include("StokesSimulation.jl")
include("MeshMovementSimulation.jl")

@testset "SimulationsTests" begin

  @testset "ViscoElastic" begin
    λx, σΓ = visco_elastic_simulation()
    @test σΓ[end] ≈ 22152.0463
  end

  @testset "StaticMechanicalDirichlet" begin
    x = static_mechanical_dirichlet_simulation()
    @test norm(x) ≈ 0.27148722276
  end

  @testset "StaticMechanicalNeumann" begin
    x = static_mechanical_neumann_simulation()
    @test norm(x) ≈ 1.000148588846
  end

  @testset "StaggeredElectroMechanical" begin
    φh, uh = staggered_electro_mechanical_simulation()
    @test norm(uh[1]) ≈ 0.01492384116
    @test norm(φh[1]) ≈ 0.00023052926
  end

  @testset "StaggeredViscoElectric" begin
    t, uz = staggered_visco_electric_simulation()
    @test uz[end] ≈ 0.002914376983
    @test norm(uz) ≈ 0.004558785342
  end

  @testset "Stokes" begin
    x = stokes_simulation()
    @test norm(x) ≈ 4199.925167995936
  end

  @testset "MeshMovement" begin
    x = mesh_movement_simulation()
    @test isapprox(norm(x), 2.8132015601158087, rtol=1e-14)
  end

end
