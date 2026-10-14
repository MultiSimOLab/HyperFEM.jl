using Gridap
using HyperFEM
using Test

@testset "SimulationsTests" begin
   
    include("ViscoElasticSimulationTest.jl")

    include("StaticMechanicalDirichletTest.jl")

    include("StaticMechanicalNeumannTest.jl")

    include("StaggeredElectroMechanicalTest.jl")

    include("StaggeredViscoElectricTest.jl")

    include("Stokes.jl")

    include("BoundaryConditions.jl")

end
