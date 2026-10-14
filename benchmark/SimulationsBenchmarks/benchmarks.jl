
# The simulations are defined in test/SimulationsTests, with the CI configuration as default arguments.
# Only the most critical ones are benchmarked, since benchmarks are time consuming.
include(projdir("test/SimulationsTests/ViscoElasticSimulation.jl"))
include(projdir("test/SimulationsTests/StaticMechanicalDirichletSimulation.jl"))
include(projdir("test/SimulationsTests/StaticMechanicalNeumannSimulation.jl"))

SUITE["Simulations"] = BenchmarkGroup()
SUITE["Simulations"]["ViscoElastic"] = @benchmarkable visco_elastic_simulation()
SUITE["Simulations"]["StaticMechanicalDirichlet"] = @benchmarkable static_mechanical_dirichlet_simulation()
SUITE["Simulations"]["StaticMechanicalNeumann"] = @benchmarkable static_mechanical_neumann_simulation()
