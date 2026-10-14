using HyperFEM
using Test


include(projdir("test/data/ViscoElasticSimulation.jl"))
λx, σΓ = visco_elastic_simulation(t_end=2, writevtk=false, verbose=false)
@test σΓ[end] ≈ 22152.0463
