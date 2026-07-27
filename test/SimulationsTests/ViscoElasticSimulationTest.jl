using HyperFEM
using Test


include(projdir("test/data/ViscoElasticSimulation.jl"))
λx, σΓ = visco_elastic_simulation(t_end=2, writevtk=false, verbose=false)
@test σΓ[end] ≈ 21872.5028


include(projdir("test/data/ViscoelasticFastSimulation.jl"))
λx2, σΓ2 = visco_elastic_fast_simulation(t_end=2, writevtk=false, verbose=false)
@test σΓ2[end] ≈ 22152.0463
