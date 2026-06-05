using Gridap.TensorValues
using HyperFEM.PhysicalModels
using HyperFEM.TensorAlgebra
using ForwardDiff


const ∇φ = VectorValue(1.0:3.0...)
const ∇u = TensorValue(1.0:9.0...) * 1e-3
const ∇un = TensorValue(1.0:9.0...) * 5e-4


@testset "IdealDielectric" begin
  model = IdealDielectric(ε=4.0*8.85e-12)
  Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂FF, ∂Ψ∂EF, ∂Ψ∂EE = model()
  E0 = VectorValue(rand(3)) * 6000 / 0.001
  F1 = I3 + 0.1*TensorValue(rand(9)...)
  @test get_array(∂Ψ∂F(F1,E0))  ≈ ForwardDiff.gradient(Fi -> Ψ(Fi, get_array(E0)), get_array(F1))
  @test get_array(∂Ψ∂E(F1,E0))  ≈ ForwardDiff.gradient(Ei -> Ψ(get_array(F1), Ei), get_array(E0))
  @test get_array(∂Ψ∂FF(F1,E0)) ≈ ForwardDiff.jacobian(Fi -> ∂Ψ∂F(Fi, get_array(E0)), get_array(F1))
  @test get_array(∂Ψ∂EF(F1,E0)) ≈ ForwardDiff.jacobian(Fi -> ∂Ψ∂E(Fi, get_array(E0)), get_array(F1))
  @test get_array(∂Ψ∂EF(F1,E0)) ≈ ForwardDiff.jacobian(Ei -> ∂Ψ∂F(get_array(F1), Ei), get_array(E0))'
  @test get_array(∂Ψ∂EE(F1,E0)) ≈ ForwardDiff.jacobian(Ei -> ∂Ψ∂E(get_array(F1), Ei), get_array(E0))
end


@testset "Electro+4*HGO_1Fiber" begin
  c1  =  [0.6639232500447778, 0.5532987701062146, 0.9912576142028674, 0.4951942011240962]
  c2  =  [0.800583033264982, 0.3141082734275339, 0.8063905248474006, 0.5850486948450955]
  M1  =  [ 0.36799630150742724, 0.8353476258002335, 0.57704047269419]
  M1  =  VectorValue(M1/norm(M1))
  M2  =  [ 0.3857610953527303, 0.024655018338846868, 0.6133770006613235]
  M2  =  VectorValue(M2/norm(M2))
  M3  =  [ 0.4516424464747618, 0.6609741557924332, 0.8441070681368911]
  M3  =  VectorValue(M3/norm(M3))
  M4  =  [0.7650638541897623, 0.8268625401770648, 0.3103412304991431]
  M4  =  VectorValue(M4/norm(M4))
  
  model1 = HGO_1Fiber(c1=c1[1], c2=c2[1])
  model2 = HGO_1Fiber(c1=c1[2], c2=c2[2])
  model3 = HGO_1Fiber(c1=c1[3], c2=c2[3])
  model4 = HGO_1Fiber(c1=c1[4], c2=c2[4])
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelelectro=modelMR+[model1 model2 model3 model4]+modelID
 

  Ψ, ∂Ψ∂F, ∂Ψ∂F∂F = modelelectro()
 

  Ke=Kinematics(Electro,Solid)
  E = get_Kinematics(Ke)
  K=Kinematics(Mechano,Solid)
  F, _, _  = get_Kinematics(K)

  @test Ψ(F(∇u),E(∇φ) ,(M1,M2,M3,M4)) == -27.513663654827152
  @test isapprox(norm(∂Ψ∂F(F(∇u),E(∇φ) ,(M1,M2,M3,M4))) ,  47.45420724735932,rtol=1e-14)
  @test isapprox(norm(∂Ψ∂F∂F(F(∇u),E(∇φ) ,(M1,M2,M3,M4))), 14.707913034885005,rtol=1e-14)
end



@testset "ElectroMechano" begin
  #  Memory estimate: 0 bytes, allocs estimate: 0.
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelelectro =modelMR+modelID
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro()
  Ke=Kinematics(Electro,Solid)
  E = get_Kinematics(Ke)
  K=Kinematics(Mechano,Solid)
  F, _, _  = get_Kinematics(K)

  @test Ψ(F(∇u), E(∇φ)) == -27.514219755428428
  @test norm(∂Ψu(F(∇u), E(∇φ))) == 47.42294370458073
  @test norm(∂Ψφ(F(∇u), E(∇φ))) == 14.707913034885005
  @test norm(∂Ψuu(F(∇u), E(∇φ))) == 131.10069227603947
  @test norm(∂Ψφu(F(∇u), E(∇φ))) == 39.03656526472973
  @test norm(∂Ψφφ(F(∇u), E(∇φ))) == 6.964428025226914
end


@testset "FlexoElectroMechanics" begin
#  Memory estimate: 0 bytes, allocs estimate: 0.
  # Constitutive models
  ∇umacro = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) * 1e-2
  ∇u1 = 1e-1 * TensorValue(1, 2, 3, 4, 5, 6, 7, 8, 9)
  Emacro = VectorValue(0.0, 0.0, sqrt((1.0 + 5.0) / (1.0 + 5.0)) * 0.1)
  A = TensorValue{3,9,Float64,27}(0.0013981268088158305, 0.0008195783555664171,
    0.0016562357569609649, 0.0008406006468943406, 0.0009224862278332126, 0.001155322042969417,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0008406006468943406,
    0.0009224862278332126, 0.001155322042969417, 0.00034502469077903774, 0.00021859521770246592,
    0.0017683239822952042, 0.0009471782270005929, 0.001800950730156155, 0.0009587801251013468,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0009471782270005929,
    0.001800950730156155, 0.0009587801251013468, 0.0008421896546088605, 0.0007114140805416631,
    0.001245006227831607)
  Kin_mec = EvolutiveKinematics(Mechano; F=(t) -> ((∇u1, x) -> ∇u1 + one(∇u1) + t * ∇umacro + t * (A ⊙ x)))
  Kin_elec = EvolutiveKinematics(Electro; E=(t) -> ((∇φ) -> -∇φ + t * Emacro))

  physmec = MooneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0)
  physelec = IdealDielectric(ε=1.0)
  physmodel = FlexoElectroModel(mechano=physmec, electro=physelec, κ=1000.0)

  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ, Φ = physmodel(1.0)

  F, _, _ = get_Kinematics(Kin_mec; Λ=1.0)
  E = get_Kinematics(Kin_elec; Λ=1.0)
  X = VectorValue(2.4, 1.9, 3.3)

  @test (Ψ(F(∇u1, X), E(∇φ))) == 13.408299698687056
  @test norm(∂Ψu(F(∇u1, X), E(∇φ))) == 58.375248703633474
  @test norm(∂Ψφ(F(∇u1, X), E(∇φ))) == 1.2365693126167825
  @test norm(∂Ψuu(F(∇u1, X), E(∇φ))) == 208.40589433833898
  @test norm(∂Ψφφ(F(∇u1, X), E(∇φ))) == 3.8963298254031042
  @test norm(∂Ψφu(F(∇u1, X), E(∇φ))) == 5.910650247536949
end


@testset "ViscoElectricModel" begin
#     157 μs        Histogram: log(frequency) by time        391 μs <
#  Memory estimate: 240.03 KiB, allocs estimate: 3069.
  hyper_elastic = NeoHookean3D(λ=1000., μ=10.)
  short_term = IncompressibleNeoHookean3D(μ=5., λ=0.)
  viscous_branch1 = ViscousIncompressible(short_term, τ=6.)
  visco_elastic = GeneralizedMaxwell(hyper_elastic, viscous_branch1)
  dielectric = IdealDielectric(ε=1.0)
  model =dielectric+visco_elastic
  Ke=Kinematics(Electro,Solid)
  E  = get_Kinematics(Ke)
  K=Kinematics(Mechano,Solid)
  F, _, _  = get_Kinematics(K)
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  Uvn *= det(Uvn)^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  update_time_step!(model, 0.01)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = model()
  @test norm(∂Ψu(F(∇u), E(∇φ), F(∇un), Avn)) ≈ 25.049301121178615
  @test norm(∂Ψuu(F(∇u), E(∇φ), F(∇un), Avn)) ≈ 3110.7607787445168
end


@testset "ViscoElectricModel 2-branch" begin
  hyper_elastic = NeoHookean3D(λ=1000., μ=10.)
  short_term = IncompressibleNeoHookean3D(μ=5., λ=0.)
  viscous_branch1 = ViscousIncompressible(short_term, τ=6.)
  viscous_branch2 = ViscousIncompressible(short_term, τ=60.)
  visco_elastic = GeneralizedMaxwell(hyper_elastic, viscous_branch1, viscous_branch2)
  dielectric = IdealDielectric(ε=1.0)
  model = dielectric+visco_elastic
  Ke=Kinematics(Electro,Solid)
  E = get_Kinematics(Ke)
  K=Kinematics(Mechano,Solid)
  F, _, _  = get_Kinematics(K)
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  Uvn *= det(Uvn)^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  update_time_step!(model, 0.01)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = model()
  @test norm(∂Ψu(F(∇u), E(∇φ), F(∇un), Avn, Avn)) ≈ 25.102080194257017
  @test norm(∂Ψuu(F(∇u), E(∇φ), F(∇un), Avn, Avn)) ≈ 3110.9722775475557
end
