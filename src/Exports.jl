
macro publish(mod, name)
  quote
    using HyperFEM.$mod: $name
    export $name
  end
end

@publish TensorAlgebra (*)
@publish TensorAlgebra (×ᵢ⁴)
@publish TensorAlgebra (⊗₁₂³)
@publish TensorAlgebra (⊗₁₃²)
@publish TensorAlgebra (⊗₁²³)
@publish TensorAlgebra (⊗₁₃²⁴)
@publish TensorAlgebra (⊗₁₂³⁴)
@publish TensorAlgebra (⊗₁²)
@publish TensorAlgebra logreg
@publish TensorAlgebra Box
@publish TensorAlgebra Ellipsoid
@publish TensorAlgebra I9
@publish TensorAlgebra Tensorize


@publish PhysicalModels DerivativeStrategy
@publish PhysicalModels LinearElasticity3D
@publish PhysicalModels LinearElasticity2D
@publish PhysicalModels Yeoh3D
@publish PhysicalModels Gent2D
@publish PhysicalModels NeoHookean3D
@publish PhysicalModels IncompressibleNeoHookean3D
@publish PhysicalModels IncompressibleNeoHookean2D
@publish PhysicalModels IncompressibleNeoHookean2D_CV
@publish PhysicalModels IncompressibleNeoHookean3D_2dP
@publish PhysicalModels VolumetricEnergy
@publish PhysicalModels MooneyRivlin3D
@publish PhysicalModels MooneyRivlin2D
@publish PhysicalModels NonlinearMooneyRivlin3D
@publish PhysicalModels NonlinearMooneyRivlin2D
@publish PhysicalModels NonlinearMooneyRivlin2D_CV
@publish PhysicalModels NonlinearNeoHookean_CV
@publish PhysicalModels NonlinearMooneyRivlin_CV
@publish PhysicalModels NonlinearIncompressibleMooneyRivlin2D_CV
@publish PhysicalModels EightChain
@publish PhysicalModels TransverseIsotropy3D
@publish PhysicalModels TransverseIsotropy2D
@publish PhysicalModels ThermalModel
@publish PhysicalModels IdealDielectric
@publish PhysicalModels Magnetic
@publish PhysicalModels IdealMagnetic
@publish PhysicalModels IdealMagnetic2D
@publish PhysicalModels HardMagnetic
@publish PhysicalModels HardMagnetic2D
@publish PhysicalModels ElectroMechModel
@publish PhysicalModels ThermoElectroMechModel
@publish PhysicalModels ThermoMechModel
@publish PhysicalModels ThermoMech_Bonet
@publish PhysicalModels ThermoMech_EntropicPolyconvex
@publish PhysicalModels FlexoElectroModel
@publish PhysicalModels ThermoElectroMech_Bonet
@publish PhysicalModels ThermoElectroMech_Govindjee
@publish PhysicalModels ThermoElectroMech_PINNs
@publish PhysicalModels MagnetoMechModel
@publish PhysicalModels ARAP2D
@publish PhysicalModels ARAP2D_regularized
@publish PhysicalModels HessianRegularization
@publish PhysicalModels Hessian∇JRegularization
@publish PhysicalModels ViscousIncompressible
@publish PhysicalModels GeneralizedMaxwell
@publish PhysicalModels HGO_4Fibers
@publish PhysicalModels HGO_1Fiber

@publish PhysicalModels Mechano
@publish PhysicalModels Thermo
@publish PhysicalModels Electro
@publish PhysicalModels Magneto
@publish PhysicalModels ThermoMechano
@publish PhysicalModels ElectroMechano
@publish PhysicalModels MagnetoMechano
@publish PhysicalModels ThermoElectro
@publish PhysicalModels FlexoElectro
@publish PhysicalModels ThermoElectroMechano
@publish PhysicalModels EnergyInterpolationScheme
@publish PhysicalModels update_state!
@publish PhysicalModels Kinematics
@publish PhysicalModels Solid
@publish PhysicalModels KinematicModel
@publish PhysicalModels EvolutiveKinematics
@publish PhysicalModels get_Kinematics
@publish PhysicalModels getIsoInvariants

@publish PhysicalModels derivatives
@publish PhysicalModels ThermalLaw
@publish PhysicalModels VolumetricLaw
@publish PhysicalModels DeviatoricLaw
@publish PhysicalModels InterceptLaw
@publish PhysicalModels TrigonometricLaw

@publish PhysicalModels SecondPiola
@publish PhysicalModels Dissipation
@publish PhysicalModels initialize_state
@publish PhysicalModels update_time_step!

@publish WeakForms residual
@publish WeakForms jacobian
@publish WeakForms mass_term

@publish ComputationalModels  DirichletBC
@publish ComputationalModels  NeumannBC
@publish ComputationalModels  get_Neumann_dΓ
@publish ComputationalModels  residual_Neumann
@publish ComputationalModels  NothingBC
@publish ComputationalModels  MultiFieldBC
@publish ComputationalModels  SingleFieldTC
@publish ComputationalModels  MultiFieldTC
@publish ComputationalModels  TrialFESpace
@publish ComputationalModels  get_state
@publish ComputationalModels  get_measure
@publish ComputationalModels  get_spaces
@publish ComputationalModels  get_trial_space
@publish ComputationalModels  get_test_space
@publish ComputationalModels  get_assemblers
@publish ComputationalModels  StaticNonlinearModel
@publish ComputationalModels  DynamicNonlinearModel
@publish ComputationalModels  StaticLinearModel
@publish ComputationalModels  solve!
@publish ComputationalModels  dirichlet_preconditioning!
@publish ComputationalModels  GmshDiscreteModel
@publish ComputationalModels  updateBC!
@publish ComputationalModels  PostProcessor
@publish ComputationalModels  vtk_save
@publish ComputationalModels  get_pvd
@publish ComputationalModels  PostMetrics
@publish ComputationalModels  StaggeredModel
@publish ComputationalModels  Cauchy
@publish ComputationalModels  Piola
@publish ComputationalModels  Jacobian
@publish ComputationalModels  Entropy
@publish ComputationalModels  D0
@publish ComputationalModels  reset!
@publish ComputationalModels  interpolate_L2_tensor
@publish ComputationalModels  interpolate_L2_vector
@publish ComputationalModels  interpolate_L2_scalar
@publish ComputationalModels  DirichletCoupling
@publish ComputationalModels  evaluate!
@publish ComputationalModels  InterpolableBC
@publish ComputationalModels  InterpolableBC!
@publish ComputationalModels TrialFESpace! # Exporting internal function of Gridap

# Note: the files FaceLabeling, CartesianTags and Evolution functions should be moved to a module different than ComputationalModels
@publish ComputationalModels add_tag_from_vertex_filter!

@publish Solvers IterativeSolver
@publish Solvers Newton_RaphsonSolver
@publish Solvers Injectivity_Preserving_LS
@publish Solvers Roman_LS
@publish Solvers update_cellstate!

@publish IO setupfolder
@publish IO projdir
@publish IO stem
@publish IO MockPVD
@publish IO mockpvd
