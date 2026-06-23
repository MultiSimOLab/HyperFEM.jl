
# Tutorials

The tutorials are designed to help users learn how to simulate hyperelastic materials and solve multiphysic problems using the Finite Element Method (FEM) in Julia with [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl).

HyperFEM.jl is built on top of the [Gridap.jl](https://github.com/gridap/Gridap.jl) ecosystem, providing specialized tools for multiphysics hyperelastic simulations. The tutorials can be found in a dedicated [repository](https://github.com/MultiSimOLab/HyperFEM_tutorials) and demonstrate the core usage of HyperFEM, and these are recommended for new users.

## Get started

1. Clone the repository:
```
git clone https://github.com/MultiSimOLab/HyperFEM_tutorials.git
cd HyperFEM_tutorials
```

2. Open the Julia REPL, type `]` to enter package mode, and activate de environment:
```julia
pkg> activate .
```

3. Install the dependencies:
```julia
pkg> instantiate
```

## Tutorials

The HyperFEM tutorials include a wide range of tutorials, carefully selected to demonstrate the toolbox's capabilities. Each tutorial focuses on a specific type of problem, from basic PDEs to complex multiphysics and optimization scenarios. These examples are ideal for understanding both the theoretical formulation and the practical implementation of FEM simulations in Julia.

* [Example 1: Poisson](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E1_Poisson.jl): Introduces fundamental FEM concepts and demonstrates solving a simple Poisson equation.

* [Example 2: Hyperelastic beam stretching](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E2_Hyperelastic_stretch.jl): Illustrates large deformation analysis of a hyperelastic beam, showcasing material nonlinearity.

* [Example 3: Hyperelastic cylinder (4 fibres model) under internal pressure](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E3_HyperelasticCylinder_4fibres.jl): Demonstrates anisotropic hyperelastic modeling with fiber-reinforced materials under internal loading.

* [Example 4: Electromechanical beam](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E4_ElectroMechanics.jl): Introduces coupled electromechanical simulations, highlighting interactions between mechanical and electrical fields.

* [Example 5: Anisotropic Electromechanical beam](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E5_ElectroMechanics_anisotropic.jl): Shows the effect of anisotropic material behavior in coupled electromechanical problems.

* [Example 6: Hyperelastic contact with third medium](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E6_Hyperelastic_contact_thirdmedium.jl): Covers contact mechanics involving hyperelastic materials interacting with a third body.

* [Example 7 Topology optimization of hyperelastic cantilever](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E7_Topology_hyperelastic_phasefield.jl): Demonstrates optimization techniques applied to hyperelastic structures for design improvement.

* [Example 8: Magnetomechanical beam](https://github.com/MultiSimOLab/HyperFEM_tutorials/blob/main/src/E8_MagnetoMechanical_beam.jl): Illustrates magnetomechanical coupling simulations, integrating magnetic and mechanical field interactions.
