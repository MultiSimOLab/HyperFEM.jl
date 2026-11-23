# Tutorials

The tutorials are designed to help users learn how to simulate hyperelastic materials and solve multiphysic problems using the Finite Element Method (FEM) in Julia with [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl).

HyperFEM.jl is built on top of the [Gridap.jl](https://github.com/gridap/Gridap.jl) ecosystem, providing specialized tools for multiphysics hyperelastic simulations. The tutorials live in a dedicated [repository](https://github.com/MultiSimOLab/HyperFEMTutorials.jl), with their own [documentation](https://multisimolab.github.io/HyperFEMTutorials.jl/dev/), and demonstrate the core usage of HyperFEM. They are recommended for new users.

## Get started

* Clone the repository:
```
git clone https://github.com/MultiSimOLab/HyperFEMTutorials.jl.git
cd HyperFEMTutorials.jl
```
* Open the Julia REPL, type `]` to enter package mode, and activate the environment:
```julia
pkg> activate .
```
* Install the dependencies:
```julia
pkg> instantiate
```

## Tutorials

Each tutorial focuses on a specific type of problem, from basic PDEs to complex multiphysics problems, linear solvers and dynamics. The list below is generated from the tutorials repository when this documentation is built.
