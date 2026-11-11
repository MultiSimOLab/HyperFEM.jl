using HyperFEM
using Gridap
using Test

@testset "Solvers" begin

  @time begin
    include("NewtonRaphsonTests.jl")
  end

  @time begin
    include("LinearSolversTests.jl")
  end

  @time begin
    include("AllocationTests.jl")
  end

  @time begin
    include("LineSearchTests.jl")
  end

  @time begin
    include("EnergyTests.jl")
  end

  @time begin
    include("HybridTests.jl")
  end

  @time begin
    include("VerboseTests.jl")
  end

  @time begin
    include("KrylovSolversTests.jl")
  end

  @time begin
    include("FieldSplitTests.jl")
  end

  @time begin
    include("MemoryTests.jl")
  end

  # mesh hierarchy by uniform refinement of Gmsh meshes (S4-B step 0): Gridap + GridapGmsh only
  @time begin
    include("RefinementTests.jl")
  end
  @time begin
    include("MeshHierarchyTests.jl")
  end
  @time begin
    include("HMultigridTests.jl")
  end
  @time begin
    include("FieldSplitSolverTests.jl")
  end

  # matrix-free tangent (S3): needs HyperFEMKernels, as the compiled backend of the Krylov batteries
  if :compiled in Symbol.(strip.(split(get(ENV, "HYPERFEM_TEST_BACKENDS", Base.find_package("HyperFEMKernels") === nothing ? "lazy" : "lazy,compiled"), ",")))
    @time begin
      include("MatrixFreeTests.jl")
    end
    @time begin
      include("CompiledFunctionalsTests.jl")
    end
    @time begin
      include("PMultigridTests.jl")
    end
  end

  # PETSc variants: local only (GridapPETSc is not in the CI test environment, see PETScSolverTests.jl)
  if get(ENV, "HYPERFEM_TEST_PETSC", "0") == "1"
    @time begin
      include("PETScSolverTests.jl")
    end
    @time begin
      include("PETScFieldSplitTests.jl")
    end
  end

end
