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
