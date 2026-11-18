using HyperFEM
using Test

@testset "ConstitutiveModels" begin

  @time begin
    include("KinematicsTests.jl")
  end

  @time begin
    include("PhysicalModelTests.jl")
  end

  @time begin
    include("ViscousModelsTests.jl")
  end

  @time begin
    include("ElectroMechanicalTests.jl")
  end

  @time begin
    include("ThermalLawsTests.jl")
  end

  @time begin
    include("DerivativeConsistencyTests.jl")
  end

end
