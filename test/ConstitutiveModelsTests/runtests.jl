using HyperFEM
using Test

@testset "ConstitutiveModels" begin

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

end
