using Test
using Gridap.TensorValues
using HyperFEM.PhysicalModels
using HyperFEM.TensorAlgebra

"""
    test_type_stability__(model::Mechano, F, args...)

Check with `@inferred` that the energy, the first Piola and the tangent of `model` are type stable and return
`Float64`, `TensorValue{D,D,Float64}` and `FourthOrderTensorValue{D,Float64}`.
`args...` are forwarded as-is: fibers `N` for AnisoElastic, `(Fn, A...)` for viscous models, etc.
"""
function test_type_stability__(model::Mechano, F::TensorValue{D}, args...) where D
  Ψ, ∂Ψu, ∂Ψuu = model()
  @test (@inferred Ψ(F, args...)) isa Float64
  @test (@inferred ∂Ψu(F, args...)) isa TensorValue{D,D,Float64}
  @test (@inferred ∂Ψuu(F, args...)) isa FourthOrderTensorValue{D,Float64}
end
