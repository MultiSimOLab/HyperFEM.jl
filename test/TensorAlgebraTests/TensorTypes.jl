using Gridap.TensorValues
using HyperFEM.TensorAlgebra
using Test


# Result type of every tensor operation.
#
# The tensors are stored with full indices. In dimension D, a tensor of order
#   1 is a `VectorValue{D}`,
#   2 is a `TensorValue{D,D}`,
#   3 is a `ThirdOrderTensorValue{D,D,D}`,
#   4 is a `FourthOrderTensorValue{D}` (Gridap's `HighOrderTensorValue{Tuple{D,D,D,D}}`).
# These tests check that each operation returns the order it is meant to, and that the return type
# is inferred. Operations which are not supported yet are marked `@test_broken`: when they are
# implemented they are reported as "Unexpected Pass" and must be moved to the supported list.
#
# Conventions of the contractions:
#   a ⊙ b   contracts all the indices of the operand of lower order: the last indices of `a` or the first indices of `b`;
#   a ⋅ b   contracts the last index of `a` with the first index of `b`;
#   a ⋅² b  contracts the last two indices of `a` with the first two indices of `b`;
#   a * b   is `a ⋅ b` for second-order tensors.

V(D)  = VectorValue{D,Float64}
T2(D) = TensorValue{D,D,Float64}
T3(D) = ThirdOrderTensorValue{D,D,D,Float64}
T4(D) = FourthOrderTensorValue{D,Float64}

function tensor_operands(D)
  v = VectorValue{D,Float64}(ntuple(i -> Float64(i), D))
  A = TensorValue{D,D,Float64}(ntuple(i -> 1.0 + i / 10, D^2))
  W = ThirdOrderTensorValue{D,D,D,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  H = FourthOrderTensorValue(ntuple(i -> 1.0 + i / 1000, D^4))
  (v, A, W, H)
end

# (label, operation, arguments, expected type)
function supported_operations(D)
  v, A, W, H = tensor_operands(D)
  ops = Any[
    # outer product: order(a) + order(b)
    ("v⊗v", ⊗, (v, v), T2(D)),
    ("v⊗A", ⊗, (v, A), T3(D)),
    ("A⊗v", ⊗, (A, v), T3(D)),
    ("A⊗A", ⊗, (A, A), T4(D)),
    ("W⊗v", ⊗, (W, v), T4(D)),
    ("v⊗W", ⊗, (v, W), T4(D)),
    # full contraction: |order(a) - order(b)|
    ("v⊙v", ⊙, (v, v), Float64),
    ("A⊙A", ⊙, (A, A), Float64),
    ("W⊙W", ⊙, (W, W), Float64),
    ("H⊙H", ⊙, (H, H), Float64),
    ("W⊙v", ⊙, (W, v), T2(D)),
    ("v⊙W", ⊙, (v, W), T2(D)),
    ("W⊙A", ⊙, (W, A), V(D)),
    ("H⊙A", ⊙, (H, A), T2(D)),
    ("A⊙H", ⊙, (A, H), T2(D)),
    # single contraction: order(a) + order(b) - 2
    ("v⋅v", ⋅, (v, v), Float64),
    ("v⋅A", ⋅, (v, A), V(D)),
    ("A⋅v", ⋅, (A, v), V(D)),
    ("A⋅A", ⋅, (A, A), T2(D)),
    ("v⋅W", ⋅, (v, W), T2(D)),
    ("W⋅v", ⋅, (W, v), T2(D)),
    ("A⋅W", ⋅, (A, W), T3(D)),
    ("W⋅A", ⋅, (W, A), T3(D)),
    ("W⋅W", ⋅, (W, W), T4(D)),
    ("A⋅H", ⋅, (A, H), T4(D)),
    ("H⋅A", ⋅, (H, A), T4(D)),
    # double contraction: order(a) + order(b) - 4
    ("A⋅²A", ⋅², (A, A), Float64),
    ("H⋅²A", ⋅², (H, A), T2(D)),
    ("A⋅²H", ⋅², (A, H), T2(D)),
    ("H⋅²H", ⋅², (H, H), T4(D)),
    ("W⋅²A", ⋅², (W, A), V(D)),
    ("W⋅²H", ⋅², (W, H), T3(D)),
    # matrix product of second-order tensors, same as dot
    ("A*v", *, (A, v), V(D)),
    ("A*A", *, (A, A), T2(D)),
    # products in index notation
    ("⊗ᵢₖʲ",    ⊗ᵢₖʲ,    (A, v), T3(D)),
    ("⊗ᵢₖʲˡ",   ⊗ᵢₖʲˡ,   (A, A), T4(D)),
    ("⊗ᵢₗʲᵏ",   ⊗ᵢₗʲᵏ,   (A, A), T4(D)),
    ("⊗ᵢₘʲᵐᵏˡ", ⊗ᵢₘʲᵐᵏˡ, (A, H), T4(D)),
    ("push_forward_C_to_F", push_forward_C_to_F, (A, H), T4(D)),
    # unary functions
    ("tr(A)",    tr,    (A,), Float64),
    ("det(A)",   det,   (A,), Float64),
    ("inv(A)",   inv,   (A,), T2(D)),
    ("IIsym(A)", IIsym, (A,), T4(D)),
    ("A'", transpose, (A,), T2(D)),
    ("W'", transpose, (W,), T3(D)),
    ("H'", transpose, (H,), T4(D)),
    ("TensorValue(Tuple(H))", H -> TensorValue(Tuple(H)), (H,), TensorValue{D^2,D^2,Float64}),
    ("FourthOrderTensorValue(Tuple(M))", M -> FourthOrderTensorValue(Tuple(M)), (TensorValue(Tuple(H)),), T4(D)),
  ]
  # operations preserving the type: sum, difference, negation and scaling
  for (name, X) in (("v", v), ("A", A), ("W", W), ("H", H))
    T = typeof(X)
    push!(ops,
      ("$name+$name", +, (X, X), T),
      ("$name-$name", -, (X, X), T),
      ("-$name", -, (X,), T),
      ("2*$name", *, (2.0, X), T),
      ("$name*2", *, (X, 2.0), T),
      ("$name/2", /, (X, 2.0), T))
  end
  if D == 3
    push!(ops,
      ("v×v", ×, (v, v), V(D)),
      ("A×A", ×, (A, A), T2(D)),
      ("A×H", ×, (A, H), T4(D)),
      ("H×A", ×, (H, A), T4(D)),
      ("W×A", ×, (W, A), T3(D)),
      ("cof(A)", cof, (A,), T2(D)),
      ("×ᵢ⁴(A)", ×ᵢ⁴, (A,), T4(D)))
  end
  ops
end

# Operations not supported yet (the current result in the comment)
function broken_operations(D)
  v, A, W, H = tensor_operands(D)
  ops = Any[
    ("A⊙v", ⊙, (A, v), V(D)),   # error
    ("v⊙A", ⊙, (v, A), V(D)),   # error
  ]
  if D == 2
    push!(ops, ("cof(A)", cof, (A,), T2(D)))  # error, the cross product is only defined in 3D
  end
  ops
end


@testset "Result types" begin
  for D in (2, 3)
    @testset "D=$D supported" begin
      for (label, f, args, T) in supported_operations(D)
        @testset "$label" begin
          @test (@inferred f(args...)) isa T
        end
      end
    end
    @testset "D=$D not supported yet" begin
      for (label, f, args, T) in broken_operations(D)
        @testset "$label" begin
          @test_broken f(args...) isa T
        end
      end
    end
  end
end
