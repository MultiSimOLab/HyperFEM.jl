using Gridap.TensorValues
using Gridap.TensorValues: ThirdOrderTensorValue
using HyperFEM.TensorAlgebra
using Test


# Result type of every tensor operation.
#
# HyperFEM extends the Gridap operators (type piracy) to work with tensors of order > 2, flattened
# into matrices with combined indices (see `_flat_idx`). In dimension D, a tensor of order
#   1 is a `VectorValue{D}`,
#   2 is a `TensorValue{D,D}`,
#   3 is a `TensorValue{D,D²}` (HyperFEM) or a `ThirdOrderTensorValue{D,D,D}` (Gridap),
#   4 is a `TensorValue{D²,D²}`.
# These tests check that each operation returns the order (and representation) it is meant to, and that the
# return type is inferred. Operations which are not supported yet, or which return a wrong representation,
# are marked `@test_broken`: when they are fixed (e.g. moving to the full indexing of Gridap 0.20) they are
# reported as "Unexpected Pass" and must be moved to the supported list.
#
# Conventions of the contractions:
#   a ⊙ b   contracts all the indices of the operand of lower order: the last indices of `a` or the first indices of `b`;
#   a ⋅ b   contracts the last index of `a` with the first index of `b`;
#   a ⋅² b  contracts the last two indices of `a` with the first two indices of `b`;
#   a * b   is `a ⋅ b` for `TensorValue` operands.
# `H⋅H` and `H*H` are still the matrix product of the flattened representation, i.e. `H ⋅² H`.

V(D)  = VectorValue{D,Float64}
T2(D) = TensorValue{D,D,Float64}
T3(D) = TensorValue{D,D*D,Float64}
G3(D) = ThirdOrderTensorValue{D,D,D,Float64}
T4(D) = TensorValue{D*D,D*D,Float64}

function tensor_operands(D)
  v = VectorValue{D,Float64}(ntuple(i -> Float64(i), D))
  A = TensorValue{D,D,Float64}(ntuple(i -> 1.0 + i / 10, D^2))
  W = TensorValue{D,D^2,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  G = ThirdOrderTensorValue{D,D,D,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  H = TensorValue{D^2,D^2,Float64}(ntuple(i -> 1.0 + i / 1000, D^4))
  (v, A, W, G, H)
end

# (label, operation, arguments, expected type)
function supported_operations(D)
  v, A, W, G, H = tensor_operands(D)
  ops = Any[
    # outer product: order(a) + order(b)
    ("v⊗v", ⊗, (v, v), T2(D)),
    ("v⊗A", ⊗, (v, A), G3(D)),  # left to Gridap, see Operations.jl
    ("A⊗v", ⊗, (A, v), G3(D)),
    ("A⊗A", ⊗, (A, A), T4(D)),
    ("W⊗v", ⊗, (W, v), T4(D)),
    # inner product: |order(a) - order(b)|
    ("v⊙v", ⊙, (v, v), Float64),
    ("A⊙A", ⊙, (A, A), Float64),
    ("W⊙W", ⊙, (W, W), Float64),
    ("G⊙G", ⊙, (G, G), Float64),
    ("H⊙H", ⊙, (H, H), Float64),
    ("W⊙v", ⊙, (W, v), T2(D)),
    ("v⊙W", ⊙, (v, W), T2(D)),
    ("W⊙A", ⊙, (W, A), V(D)),
    ("H⊙A", ⊙, (H, A), T2(D)),
    # dot product: order(a) + order(b) - 2
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
    ("H⋅H", ⋅, (H, H), T4(D)),  # matrix product of the flattened representation, i.e. H ⋅² H
    ("v⋅G", ⋅, (v, G), T2(D)),
    ("G⋅v", ⋅, (G, v), T2(D)),
    ("A⋅G", ⋅, (A, G), G3(D)),
    ("G⋅A", ⋅, (G, A), G3(D)),
    # double contraction: order(a) + order(b) - 4
    ("A⋅²A", ⋅², (A, A), Float64),
    ("H⋅²A", ⋅², (H, A), T2(D)),
    ("A⋅²H", ⋅², (A, H), T2(D)),
    ("H⋅²H", ⋅², (H, H), T4(D)),
    ("W⋅²A", ⋅², (W, A), V(D)),
    # matrix product, same as dot
    ("A*v", *, (A, v), V(D)),
    ("A*A", *, (A, A), T2(D)),
    ("A*W", *, (A, W), T3(D)),
    ("W*v", *, (W, v), T2(D)),
    ("W*A", *, (W, A), T3(D)),
    ("W*W", *, (W, W), T4(D)),
    ("A*H", *, (A, H), T4(D)),
    ("H*H", *, (H, H), T4(D)),
    # products in index notation
    ("⊗ᵢʲ",       ⊗ᵢʲ,       (v, v), T2(D)),
    ("⊗ᵢⱼᵏ",      ⊗ᵢⱼᵏ,      (A, v), T3(D)),
    ("⊗ᵢₖʲ",      ⊗ᵢₖʲ,      (A, v), T3(D)),
    ("⊗ᵢʲᵏ",      ⊗ᵢʲᵏ,      (v, A), T3(D)),
    ("⊗ᵢⱼₖˡ",     ⊗ᵢⱼₖˡ,     (W, v), T4(D)),
    ("⊗ᵢⱼᵏˡ",     ⊗ᵢⱼᵏˡ,     (A, A), T4(D)),
    ("⊗ᵢₖʲˡ",     ⊗ᵢₖʲˡ,     (A, A), T4(D)),
    ("⊗ᵢₗʲᵏ",     ⊗ᵢₗʲᵏ,     (A, A), T4(D)),
    ("⊗ᵢⱼₘᵐ",     ⊗ᵢⱼₘᵐ,     (W, v), T2(D)),
    ("⊗ₘᵐⁱʲ",     ⊗ₘᵐⁱʲ,     (v, W), T2(D)),
    ("⊗ᵢⱼₘᵐᵏ",    ⊗ᵢⱼₘᵐᵏ,    (W, A), T3(D)),
    ("⊗ᵢₘᵐʲᵏˡ",   ⊗ᵢₘᵐʲᵏˡ,   (A, H), T4(D)),
    ("⊗ᵢₘʲᵐᵏˡ",   ⊗ᵢₘʲᵐᵏˡ,   (A, H), T4(D)),
    ("⊗ᵢⱼₘᵐᵏˡ",   ⊗ᵢⱼₘᵐᵏˡ,   (W, W), T4(D)),
    ("⊗ᵢₘₙᵐⁿ",    ⊗ᵢₘₙᵐⁿ,    (W, A), V(D)),
    ("⊗ᵢⱼₘₙᵐⁿ",   ⊗ᵢⱼₘₙᵐⁿ,   (H, A), T2(D)),
    ("⊗ᵢₘⱼₙᵐⁿ",   ⊗ᵢₘⱼₙᵐⁿ,   (H, A), T2(D)),
    ("⊗ₘₙᵐⁿⁱʲ",   ⊗ₘₙᵐⁿⁱʲ,   (A, H), T2(D)),
    ("⊗ᵢⱼₘₙᵐⁿᵏˡ", ⊗ᵢⱼₘₙᵐⁿᵏˡ, (H, H), T4(D)),
    ("push_forward_C_to_F", push_forward_C_to_F, (A, H), T4(D)),
    # unary functions
    ("tr(A)",    tr,    (A,), Float64),
    ("det(A)",   det,   (A,), Float64),
    ("inv(A)",   inv,   (A,), T2(D)),
    ("IIsym(A)", IIsym, (A,), T4(D)),
    # transpose: cyclic shift of the third-order tensors, matrix transpose of the flattened ones
    ("A'", transpose, (A,), T2(D)),
    ("H'", transpose, (H,), T4(D)),
    ("G'", transpose, (G,), G3(D)),
    ("W'", transpose, (W,), TensorValue{D*D,D,Float64}),
  ]
  # operations preserving the type: sum, difference, negation and scaling
  for (name, X) in (("v", v), ("A", A), ("W", W), ("G", G), ("H", H))
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

# Operations not supported yet, or returning a wrong representation (the current result in the comment)
function broken_operations(D)
  v, A, W, G, H = tensor_operands(D)
  ops = Any[
    ("v⊗W", ⊗, (v, W), T4(D)),  # ThirdOrderTensorValue{D,D,D²} (Gridap fallback)
    ("v⊗G", ⊗, (v, G), T4(D)),  # error
    ("G⊗v", ⊗, (G, v), T4(D)),  # error
    ("A⊙v", ⊙, (A, v), V(D)),   # error
    ("v⊙A", ⊙, (v, A), V(D)),   # error
    ("A⊙H", ⊙, (A, H), T2(D)),  # error
    ("G⊙v", ⊙, (G, v), T2(D)),  # error
    ("v⊙G", ⊙, (v, G), T2(D)),  # error
    ("G⊙A", ⊙, (G, A), V(D)),   # error
    ("H⋅A", ⋅, (H, A), T4(D)),  # error
    ("W⋅G", ⋅, (W, G), T4(D)),  # error
    ("G⋅W", ⋅, (G, W), T4(D)),  # ThirdOrderTensorValue{D,D,D²} (Gridap fallback)
    ("G⋅G", ⋅, (G, G), T4(D)),  # error
    ("G*v", *, (G, v), T2(D)),  # error
    ("A*G", *, (A, G), G3(D)),  # error
    ("G*A", *, (G, A), G3(D)),  # error
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
