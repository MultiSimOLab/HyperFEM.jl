using HyperFEM.TensorAlgebra
using BenchmarkTools


# Projection of tangents onto {eigenvalues >= δ} (HessianRegularization): fast path (positive definite) and
# eigen-decomposition path (indefinite), 2D (4x4) and 3D (9x9). Allocation-free in both.
B4 = TensorValue(1.:16...)
B9 = TensorValue(1.:81...)
C4_def = B4 * B4' / 100 + TensorValue(Tuple(𝕀2D))
C4_ind = B4 + B4'
C9_def = B9 * B9' / 10000 + TensorValue(Tuple(𝕀3D))
C9_ind = B9 + B9'
SUITE["Tensor algebra"]["regularize_hessian 4x4 definite"] = @benchmarkable regularize_hessian($(Ref(C4_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 4x4 indefinite"] = @benchmarkable regularize_hessian($(Ref(C4_ind))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 definite"] = @benchmarkable regularize_hessian($(Ref(C9_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 indefinite"] = @benchmarkable regularize_hessian($(Ref(C9_ind))[], 1e-6)


# Projection of tangents onto {eigenvalues >= δ} (HessianRegularization): fast path (positive definite) and
# eigen-decomposition path (indefinite), 2D and 3D. Allocation-free in both.
B2 = FourthOrderTensorValue(1.:16...)
B3 = FourthOrderTensorValue(1.:81...)
C2_def = B2 ⋅² B2' / 100 + 𝕀2D
C2_ind = B2 + B2'
C3_def = B3 ⋅² B3' / 10000 + 𝕀3D
C3_ind = B3 + B3'
SUITE["Tensor algebra"]["regularize_hessian 2D definite"] = @benchmarkable regularize_hessian($(Ref(C2_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 2D indefinite"] = @benchmarkable regularize_hessian($(Ref(C2_ind))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 3D definite"] = @benchmarkable regularize_hessian($(Ref(C3_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 3D indefinite"] = @benchmarkable regularize_hessian($(Ref(C3_ind))[], 1e-6)



# Tensor operations, grouped by operation and dimension. The labels give the order of the operands.
# The result type of each operation is checked in test/TensorAlgebraTests/TensorTypes.jl.
# The operations which are not supported yet are skipped, and they are benchmarked once they are implemented.

# The operands are passed through a `Ref` to avoid constant propagation
function add_tensor_benchmark!(group, label, f, args...)
  try
    f(args...)
  catch
    return group  # not supported yet
  end
  if length(args) == 1
    group[label] = @benchmarkable $f($(Ref(args[1]))[])
  else
    group[label] = @benchmarkable $f($(Ref(args[1]))[], $(Ref(args[2]))[])
  end
  group
end

function add_tensor_operations_benchmarks!(SUITE, D)
  v = VectorValue{D,Float64}(ntuple(i -> Float64(i), D))
  u = VectorValue{D,Float64}(ntuple(i -> Float64(D + i), D))
  A = TensorValue{D,D,Float64}(ntuple(i -> 1.0 + i / 10, D^2))
  B = TensorValue{D,D,Float64}(ntuple(i -> 2.0 - i / 10, D^2))
  W = ThirdOrderTensorValue{D,D,D,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  H = FourthOrderTensorValue(ntuple(i -> 1.0 + i / 1000, D^4))
  dim = "$(D)D"

  group = SUITE["Tensor algebra"]["outer"][dim]
  for (label, a, b) in (("1⊗1", v, u), ("1⊗2", v, A), ("2⊗1", A, v), ("2⊗2", A, B),
                        ("1⊗3", v, W), ("3⊗1", W, v))
    add_tensor_benchmark!(group, label, ⊗, a, b)
  end

  group = SUITE["Tensor algebra"]["inner"][dim]
  for (label, a, b) in (("1⊙1", v, u), ("2⊙2", A, B), ("3⊙3", W, W), ("4⊙4", H, H),
("2⊙1", A, v), ("1⊙2", v, A), ("3⊙1", W, v), ("1⊙3", v, W), ("3⊙2", W, A),
("4⊙2", H, A), ("2⊙4", A, H))
    add_tensor_benchmark!(group, label, ⊙, a, b)
  end

  group = SUITE["Tensor algebra"]["dot"][dim]
  for (label, a, b) in (("1⋅1", v, u), ("1⋅2", v, A), ("2⋅1", A, v), ("2⋅2", A, B),
                        ("1⋅3", v, W), ("3⋅1", W, v), ("2⋅3", A, W), ("3⋅2", W, A), ("3⋅3", W, W),
                        ("2⋅4", A, H), ("4⋅2", H, A))
    add_tensor_benchmark!(group, label, ⋅, a, b)
  end

  group = SUITE["Tensor algebra"]["double contraction"][dim]
  for (label, a, b) in (("2⋅²2", A, B), ("4⋅²2", H, A), ("2⋅²4", A, H), ("4⋅²4", H, H), ("3⋅²2", W, A), ("3⋅²4", W, H))
    add_tensor_benchmark!(group, label, ⋅², a, b)
  end

  group = SUITE["Tensor algebra"]["*"][dim]
  for (label, a, b) in (("2*1", A, v), ("2*2", A, B))
    add_tensor_benchmark!(group, label, *, a, b)
  end

  if D == 3
    group = SUITE["Tensor algebra"]["cross"][dim]
    for (label, a, b) in (("1×1", v, u), ("2×2", A, B), ("2×4", A, H), ("4×2", H, A), ("3×2", W, A))
      add_tensor_benchmark!(group, label, ×, a, b)
    end
  end

  group = SUITE["Tensor algebra"]["arithmetic"][dim]
  for (name, X) in (("1", v), ("2", A), ("3", W), ("4", H))
    add_tensor_benchmark!(group, "$name+$name", +, X, X)
    add_tensor_benchmark!(group, "$name-$name", -, X, X)
    add_tensor_benchmark!(group, "-$name", -, X)
    add_tensor_benchmark!(group, "α*$name", *, 2.0, X)
    add_tensor_benchmark!(group, "$name/α", /, X, 2.0)
    add_tensor_benchmark!(group, "$name'", transpose, X)
  end

  group = SUITE["Tensor algebra"]["index notation"][dim]
  for (label, f, args) in (("⊗ᵢₖʲ", ⊗ᵢₖʲ, (A, v)), ("⊗ᵢₖʲˡ", ⊗ᵢₖʲˡ, (A, B)), ("⊗ᵢₗʲᵏ", ⊗ᵢₗʲᵏ, (A, B)),
                           ("⊗ᵢₘʲᵐᵏˡ", ⊗ᵢₘʲᵐᵏˡ, (A, H)), ("IIsym", IIsym, (A,)), ("push_forward_C_to_F", push_forward_C_to_F, (A, H)))
    add_tensor_benchmark!(group, label, f, args...)
  end
end

for D in (2, 3)
  add_tensor_operations_benchmarks!(SUITE, D)
end
