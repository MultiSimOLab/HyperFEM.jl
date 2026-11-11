using HyperFEM.TensorAlgebra
using BenchmarkTools


function _δδ_μ_2D(μ::Float64)
  TensorValue{4,4,Float64,16}(
    2 * μ,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0 * μ)
end

function _δδ_λ_2D(λ::Float64)
  TensorValue{4,4,Float64,16}(
    λ,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    λ)
end

A = TensorValue(1.:9...)
A = A + A' + I3D
F = TensorValue(1.:9...) * 1e-2 + I3D
H = TensorValue(1.:81...)

δᵢₖδⱼₗ2D, δᵢₗδⱼₖ2D, δᵢⱼδₖₗ2D = I2D ⊗ᵢₖʲˡ I2D, I2D ⊗ᵢₗʲᵏ I2D, I2D ⊗ᵢⱼᵏˡ I2D
SUITE["Tensor algebra"]["δδ_μ_2d"] = @benchmarkable $δᵢₖδⱼₗ2D + $δᵢₗδⱼₖ2D
SUITE["Tensor algebra"]["δδ_λ_2d"] = @benchmarkable 1.0 * $δᵢⱼδₖₗ2D
SUITE["Tensor algebra"]["Cofactor"] = @benchmarkable cof($A)
SUITE["Tensor algebra"]["Det(A)Inv(A')"] = @benchmarkable det($A)*inv($A')
SUITE["Tensor algebra"]["×ᵢ⁴"] = @benchmarkable ×ᵢ⁴($A)
SUITE["Tensor algebra"]["IIsym"] = @benchmarkable IIsym($A)
SUITE["Tensor algebra"]["push_forward_C_to_F"] = @benchmarkable push_forward_C_to_F($F,$H)

# Projection of tangents onto {eigenvalues >= δ} (HessianRegularization): fast path (positive definite) and
# eigen-decomposition path (indefinite), 2D (4x4) and 3D (9x9). Allocation-free in both.
B4 = TensorValue(1.:16...)
B9 = TensorValue(1.:81...)
C4_def = B4 * B4' / 100 + 𝕀2D
C4_ind = B4 + B4'
C9_def = B9 * B9' / 10000 + 𝕀3D
C9_ind = B9 + B9'
SUITE["Tensor algebra"]["regularize_hessian 4x4 definite"] = @benchmarkable regularize_hessian($(Ref(C4_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 4x4 indefinite"] = @benchmarkable regularize_hessian($(Ref(C4_ind))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 definite"] = @benchmarkable regularize_hessian($(Ref(C9_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 indefinite"] = @benchmarkable regularize_hessian($(Ref(C9_ind))[], 1e-6)


# Tensor operations, grouped by operation and dimension. The labels give the order of the operands:
#   "3" is a HyperFEM third-order tensor `TensorValue{D,D²}`, "3g" a Gridap `ThirdOrderTensorValue{D,D,D}`.
# The result type of each operation is checked in test/TensorAlgebraTests/TensorTypes.jl.
# The operations which are not supported yet are skipped, and they are benchmarked once they are implemented.
using Gridap.TensorValues: ThirdOrderTensorValue

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
  W = TensorValue{D,D^2,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  G = ThirdOrderTensorValue{D,D,D,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
  H = TensorValue{D^2,D^2,Float64}(ntuple(i -> 1.0 + i / 1000, D^4))
  dim = "$(D)D"

  group = SUITE["Tensor algebra"]["outer"][dim]
  for (label, a, b) in (("1⊗1", v, u), ("1⊗2", v, A), ("2⊗1", A, v), ("2⊗2", A, B),
                        ("1⊗3", v, W), ("3⊗1", W, v), ("1⊗3g", v, G), ("3g⊗1", G, v))
    add_tensor_benchmark!(group, label, ⊗, a, b)
  end

  group = SUITE["Tensor algebra"]["inner"][dim]
  for (label, a, b) in (("1⊙1", v, u), ("2⊙2", A, B), ("3⊙3", W, W), ("3g⊙3g", G, G), ("4⊙4", H, H),
                        ("2⊙1", A, v), ("1⊙2", v, A), ("3⊙1", W, v), ("1⊙3", v, W), ("3⊙2", W, A),
                        ("4⊙2", H, A), ("2⊙4", A, H), ("3g⊙1", G, v), ("1⊙3g", v, G), ("3g⊙2", G, A))
    add_tensor_benchmark!(group, label, ⊙, a, b)
  end

  group = SUITE["Tensor algebra"]["dot"][dim]
  for (label, a, b) in (("1⋅1", v, u), ("1⋅2", v, A), ("2⋅1", A, v), ("2⋅2", A, B),
                        ("1⋅3", v, W), ("3⋅1", W, v), ("2⋅3", A, W), ("3⋅2", W, A), ("3⋅3", W, W),
                        ("2⋅4", A, H), ("4⋅2", H, A), ("4⋅4", H, H),
                        ("1⋅3g", v, G), ("3g⋅1", G, v), ("2⋅3g", A, G), ("3g⋅2", G, A),
                        ("3⋅3g", W, G), ("3g⋅3", G, W), ("3g⋅3g", G, G))
    add_tensor_benchmark!(group, label, ⋅, a, b)
  end

  group = SUITE["Tensor algebra"]["double contraction"][dim]
  for (label, a, b) in (("2⋅²2", A, B), ("4⋅²2", H, A), ("2⋅²4", A, H), ("4⋅²4", H, H), ("3⋅²2", W, A))
    add_tensor_benchmark!(group, label, ⋅², a, b)
  end

  group = SUITE["Tensor algebra"]["*"][dim]
  for (label, a, b) in (("2*1", A, v), ("2*2", A, B), ("2*3", A, W), ("3*1", W, v), ("3*2", W, A),
                        ("3*3", W, W), ("2*4", A, H), ("4*4", H, H),
                        ("3g*1", G, v), ("2*3g", A, G), ("3g*2", G, A))
    add_tensor_benchmark!(group, label, *, a, b)
  end

  if D == 3
    group = SUITE["Tensor algebra"]["cross"][dim]
    for (label, a, b) in (("1×1", v, u), ("2×2", A, B), ("2×4", A, H), ("4×2", H, A), ("3×2", W, A))
      add_tensor_benchmark!(group, label, ×, a, b)
    end
  end

  group = SUITE["Tensor algebra"]["arithmetic"][dim]
  for (name, X) in (("1", v), ("2", A), ("3", W), ("3g", G), ("4", H))
    add_tensor_benchmark!(group, "$name+$name", +, X, X)
    add_tensor_benchmark!(group, "$name-$name", -, X, X)
    add_tensor_benchmark!(group, "-$name", -, X)
    add_tensor_benchmark!(group, "α*$name", *, 2.0, X)
    add_tensor_benchmark!(group, "$name/α", /, X, 2.0)
    add_tensor_benchmark!(group, "$name'", transpose, X)
  end

  group = SUITE["Tensor algebra"]["index notation"][dim]
  for (label, f, args) in (("⊗ᵢʲ", ⊗ᵢʲ, (v, u)), ("⊗ᵢⱼᵏ", ⊗ᵢⱼᵏ, (A, v)), ("⊗ᵢₖʲ", ⊗ᵢₖʲ, (A, v)),
                           ("⊗ᵢʲᵏ", ⊗ᵢʲᵏ, (v, A)), ("⊗ᵢⱼᵏˡ", ⊗ᵢⱼᵏˡ, (A, B)), ("⊗ᵢₖʲˡ", ⊗ᵢₖʲˡ, (A, B)),
                           ("⊗ᵢₗʲᵏ", ⊗ᵢₗʲᵏ, (A, B)), ("⊗ᵢⱼₖˡ", ⊗ᵢⱼₖˡ, (W, v)),
                           ("⊗ᵢⱼₘᵐ", ⊗ᵢⱼₘᵐ, (W, v)), ("⊗ₘᵐⁱʲ", ⊗ₘᵐⁱʲ, (v, W)), ("⊗ᵢₘₙᵐⁿ", ⊗ᵢₘₙᵐⁿ, (W, A)),
                           ("⊗ᵢⱼₘₙᵐⁿ", ⊗ᵢⱼₘₙᵐⁿ, (H, A)), ("⊗ᵢₘⱼₙᵐⁿ", ⊗ᵢₘⱼₙᵐⁿ, (H, A)),
                           ("⊗ᵢⱼₘᵐᵏ", ⊗ᵢⱼₘᵐᵏ, (W, A)),
                           ("⊗ᵢⱼₘᵐᵏˡ", ⊗ᵢⱼₘᵐᵏˡ, (W, W)),
                           ("⊗ᵢₘᵐʲᵏˡ", ⊗ᵢₘᵐʲᵏˡ, (A, H)),
                           ("⊗ᵢₘʲᵐᵏˡ", ⊗ᵢₘʲᵐᵏˡ, (A, H)),
                           ("⊗ₘₙᵐⁿⁱʲ", ⊗ₘₙᵐⁿⁱʲ, (A, H)),
                           ("⊗ᵢⱼₘₙᵐⁿᵏˡ", ⊗ᵢⱼₘₙᵐⁿᵏˡ, (H, H)))
    add_tensor_benchmark!(group, label, f, args...)
  end
end

for D in (2, 3)
  add_tensor_operations_benchmarks!(SUITE, D)
end
