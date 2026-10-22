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
A = A + A' + I3
F = TensorValue(1.:9...) * 1e-2 + I3
H = TensorValue(1.:81...)

SUITE["Tensor algebra"]["δδ_μ_2d"] = @benchmarkable δᵢₖδⱼₗ2D + δᵢₗδⱼₖ2D
SUITE["Tensor algebra"]["δδ_λ_2d"] = @benchmarkable 1.0 * δᵢⱼδₖₗ2D
SUITE["Tensor algebra"]["Cofactor"] = @benchmarkable cof($A)
SUITE["Tensor algebra"]["Det(A)Inv(A')"] = @benchmarkable det($A)*inv($A')
SUITE["Tensor algebra"]["×ᵢ⁴"] = @benchmarkable ×ᵢ⁴($A)
SUITE["Tensor algebra"]["IIsym"] = @benchmarkable IIsym($A)
SUITE["Tensor algebra"]["push_forward_C_to_F"] = @benchmarkable push_forward_C_to_F($F,$H)

# Projection of tangents onto {eigenvalues >= δ} (HessianRegularization): fast path (positive definite) and
# eigen-decomposition path (indefinite), 2D (4x4) and 3D (9x9). Allocation-free in both.
B4 = TensorValue(1.:16...)
B9 = TensorValue(1.:81...)
C4_def = B4 * B4' / 100 + I4
C4_ind = B4 + B4'
C9_def = B9 * B9' / 10000 + I9
C9_ind = B9 + B9'
SUITE["Tensor algebra"]["regularize_hessian 4x4 definite"] = @benchmarkable regularize_hessian($(Ref(C4_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 4x4 indefinite"] = @benchmarkable regularize_hessian($(Ref(C4_ind))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 definite"] = @benchmarkable regularize_hessian($(Ref(C9_def))[], 1e-6)
SUITE["Tensor algebra"]["regularize_hessian 9x9 indefinite"] = @benchmarkable regularize_hessian($(Ref(C9_ind))[], 1e-6)
