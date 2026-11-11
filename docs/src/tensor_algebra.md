# Tensor algebra

HyperFEM works with tensors of order 1 to 4, stored with full indices (Gridap ≥ 0.20). This page
states the conventions and lists the changes of the tensor algebra revision, for the users of the
former names.

## Storage

| Order | Type                             | Components     |
|:-----:|:---------------------------------|:---------------|
| 1     | `VectorValue{D}`                 | `v[i]`         |
| 2     | `TensorValue{D,D}`               | `A[i,j]`       |
| 3     | `ThirdOrderTensorValue{D,D,D}`   | `W[i,j,k]`     |
| 4     | `FourthOrderTensorValue{D}`      | `H[i,j,k,l]`   |

`FourthOrderTensorValue{D}` is an alias of Gridap's `HighOrderTensorValue{Tuple{D,D,D,D}}`. The
components are stored in column-major order (first index fastest).

For linear algebra (inverse, eigenvalues, linear systems), a fourth-order tensor is a D²×D² matrix,
the linear map on second-order tensors, with rows `(i,j)` and columns `(k,l)`. Both hold the same
components in the same order, so the conversion is the constructor applied to the components:

```julia
M = TensorValue(Tuple(H))              # TensorValue{D²,D²}
H = FourthOrderTensorValue(Tuple(M))   # from a TensorValue{D²,D²}, SMatrix or Matrix (e.g. a ForwardDiff Hessian)
W = ThirdOrderTensorValue(Tuple(M))    # from a D×D² matrix (e.g. a ForwardDiff Jacobian)
H = FourthOrderTensorValue(1.:81...)   # also from the components
```

Pass the components as a tuple in the constitutive models: splatting 81 components (`H...`) is not
type-stable and allocates.

## Operators

| Operator | Name               | Contracted indices                                             |
|:--------:|:-------------------|:---------------------------------------------------------------|
| `a ⊗ b` | outer product      | none                                                           |
| `a ⋅ b`  | single contraction | the last index of `a` with the first index of `b`              |
| `a ⋅² b` | double contraction | the last two indices of `a` with the first two indices of `b`  |
| `a ⊙ b` | full contraction   | all the indices of the operand of lower order                  |
| `a × b`  | cross product (3D) | the last two indices of `a` with the first two indices of `b`  |
| `a'`     | transpose          | `W'[i,j,k] = W[k,i,j]`, `H'[i,j,k,l] = H[k,l,i,j]`             |
| `a * b`  | same as `a ⋅ b`    | second-order tensors only, to be removed                       |

`⊗`, `⋅`, `⋅²` and `⊙` between tensors of the same order are Gridap's. `⋅²` (`\cdot\^2`) is the
alias of Gridap's `double_contraction`. HyperFEM adds `⊙` between tensors of different orders,
the transposes, the norm of high-order tensors and the cross products. It also redefines, with the
same results, the operations that are slower in Gridap for 81 components: `+`, `-` and the scaling
of high-order tensors, and `A ⊗ B` of second-order tensors.

The middle dot `·` (U+00B7) is the same operator as `⋅` (U+22C5): Julia normalizes it when parsing,
so `A · B` and `H ·² G` are `A ⋅ B` and `H ⋅² G`.

## Index notation

The products that the operators cannot write are named in index notation: the subscripts are the
indices of the first operand and the superscripts those of the second one. The free indices
`i, j, k, l` are the indices of the result, in this order, and `m` is summed.

```julia
A ⊗ᵢₖʲ v      # R[i,j,k]   = A[i,k] v[j]
A ⊗ᵢₖʲˡ B     # R[i,j,k,l] = A[i,k] B[j,l]
A ⊗ᵢₗʲᵏ B     # R[i,j,k,l] = A[i,l] B[j,k]
A ⊗ᵢₘʲᵐᵏˡ H   # R[i,j,k,l] = Σm A[i,m] H[j,m,k,l]
```

They are typed with the LaTeX completions of the REPL and editors, e.g. `\otimes\_i\_k\^j\^l<tab>`.
Each one is a line in `TensorAlgebra/Operations.jl` with the index labels of its operands, and
`einstein_sum` (`TensorAlgebra/Kernel.jl`) generates its code:

```julia
(⊗ᵢₘʲᵐᵏˡ)(A::TensorValue, H::FourthOrderTensorValue) = einstein_sum(Val((:i, :m)), Val((:j, :m, :k, :l)), A, H)
```

## Changes of the revision

### Storage with full indices

Third-order tensors are `ThirdOrderTensorValue{D,D,D}` instead of `TensorValue{D,D²}`, and
fourth-order tensors are `FourthOrderTensorValue{D}` instead of `TensorValue{D²,D²}`. The components
keep their order, so a 9×9 matrix and the corresponding fourth-order tensor hold the same values.

- Products of fourth-order tensors are written `⋅²` instead of `*` (Gridap's `*` is an error
  for them): `H ⋅² G`, `W ⋅² H`.
- `H'` is the major transpose, as the transpose of the 9×9 matrix was.
- `norm(H)` (Frobenius) is defined for high-order tensors, so `isapprox` works on them.
- The tangents computed with ForwardDiff are converted with `FourthOrderTensorValue(Tuple(M))` and
  `ThirdOrderTensorValue(Tuple(M))` instead of `TensorValue(M)`.
- `regularize_hessian` accepts fourth-order tensors.

### Renamed operators

| Former name                      | New expression   | Formula                         |
|:---------------------------------|:-----------------|:--------------------------------|
| `⊗₁²`                            | `⊗`              | `vᵢ uⱼ`                         |
| `⊗₁₂³`, `⊗₁²³`                   | `⊗`              | `Aᵢⱼ vₖ`, `vᵢ Aⱼₖ`              |
| `⊗₁₂³⁴`                          | `⊗`              | `Aᵢⱼ Bₖₗ`                       |
| `⊗₁₂₃⁴`                          | `⊗`              | `Wᵢⱼₖ vₗ` (see the fixes below) |
| `⊗₁₃²`                           | `⊗ᵢₖʲ`           | `Aᵢₖ vⱼ`                        |
| `⊗₁₃²⁴`                          | `⊗ᵢₖʲˡ`          | `Aᵢₖ Bⱼₗ`                       |
| `⊗₁₄²³`                          | `⊗ᵢₗʲᵏ`          | `Aᵢₗ Bⱼₖ`                       |
| `⊙₁₂₃³`, `⊙₁¹²³`                 | `⋅`              | `Σ Wᵢⱼₘ vₘ`, `Σ vₘ Wₘᵢⱼ`        |
| `⊙₁₂₃²³`, `⊙₁₂₃₄³⁴`              | `⋅²` (or `⊙`)    | `Σ Wᵢₘₙ Aₘₙ`, `Σ Hᵢⱼₘₙ Aₘₙ`     |
| `contraction_IP_PJKL`            | `⋅`              | `Σ Aᵢₘ Hₘⱼₖₗ`                   |
| `contraction_IP_JPKL`            | `⊗ᵢₘʲᵐᵏˡ`        | `Σ Aᵢₘ Hⱼₘₖₗ`                   |
| `contraction_IJK_KLP`            | `⋅`              | `Σ Wᵢⱼₘ Wₘₖₗ`                   |
| `contraction_IJK_KL`             | `⋅`              | `Σ Wᵢⱼₘ Aₘₖ`                    |
| `⊙₁₂₃₄²⁴`, `contraction_IJKL_JL` | removed          | (unused, see the fixes below)   |
| `transpose_IJK_KIJ`              | `W'`             | `Wₖᵢⱼ`                          |
| `H * G` (fourth-order)           | `H ⋅² G`         | `Σ Hᵢⱼₘₙ Gₘₙₖₗ`                 |

`×ᵢ⁴`, `IIsym`, `cof` and `push_forward_C_to_F` keep their names. The cross products keep their
implementation and will be revised separately.

### Renamed or removed constants

| Former name                   | New name or replacement                     |
|:------------------------------|:--------------------------------------------|
| `I2`, `I3`                    | `I2D`, `I3D` (second-order identities)      |
| `I4`, `I9`                    | `𝕀2D`, `𝕀3D` (fourth-order identities, `FourthOrderTensorValue`) |
| `zerotensor3`, `zerotensor9`  | `zero(TensorValue{3,3,Float64})`, `zero(FourthOrderTensorValue{3,Float64,81})` |
| `δᵢₖδⱼₗ2D`, `δᵢₖδⱼₗ3D`        | `𝕀2D`, `𝕀3D`, or `I2D ⊗ᵢₖʲˡ I2D`            |
| `δᵢₗδⱼₖ2D`, `δᵢₗδⱼₖ3D`, `δⱼₖδᵢₗ3D` | `I2D ⊗ᵢₗʲᵏ I2D`, `I3D ⊗ᵢₗʲᵏ I3D`       |
| `δᵢⱼδₖₗ2D`, `δᵢⱼδₖₗ3D`        | `I2D ⊗ I2D`, `I3D ⊗ I3D`                    |
| `_flat_idx`, `_ij`, `_ijk`, `_ijkl` | removed: index the tensors with `A[i,j,k,l]` |

The products of identities give the same values as the former constants. In a constitutive model,
compute them once when the model is built (outside the functions of `F`), as in `LinearElasticity3D`.

### Fixes

Two former operations computed something different from their name:

- `⊗₁₂₃⁴` computed `Wₖᵢⱼ vₗ` instead of `Wᵢⱼₖ vₗ`. `W ⊗ v` computes `Wᵢⱼₖ vₗ`.
- `⊙₁₂₃₄²⁴` (`contraction_IJKL_JL`) returned the transpose of its name, `Rₖᵢ = Σ Hᵢⱼₖₗ Aⱼₗ`.

Neither was used by the constitutive models. All the other operations return results bitwise
identical to the former implementation.
