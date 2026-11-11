# Tensor algebra

HyperFEM extends the Gridap tensor operators to third- and fourth-order tensors, and redefines some of
them for performance (zero allocations, fully unrolled). This page states the conventions and lists the
changes of the naming revision, for the users of the former names.

## Storage

The tensors use the flattened convention. A tensor of order `o` in dimension `D` keeps its `D^o`
components in column-major order (first index fastest):

| Order | Type                 | Example (3D)          |
|:-----:|:---------------------|:----------------------|
| 1     | `VectorValue{D}`     | `VectorValue{3}`      |
| 2     | `TensorValue{D,D}`   | `TensorValue{3,3}`    |
| 3     | `TensorValue{D,D²}`  | `TensorValue{3,9}`    |
| 4     | `TensorValue{D²,D²}` | `TensorValue{9,9}`    |

Gridap's `ThirdOrderTensorValue{D,D,D}` (returned by `v ⊗ A`, `A ⊗ v` and `∇∇`) has the same
component order.

## Operators

| Operator | Name               | Contracted indices                                                  |
|:--------:|:-------------------|:--------------------------------------------------------------------|
| `a ⊗ b`  | outer product      | none                                                                |
| `a ⋅ b`  | single contraction | the last index of `a` with the first index of `b`                   |
| `a ⋅² b` | double contraction | the last two indices of `a` with the first two indices of `b`       |
| `a ⊙ b`  | full contraction   | all the indices of the operand of lower order                        |
| `a × b`  | cross product (3D) | the last two indices of `a` with the first two indices of `b`       |
| `a * b`  | same as `a ⋅ b`    | to be removed                                                       |

`⋅²` (`\cdot\^2`) is Gridap's alias of `double_contraction`, re-exported by HyperFEM.

## Index notation

Any other product is written in index notation, as a `⊗` followed by the indices of the operands:
the subscripts are the indices of the first operand and the superscripts those of the second one.

- The **free indices** are `i, j, k, l`. They give the indices of the result, in this order.
- The **summed indices** are `m, n, p`, named in order of first appearance. Each appears once in
  each operand.

So every operation has exactly one name, and the name is its formula:

```julia
A ⊗ᵢₖʲˡ B      # R[i,j,k,l] = A[i,k] B[j,l]
H ⊗ᵢⱼₘₙᵐⁿ A    # R[i,j]     = Σmn H[i,j,m,n] A[m,n]          (= H ⋅² A = H ⊙ A)
A ⊗ᵢₘʲᵐᵏˡ H    # R[i,j,k,l] = Σm  A[i,m] H[j,m,k,l]
H ⊗ᵢⱼₘₙᵐⁿᵏˡ G  # R[i,j,k,l] = Σmn H[i,j,m,n] G[m,n,k,l]     (= H ⋅² G)
```

They are typed with the LaTeX completions of the REPL and editors, e.g. `\otimes\_i\_k\^j\^l<tab>`.
The products available are listed in `HyperFEM.TensorAlgebra._INDEX_PRODUCTS`, and adding one is a
single line there: the name is parsed and checked when the package is loaded, and the kernel
(`TensorAlgebra/Kernel.jl`) generates the code from it.

## Changes of the naming revision

### Renamed operators

| Former name                         | New name     | Formula                         |
|:------------------------------------|:-------------|:--------------------------------|
| `⊗₁²`                               | `⊗ᵢʲ`        | `vᵢ uⱼ`                         |
| `⊗₁₂³`                              | `⊗ᵢⱼᵏ`       | `Aᵢⱼ vₖ`                        |
| `⊗₁₃²`                              | `⊗ᵢₖʲ`       | `Aᵢₖ vⱼ`                        |
| `⊗₁²³`                              | `⊗ᵢʲᵏ`       | `vᵢ Aⱼₖ`                        |
| `⊗₁₂³⁴`                             | `⊗ᵢⱼᵏˡ`      | `Aᵢⱼ Bₖₗ`                       |
| `⊗₁₃²⁴`                             | `⊗ᵢₖʲˡ`      | `Aᵢₖ Bⱼₗ`                       |
| `⊗₁₄²³`                             | `⊗ᵢₗʲᵏ`      | `Aᵢₗ Bⱼₖ`                       |
| `⊗₁₂₃⁴`                             | `⊗ᵢⱼₖˡ`      | `Wᵢⱼₖ vₗ` (see the fixes below) |
| `⊙₁₂₃³`                             | `⊗ᵢⱼₘᵐ`      | `Σ Wᵢⱼₘ vₘ`                     |
| `⊙₁¹²³`                             | `⊗ₘᵐⁱʲ`      | `Σ vₘ Wₘᵢⱼ`                     |
| `⊙₁₂₃²³`                            | `⊗ᵢₘₙᵐⁿ`     | `Σ Wᵢₘₙ Aₘₙ`                    |
| `⊙₁₂₃₄³⁴`                           | `⊗ᵢⱼₘₙᵐⁿ`    | `Σ Hᵢⱼₘₙ Aₘₙ`                   |
| `⊙₁₂₃₄²⁴`, `contraction_IJKL_JL`    | `⊗ᵢₘⱼₙᵐⁿ`    | `Σ Hᵢₘⱼₙ Aₘₙ` (see the fixes below) |
| `contraction_IP_PJKL`               | `⊗ᵢₘᵐʲᵏˡ`    | `Σ Aᵢₘ Hₘⱼₖₗ`                   |
| `contraction_IP_JPKL`               | `⊗ᵢₘʲᵐᵏˡ`    | `Σ Aᵢₘ Hⱼₘₖₗ`                   |
| `contraction_IJK_KLP`               | `⊗ᵢⱼₘᵐᵏˡ`    | `Σ Wᵢⱼₘ Wₘₖₗ`                   |
| `contraction_IJK_KL`                | `⊗ᵢⱼₘᵐᵏ`     | `Σ Wᵢⱼₘ Aₘₖ`                    |
| —                                   | `⊗ₘₙᵐⁿⁱʲ`    | `Σ Aₘₙ Hₘₙᵢⱼ` (new, `A ⋅² H`)   |
| —                                   | `⊗ᵢⱼₘₙᵐⁿᵏˡ`  | `Σ Hᵢⱼₘₙ Gₘₙₖₗ` (new, `H ⋅² G`) |

`×ᵢ⁴`, `IIsym`, `cof` and `push_forward_C_to_F` keep their names.

### Renamed or removed constants

| Former name                   | New name or replacement                     |
|:------------------------------|:--------------------------------------------|
| `I2`, `I3`                    | `I2D`, `I3D` (second-order identities)      |
| `I4`, `I9`                    | `𝕀2D`, `𝕀3D` (fourth-order identities)      |
| `zerotensor3`, `zerotensor9`  | `zero(TensorValue{3,3,Float64})`, `zero(TensorValue{9,9,Float64})` |
| `δᵢₖδⱼₗ2D`, `δᵢₖδⱼₗ3D`        | `𝕀2D`, `𝕀3D`, or `I2D ⊗ᵢₖʲˡ I2D`            |
| `δᵢₗδⱼₖ2D`, `δᵢₗδⱼₖ3D`, `δⱼₖδᵢₗ3D` | `I2D ⊗ᵢₗʲᵏ I2D`, `I3D ⊗ᵢₗʲᵏ I3D`       |
| `δᵢⱼδₖₗ2D`, `δᵢⱼδₖₗ3D`        | `I2D ⊗ᵢⱼᵏˡ I2D`, `I3D ⊗ᵢⱼᵏˡ I3D`           |

The products of identities give the same values as the former constants. In a constitutive model,
compute them once when the model is built (outside the functions of `F`), as in `LinearElasticity3D`.

### Changes of behaviour

- `H ⋅² G` with two fourth-order tensors is now their double contraction (a fourth-order tensor).
  Gridap returned a scalar (it took them as second-order tensors). `H ⋅² A` and `A ⋅² H` are new.
- `transpose` and `'` apply the cyclic shift `Aᵀ[i,j,k] = A[k,i,j]` to Gridap's
  `ThirdOrderTensorValue{D,D,D}` (Gridap returned the tensor unchanged), and no longer to
  `TensorValue{D,D²}`, which is now transposed as a matrix (`TensorValue{D²,D}`).
  `transpose_IJK_KIJ` is removed.
- `inner(::VectorValue, ::TensorValue)` for any sizes is removed (it always failed).
- `W ⊗ v` with `W::TensorValue{D,D²}` now returns the flattened `TensorValue{D²,D²}` (it fell back
  to Gridap's `ThirdOrderTensorValue{D,D,D²}`).

### Fixes

Two former operations computed something different from their name:

- `⊗₁₂₃⁴` computed `Wₖᵢⱼ vₗ` instead of `Wᵢⱼₖ vₗ`. `⊗ᵢⱼₖˡ` computes `Wᵢⱼₖ vₗ`.
- `⊙₁₂₃₄²⁴` (`contraction_IJKL_JL`) returned the transpose of its name, `Rₖᵢ = Σ Hᵢⱼₖₗ Aⱼₗ`.
  `⊗ᵢₘⱼₙᵐⁿ` computes `Rᵢⱼ = Σ Hᵢₘⱼₙ Aₘₙ`.

Neither was used by the constitutive models. All the other operations return results bitwise
identical to the former implementation.

### Constitutive models

`ViscousModels.jl` writes the composition of the fourth-order tangents of the return mapping as a
double contraction, `(γα * ∂Se∂Ce_trial) ⋅² ∂Ce_trial_∂C`, with the same result. The other products
of fourth-order tensors with `*` are kept until `*` is removed.
