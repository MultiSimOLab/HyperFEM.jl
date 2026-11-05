# ===================
# Hessian of FE fields
# ===================

using Gridap.TensorValues: ThirdOrderTensorValue
using Gridap.Arrays: lazy_map
using Gridap.Fields: Broadcasting, Operation
using Gridap.CellData: CellField, DomainStyle, ReferenceDomain, get_data, get_triangulation, similar_cell_field
using Gridap.Geometry: get_cell_map

"""
    push_hessian(Ĥ, ∇̂u, Jt, ∇̂∇̂φ)

Physical Hessian at one point from reference-domain quantities: the reference Hessian `Ĥ`, the reference
gradient `∇̂u`, the gradient of the cell map `Jt = ∇̂φ` (Gridap's convention, the transpose of the usual
Jacobian) and the Hessian of the cell map `∇̂∇̂φ`:

    H_abk = Jinv_ai Jinv_bj (Ĥ_ijk - ∇̂∇̂φ_ijc ∇u_ck),   ∇u = Jinv ⋅ ∇̂u,   Jinv = inv(Jt)

The curvature term `∇̂∇̂φ` vanishes for affine maps (simplices, parallelograms) and is kept for bilinear
quadrilaterals and trilinear hexahedra.
"""
function push_hessian(Ĥ::ThirdOrderTensorValue{D,D,N}, ∇̂u::TensorValue{D,N},
                      Jt::TensorValue{D,D}, ∇∇φ::ThirdOrderTensorValue{D,D,D}) where {D,N}
  Jinv = inv(Jt)
  ∇u = Jinv ⋅ ∇̂u
  T = promote_type(eltype(Ĥ), eltype(Jinv))
  ThirdOrderTensorValue{D,D,N}(ntuple(Val(D*D*N)) do l
    a, b, k = Tuple(CartesianIndices((D, D, N))[l])
    s = zero(T)
    for i in 1:D, j in 1:D
      t = Ĥ[i, j, k]
      for c in 1:D
        t -= ∇∇φ[i, j, c] * ∇u[c, k]
      end
      s += Jinv[a, i] * Jinv[b, j] * t
    end
    s
  end)
end

function push_hessian(Ĥ::TensorValue{D,D}, ∇̂u::VectorValue{D},
                      Jt::TensorValue{D,D}, ∇∇φ::ThirdOrderTensorValue{D,D,D}) where D
  Jinv = inv(Jt)
  ∇u = Jinv ⋅ ∇̂u
  G = TensorValue{D,D}(ntuple(Val(D*D)) do l
    i, j = Tuple(CartesianIndices((D, D))[l])
    Ĥ[i, j] - sum(∇∇φ[i, j, c] * ∇u[c] for c in 1:D)
  end)
  Jinv ⋅ G ⋅ transpose(Jinv)
end

"""
    hessian_nonaffine(u::CellField)

Hessian of an FE function or basis `u` (a field defined in the reference domain), valid on any cell map:
affine (triangles, tetrahedra, parallelograms) and non-affine (bilinear quadrilaterals, trilinear
hexahedra). Gridap's `∇∇` only supports affine maps. For a vector field the result is a
`ThirdOrderTensorValue{D,D,N}` with `H[a,b,k] = ∂²u_k/∂x_a∂x_b`, the same layout as Gridap's `∇∇`.

```julia
∫(hessian_nonaffine(u) ⊙ hessian_nonaffine(v))dΩ
∫(jump(hessian_nonaffine(u)) ⊙ jump(hessian_nonaffine(v)))dΓ
```
"""
function hessian_nonaffine(a::CellField)
  @assert DomainStyle(a) == ReferenceDomain() "hessian_nonaffine needs a field defined in the reference domain (an FE function or basis)"
  trian    = get_triangulation(a)
  cell_a   = get_data(a)
  cell_Ĥ   = lazy_map(Broadcasting(∇∇), cell_a)
  cell_∇̂   = lazy_map(Broadcasting(∇), cell_a)
  cell_map = get_cell_map(trian)
  cell_Jt  = lazy_map(∇, cell_map)
  cell_∇∇φ = lazy_map(∇∇, cell_map)
  h = lazy_map(Broadcasting(Operation(push_hessian)), cell_Ĥ, cell_∇̂, cell_Jt, cell_∇∇φ)
  similar_cell_field(a, h, trian, DomainStyle(a))
end
