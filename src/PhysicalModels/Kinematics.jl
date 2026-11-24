

"""
    cof(F::TensorValue)::TensorValue

Cofactor of the deformation gradient, `H = det(F) F⁻ᵀ`. In 3D, `H = ½ F × F`.
"""
@inline cof(F::TensorValue{3,3}) = 0.5F × F
@inline cof(F::TensorValue{2,2}) = TensorValue(F[2, 2], -F[1, 2], -F[2, 1], F[1, 1])


"""
    ∂H∂F(F::TensorValue)::FourthOrderTensorValue

Derivative of the cofactor, `∂H/∂F`. In 3D, `∂H/∂F = ×ᵢ⁴(F)`; in 2D it is constant.
"""
@inline ∂H∂F(F::TensorValue{3,3}) = ×ᵢ⁴(F)
@inline function ∂H∂F(::TensorValue{2,2,T}) where {T}
  o, z = one(T), zero(T)
  FourthOrderTensorValue{2}(z, z, z, o, z, z, -o, z, z, -o, z, z, o, z, z, z)
end


# Isochoric first invariant, I₁iso = J^(-2/3) F : F (3D), and its derivatives.
# `_Jconst`: partial derivative treating F and J as independent variables.
I₁iso(F::TensorValue{3,3}) = det(F)^(-2 / 3) * F ⊙ F
∂I₁iso∂F_Jconst(F::TensorValue{3,3}) = 2 * det(F)^(-2 / 3) * F
∂I₁iso∂J(F::TensorValue{3,3}) = -(2 / 3) * det(F)^(-5 / 3) * F ⊙ F
∂∂I₁iso∂FF_Jconst(F::TensorValue{3,3}) = 2 * det(F)^(-2 / 3) * 𝕀3D
∂∂I₁iso∂JJ(F::TensorValue{3,3}) = (10 / 9) * det(F)^(-8 / 3) * F ⊙ F
∂∂I₁iso∂FJ_Jconst(F::TensorValue{3,3}) = -(4 / 3) * det(F)^(-5 / 3) * F

∂I₁iso∂F(F::TensorValue{3,3}) = ∂I₁iso∂F_Jconst(F) + ∂I₁iso∂J(F) * cof(F)
∂∂I₁iso∂FF(F::TensorValue{3,3}) = ∂∂I₁iso∂FF_Jconst(F) + ∂∂I₁iso∂FJ_Jconst(F) ⊗ cof(F) + cof(F) ⊗ ∂∂I₁iso∂FJ_Jconst(F) +
                                 ∂∂I₁iso∂JJ(F) * cof(F) ⊗ cof(F) + ∂I₁iso∂J(F) * ∂H∂F(F)


function trAA(A::TensorValue{3, 3, T, N}) where {T, N}
  return sum(A.data[i]*A.data[i] for i in 1:N)
end


"""
Jacobian regularization
"""
function logreg(J; threshold=0.01)
  if J >= threshold
    return log(J)
  else
    return log(threshold) - (3.0 / 2.0) + (2 / threshold) * J - (1 / (2 * threshold^2)) * J^2
  end
end


"""
Jacobian regularization
"""
function ∂log∂J(J; threshold=0.01)
  if J >= threshold
    1 / J
  else
    2 / threshold - J / (threshold^2)
  end
end


"""
Jacobian regularization
"""
function ∂∂log∂JJ(J; threshold=0.01)
  if J >= threshold
    -1 / (J^2)
  else
    -1 / (threshold^2)
  end
end


