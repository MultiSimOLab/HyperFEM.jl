

"""
    cof(A::TensorValue)::TensorValue

Calculate the cofactor of a matrix.
"""
@inline function cof(A::TensorValue)
  0.5A×A
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


function _∂H∂F_2D()
  TensorValue(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
end


function trAA(A::TensorValue{3, 3, T, N}) where {T, N}
  return sum(A.data[i]*A.data[i] for i in 1:N)
end


@generated function Tensorize(A::VectorValue{D,Float64}) where {D}
  str = ""
  for i in 1:D
    str *= "A.data[$i], "
  end
  Meta.parse("TensorValue($str)")
end