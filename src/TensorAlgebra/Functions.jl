

"""
    cof(A::TensorValue)::TensorValue

Calculate the cofactor of a matrix.
"""
function cof(A::TensorValue)
  0.5A×A
end


"""
Jacobian regularization
"""
function logreg(J; Threshold=0.01)
  if J >= Threshold
    return log(J)
  else
    return log(Threshold) - (3.0 / 2.0) + (2 / Threshold) * J - (1 / (2 * Threshold^2)) * J^2
  end
end


"""
Fast and dependency-free implementation of erf function, up to 1e-6 precision.
"""
@inline function erf(x::Real)
  p  = 0.3275911
  a1 = 0.254829592
  a2 = -0.284496736
  a3 = 1.421413741
  a4 = -1.453152027
  a5 = 1.061405429

  ax = abs(x)
  t = 1.0 / (1.0 + p*ax)

  y = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)

  r = 1.0 - y*exp(-ax*ax)

  return copysign(r, x)
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