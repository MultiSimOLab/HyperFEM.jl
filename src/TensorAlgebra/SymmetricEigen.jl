#*******************************************************************************
#    Projection of symmetric tangents onto {eigenvalues >= δ}, without allocations
#*******************************************************************************
#
# `regularize_hessian(C, δ)` returns Q * Diagonal(max.(δ, λ)) * Q' for the eigen-decomposition
# C = Q * Diagonal(λ) * Q' of the symmetric part read from the upper triangle (as
# `eigen(Symmetric(C))` does). Everything runs on stack-allocated StaticArrays: StaticArrays only
# has an allocation-free `eigen` for 2x2 and 3x3, larger sizes go through LAPACK on a heap copy.
#   - Fast path: when every eigenvalue is > δ (checked with a Cholesky factorization of C - δI)
#     the projection is the identity and the (symmetrized) tangent is returned as it is.
#   - Otherwise: cyclic Jacobi eigen-decomposition and reconstruction.

"Symmetric matrix built from the upper triangle of `A` (the part read by `Symmetric(A)`)."
@inline function _sym_upper(A::SMatrix{N,N,T}) where {N,T}
  SMatrix{N,N,T}(ntuple(k -> begin
      i = (k - 1) % N + 1
      j = (k - 1) ÷ N + 1
      i <= j ? A[i, j] : A[j, i]
    end, Val(N * N)))
end

"`true` when every eigenvalue of the symmetric matrix `A` is greater than `δ` (Cholesky of `A - δI`)."
@inline function _all_eigs_above(A::SMatrix{N,N,T}, δ) where {N,T}
  L = MMatrix{N,N,T}(undef)
  @inbounds for j in 1:N
    s = A[j, j] - δ
    for k in 1:j-1
      s -= L[j, k]^2
    end
    s > 0 || return false
    L[j, j] = sqrt(s)
    for i in j+1:N
      t = A[i, j]
      for k in 1:j-1
        t -= L[i, k] * L[j, k]
      end
      L[i, j] = t / L[j, j]
    end
  end
  return true
end

"""
    _sym_eigen(A::SMatrix{N,N}) -> (λ::SVector{N}, Q::SMatrix{N,N})

Eigenvalues and eigenvectors of the symmetric matrix `A` by the cyclic Jacobi method (A = Q Λ Qᵀ).
"""
function _sym_eigen(A::SMatrix{N,N,T}; maxsweeps::Int=50) where {N,T}
  a = MMatrix{N,N,T}(A)
  v = MMatrix{N,N,T}(ntuple(k -> (k - 1) % N == (k - 1) ÷ N ? one(T) : zero(T), Val(N * N)))
  for _ in 1:maxsweeps
    off = zero(T)
    fro = zero(T)
    @inbounds for q in 1:N, p in 1:N
      fro += a[p, q]^2
      p < q && (off += a[p, q]^2)
    end
    off <= eps(T)^2 * fro && break
    @inbounds for p in 1:N-1, q in p+1:N
      apq = a[p, q]
      iszero(apq) && continue
      θ = (a[q, q] - a[p, p]) / (2apq)
      t = iszero(θ) ? one(T) : sign(θ) / (abs(θ) + sqrt(θ^2 + 1))
      c = 1 / sqrt(t^2 + 1)
      s = t * c
      for k in 1:N   # columns p, q
        akp, akq = a[k, p], a[k, q]
        a[k, p] = c * akp - s * akq
        a[k, q] = s * akp + c * akq
      end
      for k in 1:N   # rows p, q
        apk, aqk = a[p, k], a[q, k]
        a[p, k] = c * apk - s * aqk
        a[q, k] = s * apk + c * aqk
      end
      for k in 1:N   # eigenvectors
        vkp, vkq = v[k, p], v[k, q]
        v[k, p] = c * vkp - s * vkq
        v[k, q] = s * vkp + c * vkq
      end
    end
  end
  return SVector{N,T}(ntuple(i -> a[i, i], Val(N))), SMatrix{N,N,T}(v)
end

"""
    regularize_hessian(C::TensorValue{N,N}, δ) -> TensorValue{N,N}

Tangent `C` with its eigenvalues bounded below by `δ`: `Q * Diagonal(max.(δ, λ)) * Q'` with
`C = Q * Diagonal(λ) * Q'` (upper triangle of `C`, as `eigen(Symmetric(C))`). Allocation-free.
"""
function regularize_hessian(C::TensorValue{N,N,T}, δ::Real) where {N,T}
  A = _sym_upper(SMatrix{N,N,T}(C.data))
  _all_eigs_above(A, δ) && return TensorValue{N,N,T}(Tuple(A))
  λ, Q = _sym_eigen(A)
  R = Q * Diagonal(max.(T(δ), λ)) * Q'
  return TensorValue{N,N,T}(Tuple(R))
end
