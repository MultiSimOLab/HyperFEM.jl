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

# Unchecked element access to the MMatrix scratch buffers of `_sym_eigen`, through the raw pointer
# (what StaticArrays does for `a[i, j]` under `@inbounds`). `Pkg.test` runs with --check-bounds=yes,
# which ignores `@inbounds`: the BoundsError path of `a[i, j]` then references the buffer, which
# escapes and is heap allocated. Through these helpers there is no such path, so `_sym_eigen` stays
# allocation-free whatever the bounds-checking mode. Callers guarantee 1 <= i, j <= N.
@inline _uget(a::MMatrix{N,N,T}, i::Int, j::Int) where {N,T} =
  GC.@preserve a unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(a)), (j - 1) * N + i)
@inline function _uset!(a::MMatrix{N,N,T}, x, i::Int, j::Int) where {N,T}
  GC.@preserve a unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(a)), convert(T, x), (j - 1) * N + i)
  return nothing
end

"""
    _sym_eigen(A::SMatrix{N,N}) -> (λ::SVector{N}, Q::SMatrix{N,N})

Eigenvalues and eigenvectors of the symmetric matrix `A` by the cyclic Jacobi method (A = Q Λ Qᵀ).
Allocation-free, also with --check-bounds=yes (see `_uget`).
"""
function _sym_eigen(A::SMatrix{N,N,T}; maxsweeps::Int=50) where {N,T}
  a = MMatrix{N,N,T}(A)
  v = MMatrix{N,N,T}(ntuple(k -> (k - 1) % N == (k - 1) ÷ N ? one(T) : zero(T), Val(N * N)))
  for _ in 1:maxsweeps
    off = zero(T)
    fro = zero(T)
    for q in 1:N, p in 1:N
      apq = _uget(a, p, q)
      fro += apq^2
      p < q && (off += apq^2)
    end
    off <= eps(T)^2 * fro && break
    for p in 1:N-1, q in p+1:N
      apq = _uget(a, p, q)
      iszero(apq) && continue
      θ = (_uget(a, q, q) - _uget(a, p, p)) / (2apq)
      t = iszero(θ) ? one(T) : sign(θ) / (abs(θ) + sqrt(θ^2 + 1))
      c = 1 / sqrt(t^2 + 1)
      s = t * c
      for k in 1:N   # columns p, q
        akp, akq = _uget(a, k, p), _uget(a, k, q)
        _uset!(a, c * akp - s * akq, k, p)
        _uset!(a, s * akp + c * akq, k, q)
      end
      for k in 1:N   # rows p, q
        apk, aqk = _uget(a, p, k), _uget(a, q, k)
        _uset!(a, c * apk - s * aqk, p, k)
        _uset!(a, s * apk + c * aqk, q, k)
      end
      for k in 1:N   # eigenvectors
        vkp, vkq = _uget(v, k, p), _uget(v, k, q)
        _uset!(v, c * vkp - s * vkq, k, p)
        _uset!(v, s * vkp + c * vkq, k, q)
      end
    end
  end
  return SVector{N,T}(ntuple(i -> _uget(a, i, i), Val(N))), SMatrix{N,N,T}(Tuple(v))
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

"""
    regularize_hessian(H::FourthOrderTensorValue, δ) -> FourthOrderTensorValue

Same as for the D²×D² matrix of `H`, `flat_tensor(H)`.
"""
regularize_hessian(H::FourthOrderTensorValue, δ::Real) = FourthOrderTensorValue(Tuple(regularize_hessian(flat_tensor(H), δ)))

"""
    sqrt_sym(A::TensorValue{N,N}) -> TensorValue{N,N}

Principal square root of the symmetric positive (semi)definite tensor `A` (upper triangle, as
`sqrt(Symmetric(A))`): `Q * Diagonal(sqrt.(λ)) * Q'` with `A = Q * Diagonal(λ) * Q'`.
Unlike the general matrix `sqrt`, the result is always real, so it is type stable, and allocation-free.
"""
function sqrt_sym(A::TensorValue{N,N,T}) where {N,T}
  λ, Q = _sym_eigen(_sym_upper(SMatrix{N,N,T}(A.data)))
  R = Q * Diagonal(sqrt.(max.(zero(T), λ))) * Q'
  return TensorValue{N,N,T}(Tuple(R))
end
