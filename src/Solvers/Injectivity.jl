# Exact injectivity step: along a direction G = ∇δu from F = I + ∇u the Jacobian J(α) = det(F + α G) is a
# polynomial in α (quadratic in 2D, cubic in 3D). The admissible step at a quadrature point is β times
# the smallest root of J(α) = J_min in (0, 1] (J(α) = 0 at points already below J_min), 1 if there is
# none; the step of the field is the minimum over the quadrature points.

# Coefficients (c, t, D, E) of det(F + α G) - jm = c + t α + D α² + E α³
@inline function _det_coefficients(F::TensorValue{2,2,T}, G::TensorValue{2,2,T}, jm) where {T}
  J = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
  t = F[1, 1] * G[2, 2] + F[2, 2] * G[1, 1] - F[1, 2] * G[2, 1] - F[2, 1] * G[1, 2]   # cof(F) : G
  D = G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1]
  (J - jm, t, D, zero(T)), J
end

# Cofactor matrix; TensorValue stores its entries column-major, so the arguments are c11, c21, c31, c12, …
@inline function _cof3(A::TensorValue{3,3,T}) where {T}
  TensorValue{3,3,T}(
    A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2], A[1, 3] * A[3, 2] - A[1, 2] * A[3, 3], A[1, 2] * A[2, 3] - A[1, 3] * A[2, 2],
    A[2, 3] * A[3, 1] - A[2, 1] * A[3, 3], A[1, 1] * A[3, 3] - A[1, 3] * A[3, 1], A[1, 3] * A[2, 1] - A[1, 1] * A[2, 3],
    A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1], A[1, 2] * A[3, 1] - A[1, 1] * A[3, 2], A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
end

# det(F + αG) = det F + α cof(F):G + α² F:cof(G) + α³ det G
@inline function _det_coefficients(F::TensorValue{3,3,T}, G::TensorValue{3,3,T}, jm) where {T}
  cF = _cof3(F)
  cG = _cof3(G)
  J = F[1, 1] * cF[1, 1] + F[1, 2] * cF[1, 2] + F[1, 3] * cF[1, 3]
  E = G[1, 1] * cG[1, 1] + G[1, 2] * cG[1, 2] + G[1, 3] * cG[1, 3]
  (J - jm, cF ⊙ G, F ⊙ cG, E), J
end

# Smallest root of c + t α + D α² in (0, 1]; Inf if there is none
function _smallest_root(c, t, D)
  r = Inf
  if D == 0
    if t != 0
      a = -c / t
      0 < a <= 1 && (r = a)
    end
  else
    disc = t * t - 4D * c
    disc < 0 && return r
    s = sqrt(disc)
    q = t >= 0 ? -0.5 * (t + s) : -0.5 * (t - s)
    q == 0 && return r                      # t = 0 and c = 0: double root at 0
    a = q / D
    0 < a <= 1 && (r = a)
    a = c / q
    0 < a <= 1 && (r = min(r, a))
  end
  r
end

# Smallest root of p(α) = c + t α + D α² + E α³ in (0, 1]; Inf if there is none. The cubic is monotone
# between its critical points, so the first monotone segment of (0, 1] with a sign change holds the
# smallest root, found by bisection to machine precision.
function _smallest_root(c, t, D, E)
  E == 0 && return _smallest_root(c, t, D)
  p(α) = c + α * (t + α * (D + α * E))
  # critical points in (0, 1), sorted
  b1, b2 = 1.0, 1.0
  disc = 4D * D - 12E * t
  if disc >= 0
    s = sqrt(disc)
    x1 = (-2D - s) / (6E)
    x2 = (-2D + s) / (6E)
    x1, x2 = min(x1, x2), max(x1, x2)
    0 < x1 < 1 && (b1 = x1)
    0 < x2 < 1 && (b2 = x2)
    b1 > b2 && ((b1, b2) = (b2, b1))
  end
  a0, p0 = 0.0, c
  for a1 in (b1, b2, 1.0)
    a1 <= a0 && continue
    p1 = p(a1)
    if p1 == 0
      return a1
    elseif p0 * p1 < 0
      lo, hi = a0, a1
      for _ in 1:80
        mid = 0.5 * (lo + hi)
        (mid <= lo || mid >= hi) && break
        pm = p(mid)
        if pm == 0
          return mid
        elseif (pm < 0) == (p0 < 0)
          lo = mid
        else
          hi = mid
        end
      end
      return hi
    end
    a0, p0 = a1, p1
  end
  Inf
end

"""
    injectivity_step(∇u, ∇du, β, jmin) -> α

Step length at a quadrature point: `β` times the smallest root in `(0, 1]` of
`det(I + ∇u + α ∇du) = jmin` (`= 0` where `det(I + ∇u) ≤ jmin`), or `1` if there is none. Exact:
`det` is quadratic in `α` in 2D and cubic in 3D. Allocation-free.
"""
@inline function injectivity_step(∇u::TensorValue{D,D,T}, ∇du::TensorValue{D,D,T}, β, jmin) where {D,T}
  F = ∇u + one(∇u)
  (c, t, Dc, E), J = _det_coefficients(F, ∇du, J_lower(∇u, jmin))
  r = _smallest_root(c, t, Dc, E)
  return isfinite(r) ? β * r : 1.0
end

@inline function J_lower(∇u::TensorValue{2,2}, jmin)
  F = ∇u + one(∇u)
  J = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
  J > jmin ? jmin : zero(jmin)
end
@inline function J_lower(∇u::TensorValue{3,3}, jmin)
  F = ∇u + one(∇u)
  J = F[1, 1] * (F[2, 2] * F[3, 3] - F[2, 3] * F[3, 2]) - F[1, 2] * (F[2, 1] * F[3, 3] - F[2, 3] * F[3, 1]) +
      F[1, 3] * (F[2, 1] * F[3, 2] - F[2, 2] * F[3, 1])
  J > jmin ? jmin : zero(jmin)
end

"""
    injectivity_step(dΩ, uh, duh, β, jmin) -> α

Minimum of [`injectivity_step`](@ref) over the quadrature points of the measure `dΩ` for the
displacement `uh` and the increment `duh` (`CellField`s); the cells are visited one by one with
Gridap's lazy evaluation (no array of all the values is built).
"""
function injectivity_step(dΩ::Measure, uh, duh, β, jmin)
  f(A, B) = injectivity_step(A, B, β, jmin)
  cf = f ∘ (∇(uh)', ∇(duh)')
  vals = cf(get_cell_points(dΩ.quad))
  cache = array_cache(vals)
  α = 1.0
  for c in 1:length(vals)
    v = getindex!(cache, vals, c)
    for q in eachindex(v)
      α = min(α, v[q])
    end
  end
  α
end
