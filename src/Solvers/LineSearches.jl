abstract type AbstractLineSearch end

"""
    search!(ls, w, op, x, dx, b) -> α

Step length `α` along the Newton direction `dx` from `x` chosen by the line search `ls`. On entry
`b = -R(x)`; on return `b = R(x + α dx)` (the residual Newton continues with). `w` is a work vector
of the size of `x` (the trial points `x + α dx` are built there, without allocating). Line searches
that implement the former callable interface `ls(x, dx, b, op)` keep working through the fallback.
"""
search!(ls::AbstractLineSearch, w, op, x, dx, b) = ls(x, dx, b, op)

"""
    update_cellstate!(ls, xh, dxh) -> α

Largest safe step along the increment `dxh` (an `FEFunction`) from the state `xh` according to the
line search (1 for the line searches without an injectivity test). Used by the drivers for the safe
load factor of the Dirichlet projection.
"""
update_cellstate!(::AbstractLineSearch, xh, dxh) = 1.0


"""
    LineSearch()

Full Newton step (`α = 1`).
"""
struct LineSearch <: AbstractLineSearch end

function search!(::LineSearch, w, op, x, dx, b)
  w .= x .+ dx
  residual!(b, op, w)
  return 1.0
end

function (obj::LineSearch)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
  residual!(b, op, x + dx)
  return 1.0
end


"""
    Roman_LS(; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95)

Backtracking on the directional derivative of the residual: starting from `α = 1`, `α ← ρ α` until
`R(x + α dx)·dx ≤ c R(x)·dx` (at most `maxiter` reductions, `α > αmin`).
"""
struct Roman_LS <: AbstractLineSearch
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  function Roman_LS(; maxiter::Int64=50, αmin::Float64=1e-16, ρ::Float64=0.5, c::Float64=0.95)
    new(maxiter, αmin, ρ, c)
  end
end

function search!(obj::Roman_LS, w, op, x, dx, b)
  maxiter, αmin, ρ, c = obj.maxiter, obj.αmin, obj.ρ, obj.c
  m = 0
  α = 1.0
  R₀ = dot(b, dx)
  while α > αmin && m < maxiter
    w .= x .+ α .* dx
    residual!(b, op, w)
    R = dot(b, dx)
    if R <= c * R₀
      break
    end
    α *= ρ
    m += 1
  end
  return α
end

function (obj::Roman_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
  search!(obj, similar(x), op, x, dx, b)
end


"""
    Injectivity_Preserving_LS(α::CellState, U, V; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95, β=0.95, maskphys=0)

Backtracking on `|R·dx|` (per field when `maskphys ≠ 0`) starting from the injectivity-preserving
step estimated with a secant on `det F` at the quadrature points of the state `α` (field
`maskphys` of the solution, or the whole solution). Kept with its current behaviour for one release;
the exact root of `det(F + α ∇δu)` is the `InjectivityLS` of the next commits.
"""
struct Injectivity_Preserving_LS{A} <: AbstractLineSearch
  α::CellState
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  β::CellField
  maskphys::Int64
  caches::A
  function Injectivity_Preserving_LS(α::CellState, U, V; maxiter::Int64=50, αmin::Float64=1e-16, ρ::Float64=0.5, c::Float64=0.95, β::Float64=0.95, maskphys::Int64=0)
    # extract parent indices
    ranges = if maskphys == 0
      (1:U.space.nfree,)
    else
      nfree = ntuple(i -> U[i].space.nfree, length(U))
      offsets = cumsum((0, nfree...))
      ntuple(i -> offsets[i]+1:offsets[i+1], length(U))
    end
    caches = (U, V, ranges)
    new{typeof(caches)}(α, maxiter, αmin, ρ, c, CellField(β, α.points.trian), maskphys, caches)
  end
end

@inline _directional(b, dx, ranges) = sum(abs(dot(view(b, r), view(dx, r))) for r in ranges)

function search!(obj::Injectivity_Preserving_LS, w, op, x, dx, b)
  maxiter, αmin, ρ, c = obj.maxiter, obj.αmin, obj.ρ, obj.c
  U, V, ranges = obj.caches
  xh = FEFunction(U, x)
  dxh = FEFunction(V, dx)
  α = update_cellstate!(obj, xh, dxh)
  m = 0
  R₀ = _directional(b, dx, ranges)
  while α > αmin && m < maxiter
    w .= x .+ α .* dx
    residual!(b, op, w)
    R = _directional(b, dx, ranges)
    if abs(R) <= abs(c * R₀)
      break
    end
    α *= ρ
    m += 1
  end
  return α
end

function (obj::Injectivity_Preserving_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
  search!(obj, similar(x), op, x, dx, b)
end

function InjectivityCheck(α, ∇u, ∇du, β)
  F = ∇u + one(∇u)
  J = det(F)
  H = J * inv(F)'
  return true, min(β * abs((-J) / (det(∇du) + tr(H' * ∇du))), 1.0)
end

function update_cellstate!(obj::Injectivity_Preserving_LS, xh, dxh)
  uh = obj.maskphys == 0 ? xh : xh[obj.maskphys]
  duh = obj.maskphys == 0 ? dxh : dxh[obj.maskphys]
  update_state!(InjectivityCheck, obj.α, ∇(uh)', ∇(duh)', obj.β)
  return minimum(minimum((obj.α.values)))
end
