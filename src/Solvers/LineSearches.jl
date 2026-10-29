abstract type AbstractLineSearch end

"""
    search!(ls, w, op, x, dx, b) -> α

Step length `α` along the Newton direction `dx` from `x` chosen by the line search `ls`. On entry
`b = -R(x)`; on return `b = R(x + α dx)` (the residual Newton continues with). `w` is a work vector
of the size of `x` (the trial points `x + α dx` are built there, without allocating). A failed
search returns `α = 0` with `b = R(x)`. Line searches that implement the former callable interface
`ls(x, dx, b, op)` keep working through the fallback.
"""
search!(ls::AbstractLineSearch, w, op, x, dx, b) = ls(x, dx, b, op)

"""
    update_cellstate!(ls, xh, dxh) -> α

Largest safe step along the increment `dxh` (an `FEFunction`) from the state `xh` according to the
line search (1 for the line searches without an injectivity test). Former interface of the drivers
for the safe load factor of the Dirichlet projection; `InjectivityLS` asks the model instead.
"""
update_cellstate!(::AbstractLineSearch, xh, dxh) = 1.0

# residual at x + α dx, built in w
@inline function _trial_residual!(b, w, op, x, dx, α)
  w .= x .+ α .* dx
  residual!(b, op, w)
end


"""
    FullStepLS()

Full Newton step (`α = 1`). Former name `LineSearch()`.
"""
struct FullStepLS <: AbstractLineSearch end

function search!(::FullStepLS, w, op, x, dx, b)
  w .= x .+ dx
  residual!(b, op, w)
  return 1.0
end

function (obj::FullStepLS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
  residual!(b, op, x + dx)
  return 1.0
end


"""
    BacktrackingLS(; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95)

Backtracking on the directional derivative of the residual: starting from `α = 1`, `α ← ρ α` until
`R(x + α dx)·dx ≤ c R(x)·dx` (at most `maxiter` reductions, `α > αmin`); the last `α` is taken if
the condition is never met. Former name `Roman_LS`.
"""
struct BacktrackingLS <: AbstractLineSearch
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  function BacktrackingLS(; maxiter::Integer=50, αmin::Real=1e-16, ρ::Real=0.5, c::Real=0.95)
    new(maxiter, αmin, ρ, c)
  end
end

function search!(obj::BacktrackingLS, w, op, x, dx, b)
  maxiter, αmin, ρ, c = obj.maxiter, obj.αmin, obj.ρ, obj.c
  m = 0
  α = 1.0
  R₀ = dot(b, dx)
  while α > αmin && m < maxiter
    _trial_residual!(b, w, op, x, dx, α)
    R = dot(b, dx)
    if R <= c * R₀
      break
    end
    α *= ρ
    m += 1
  end
  return α
end

function (obj::BacktrackingLS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
  search!(obj, similar(x), op, x, dx, b)
end


"""
    InjectivityLS(; jmin=0.0, armijo=false, c1=1e-4, β=1-1e-6, ρ=0.5, c=0.95, maxiter=50, αmin=1e-16)

Injectivity-preserving line search: the step starts from the exact injectivity limit — `β` times
the smallest root of `det(I + ∇u + α ∇δu) = jmin` over the quadrature points of the model
(`injectivity = …` in the driver; quadratic in 2D, cubic in 3D, see [`injectivity_step`](@ref)) —
and is then reduced by `ρ` until it is accepted:

- `armijo = false`: backtracking on `|R(x + α dx)·dx| ≤ c |R(x)·dx|` (at most `maxiter` reductions);
- `armijo = true`: Armijo on the incremental energy of the model (`energy = …` in the driver),
  `E(x + α dx) ≤ E(x) + c1 α R(x)·dx`, a non-finite energy counting as a rejection.

Exhausting the reductions, or (Armijo) a direction that is not of descent (`R·dx ≥ 0`), is a
failure: `α = 0`, the residual left at `x`, and the Newton reacts according to its tangent strategy.
Replaces `Injectivity_Preserving_LS` (secant estimate of the step).
"""
struct InjectivityLS <: AbstractLineSearch
  jmin::Float64
  armijo::Bool
  c1::Float64
  β::Float64
  ρ::Float64
  c::Float64
  maxiter::Int
  αmin::Float64
  function InjectivityLS(; jmin::Real=0.0, armijo::Bool=false, c1::Real=1e-4, β::Real=1 - 1e-6, ρ::Real=0.5,
                         c::Real=0.95, maxiter::Integer=50, αmin::Real=1e-16)
    0 <= jmin || throw(ArgumentError("InjectivityLS: jmin must be ≥ 0"))
    0 < β <= 1 || throw(ArgumentError("InjectivityLS: β must be in (0, 1]"))
    0 < ρ < 1 || throw(ArgumentError("InjectivityLS: ρ must be in (0, 1)"))
    new(jmin, armijo, c1, β, ρ, c, maxiter, αmin)
  end
end

function search!(obj::InjectivityLS, w, op, x, dx, b)
  α = max_step(op, x, dx, obj.β, obj.jmin)
  if obj.armijo
    g = -dot(b, dx)                       # R(x)·dx (b = -R(x) on entry)
    ok = false
    if g < 0
      E0 = energy(op, x)
      m = 0
      while m < obj.maxiter
        w .= x .+ α .* dx
        Eα = energy(op, w)
        if isfinite(Eα) && Eα <= E0 + obj.c1 * α * g
          ok = true
          break
        end
        α *= obj.ρ
        m += 1
      end
    end
    if !ok
      residual!(b, op, x)
      return 0.0
    end
    _trial_residual!(b, w, op, x, dx, α)
    return α
  else
    R₀ = abs(dot(b, dx))
    m = 0
    while true
      _trial_residual!(b, w, op, x, dx, α)
      R = abs(dot(b, dx))
      R <= obj.c * R₀ && return α
      α *= obj.ρ
      m += 1
      if m >= obj.maxiter || α <= obj.αmin
        residual!(b, op, x)
        return 0.0
      end
    end
  end
end


# Former names, deprecated for one release
Base.@deprecate_binding LineSearch FullStepLS
Base.@deprecate_binding Roman_LS BacktrackingLS

"""
    Injectivity_Preserving_LS(α::CellState, U, V; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95, β=0.95, maskphys=0)

Deprecated: use [`InjectivityLS`](@ref) with `injectivity = …` in the driver (exact root of
`det(F + α ∇δu)` instead of a secant estimate). Kept with its behaviour for one release:
backtracking on `|R·dx|` (per field when `maskphys ≠ 0`) from the secant estimate of the
injectivity-preserving step at the quadrature points of the state `α`.
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
    _trial_residual!(b, w, op, x, dx, α)
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
