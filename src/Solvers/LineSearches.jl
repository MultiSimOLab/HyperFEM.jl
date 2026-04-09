abstract type AbstractLineSearch end



struct LineSearch <: AbstractLineSearch
  function LineSearch()
    new()
  end
  function (obj::LineSearch)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
    α = 1.0
    residual!(b, op, x + α * dx)
    return α
  end

end


function update_cellstate!(obj::LineSearch, xh, dxh)
  return 1.0
end



struct Roman_LS <: AbstractLineSearch
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  function Roman_LS(; maxiter::Int64=50, αmin::Float64=1e-16, ρ::Float64=0.5, c::Float64=0.95)
    new(maxiter, αmin, ρ, c)
  end


  function (obj::Roman_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)

    maxiter, αmin, ρ, c = obj.maxiter, obj.αmin, obj.ρ, obj.c
    m = 0
    α = 1.0
    R₀ = b' * dx

    while α > αmin && m < maxiter
      residual!(b, op, x + α * dx)
      R = b' * dx
      if R <= c * R₀
        break
      end
      α *= ρ
      m += 1
    end
    return α
  end
end

function update_cellstate!(obj::Roman_LS, xh, dxh)
  return 1.0
end


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
    caches = (U, V)
    new{typeof(caches)}(α, maxiter, αmin, ρ, c, CellField(β, α.points.trian), maskphys, caches)
  end


  function (obj::Injectivity_Preserving_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)

    _, maxiter, αmin, ρ, c = obj.α, obj.maxiter, obj.αmin, obj.ρ, obj.c
    #update cell state
    U, V = obj.caches
    xh = FEFunction(U, x)
    dxh = FEFunction(V, dx)
    α = update_cellstate!(obj, xh, dxh)
    m = 0
    R₀ = b' * dx

    while α > αmin && m < maxiter
      residual!(b, op, x + α * dx)
      R = b' * dx
      if abs(R) <= abs(c * R₀)
        break
      end
      α *= ρ
      m += 1
    end
    return α
  end

end

function InjectivityCheck(α, ∇u, ∇du, β)
  # ε = 1e-6
  F = ∇u + one(∇u)
  J = det(F)
  H = J * inv(F)'
  # if det(F+∇du) < 0.2
  # @show det(F), det(F+∇du)
  # end
  return true, min(β * abs((-J) / (det(∇du) + tr(H' * ∇du))), 1.0)

end

function update_cellstate!(obj::Injectivity_Preserving_LS, xh, dxh)
  uh = obj.maskphys == 0 ? xh : xh[obj.maskphys]
  duh = obj.maskphys == 0 ? dxh : dxh[obj.maskphys]
  update_state!(InjectivityCheck, obj.α, ∇(uh)', ∇(duh)', obj.β)
  return minimum(minimum((obj.α.values)))
end


# function update_cellstate!(obj::Injectivity_Preserving_LS, xh, dxh)
#   update_state!(InjectivityCheck, obj.α, ∇(xh)', ∇(dxh)', obj.β)
#   return minimum(minimum((obj.α.values)))
# end