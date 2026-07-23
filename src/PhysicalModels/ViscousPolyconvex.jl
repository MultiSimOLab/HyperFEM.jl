
"""
A fast implementation of a viscous constitutive model with three key points:
 - The computation of the intermediate state does not require factorization
 - The underlying equilibrium term is neohookean
 - Uses distortional invariants instead of deviatoric invariants

The combination of the three features aims to provide a fast implementation.
"""
struct ViscousPolyconvex <: Visco
  μ::Float64
  τ::Float64
  Δt::REf{Float64}
  ViscousPolyconvex(; μ::Real, τ::Real) = new(Float64(μ), Float64(τ), 0.0)
end

function (obj::ViscousPolyconvex)(::Any)
  Ψe, Se, ∂Se∂Ce   = SecondPiola(obj)
  Ψ(F, Fn, A)      = Energy(obj, Ψe, Se, ∂Se∂Ce, F, Fn, A)
  ∂Ψ∂F(F, Fn, A)   = FirstPiola(obj, Se, ∂Se∂Ce, F, Fn, A)
  ∂Ψ∂F∂F(F, Fn, A) = Tangent(obj, Se, ∂Se∂Ce, F, Fn, A)
  Ψ, ∂Ψ∂F, ∂Ψ∂F∂F
end

function SecondPiola(obj::ViscousPolyconvex)

end

function Energy(obj::ViscousPolyconvex, Ψe, Se, ∂Se∂Ce, F, Fn, A)
  
end

function FirstPiola(obj::ViscousPolyconvex, Se, ∂Se∂Ce, F, Fn, A)
  
end

function Tanget(obj::ViscousPolyconvex, Se, ∂Se∂Ce, F, Fn, A)
  
end

