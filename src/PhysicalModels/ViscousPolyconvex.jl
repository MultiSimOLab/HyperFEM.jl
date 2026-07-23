
"""
Polyconvex viscoelastic constitutive model for the set of variables `{F, J, Cᵥ}`,
where `F` is the deformation gradient, `J` is the jacobian and `Cᵥ` is the viscous strain.

### Key features:
 - **Factorization-free:** Fast calculation of the intermediate state without matrix factorizations.
 - **Neo-Hookean equilibrium:** Uses a neo-Hookean expression for the underlying equilibrium term.
 - **Distortional invariants:** Formulated using distortional invariants rather than deviatoric ones.

### Fields
- `μ::Float64`: Shear modulus.
- `τ::Float64`: Relaxation time.
- `Δt::Ref{Float64}`: `Reference` to the time step.
 """
struct ViscousPolyconvex <: Visco
  μ::Float64
  τ::Float64
  Δt::Ref{Float64}
  ViscousPolyconvex(; μ::Real, τ::Real) = new(Float64(μ), Float64(τ), 0.0)
end

function (obj::ViscousPolyconvex)(::Any)
  Ψ(F, Fn, A)      = energy(obj, F, Fn, A)
  ∂Ψ∂F(F, Fn, A)   = first_piola(obj, F, Fn, A)
  ∂Ψ∂F∂F(F, Fn, A) = tangent(obj, F, Fn, A)
  Ψ, ∂Ψ∂F, ∂Ψ∂F∂F
end

# --- Underlying neo-Hookean model in terms of viscous distortional invariants ---

function Ψv(obj::ViscousPolyconvex, C, Cv)
  μ = obj.μ
  IIIc = det(C)
  0.5μ * (C ⊗ inv(Cv) -3*IIIc^(1/3))
end

function Sv(obj::ViscousPolyconvex, C, Cv)
  μ = obj.μ
  IIIc = det(C)
  μ * (inv(Cv) -IIIc^(1/3) * inv(C))
end

function Hv(obj::ViscousPolyconvex, C, Cv)
  μ = obj.μ
  IIIc = det(C)
  invC = inv(C)
  -μ * IIIc^(1/3) * (1/3 * invC ⊗ invC - ×ᵢ⁴(invC))
end

# --- Implementation of derivatives ---

function energy(obj::ViscousPolyconvex, F, Fn, A)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj::ViscousPolyconvex, C, Cn, A)
  Ψv(obj, C, Cv)
end

function first_piola(obj::ViscousPolyconvex, F, Fn, A)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj::ViscousPolyconvex, C, Cn, A)
  F * Sv(obj, C, Cv)
end

function tangent(obj::ViscousPolyconvex, F, Fn, A)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj::ViscousPolyconvex, C, Cn, A)
  H = Hv(obj, C, Cv)
  DCDF = F' ⊗₁₃²⁴ I3 + I3 ⊗₁₄²³ F'
  DCDF' · H · DCDF
end

function return_mapping(obj::ViscousPolyconvex, C, Cn, A)
  τ = obj.τ
  Δt = obj.Δt[]
  Cvn = TensorValue{3,3}(A[1:9]...)
  invC = inv(C)
  B = Δt/τ * invC + inv(Cvn)
  invCv = det(B)^(1/3) * B
  Cv = inv(Cv)
  λv = 3 / Cv ⊙ invC
end
