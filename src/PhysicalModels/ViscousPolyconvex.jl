
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
- `Δt::Base.RefValue{Float64}`: `Reference` to the time step.
 """
struct ViscousPolyconvex <: Visco
  μ::Float64
  τ::Float64
  Δt::Base.RefValue{Float64}
  ViscousPolyconvex(; μ::Real, τ::Real) = new(Float64(μ), Float64(τ), Ref(0.0))
end

function (obj::ViscousPolyconvex)(::Float64=0.0)
  Ψ(F, Fn, A)      = energy(obj, F, Fn, A)
  ∂Ψ∂F(F, Fn, A)   = first_piola(obj, F, Fn, A)
  ∂Ψ∂F∂F(F, Fn, A) = tangent(obj, F, Fn, A)
  Ψ, ∂Ψ∂F, ∂Ψ∂F∂F
end

function update_time_step!(obj::ViscousPolyconvex, Δt::Float64)
  obj.Δt[] = Δt
end

function Gridap.CellData.CellState(::ViscousPolyconvex, Cv₀::TensorValue, points::Measure)
  CellState(Cv₀, points)
end

function Gridap.CellData.CellState(obj::ViscousPolyconvex, points::Measure)
  CellState(I3, points)
end

function Gridap.CellData.update_state!(obj::ViscousPolyconvex, A, F, Fn)
  update_state!(return_mapping(obj), A, F, Fn)
end

function Dissipation(obj::ViscousPolyconvex)
  D(F, Fn, A) = dissipation(obj, F, Fn, A)
end

# --- Underlying neo-Hookean model in terms of viscous distortional invariants ---

@inline function Ψv(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  0.5μ * (C ⊙ invCv -3*∛(IIIc))
end

@inline function Sv(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  μ * (invCv -∛(IIIc) * inv(C))
end

@inline function ∂Sv∂C_Cᵥfix(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  G    = cof(C)
  μ * (1/∛(IIIc)^2) * (2/3 * (1/IIIc) * G ⊗ G - ×ᵢ⁴(C))
end

# --- Implementation of derivatives ---

@inline function energy(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  Ψv(obj, C, invCv)
end

@inline function first_piola(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  F * Sv(obj, C, invCv)
end

@inline function tangent(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  invCv, ∂invCv = ∂Cv⁻¹∂C(obj, C, Cn, Cvn)
  H1 = obj.μ * ∂invCv
  H2 = ∂Sv∂C_Cᵥfix(obj, C, invCv)
  H3 = I3 ⊗₁₃²⁴ Sv(obj, C, invCv)
  push_forward_C_to_F(F, H1 + H2) + H3
end

@inline function dissipation(obj::ViscousPolyconvex, F, Fn, Cvn)
  γ = obj.μ / obj.τ
  Τ = obj.τ / obj.Δt[]
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  invC = inv(C)
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  Cv = inv(invCv)
  λ_algo = 1 / (∛(det(invC + Τ*inv(Cvn))) - Τ)  # λ = 3 / (Cv ⊙ invC)
  -0.5γ * (C -λ_algo*Cv) ⊙ (invC - (1/λ_algo)*invCv)
end

# --- Return mapping and derivatives for the underlying neo-Hookean ---

@inline function Cv⁻¹(obj::ViscousPolyconvex, C, Cn, Cvn)
  Τ = obj.Δt[] / obj.τ
  B = Τ * inv(C) + inv(Cvn)
  1/∛(det(B)) * B
end

@inline function ∂Cv⁻¹∂C(obj::ViscousPolyconvex, C, Cn, Cvn)
  Τ = obj.Δt[] / obj.τ
  invC = inv(C)
  B = Τ * invC + inv(Cvn)
  invB = inv(B)
  IIIb = det(B)
  IIIb⁻¹´³ = 1/∛(IIIb)
  M = invC * invB * invC
  invCv = IIIb⁻¹´³ * B
  ∂invCv = -Τ * IIIb⁻¹´³ * (IIsym(invC) - (1/3) * (B ⊗ M))
  (invCv, ∂invCv)
end

@inline function return_mapping(obj::ViscousPolyconvex, C, Cn, Cvn)
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  inv(invCv)
end

function return_mapping(obj::ViscousPolyconvex)
  (A, F, Fn) -> begin
    C = Cauchy(F)
    Cn = Cauchy(Fn)
    Cv = return_mapping(obj, C, Cn, A)
    (true, Cv)
  end
end
