
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

function (obj::ViscousPolyconvex)(_...)
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

function Ψv(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  0.5μ * (C ⊙ invCv -3*IIIc^(1/3))
end

function Sv(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  μ * (invCv -IIIc^(1/3) * inv(C))
end

function ∂Sv∂C_Cᵥfix(obj::ViscousPolyconvex, C, invCv)
  μ = obj.μ
  IIIc = det(C)
  G    = cof(C)
  μ * IIIc^(-2/3) * (2/3 * (1/IIIc) * G ⊗ G - ×ᵢ⁴(C))
end

# --- Implementation of derivatives ---

function energy(obj::ViscousPolyconvex, F, Fn, Cvn)
  C, Cn = Cauchy.((F, Fn))
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  Ψv(obj, C, invCv)
end

function first_piola(obj::ViscousPolyconvex, F, Fn, Cvn)
  C, Cn = Cauchy.((F, Fn))
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  F * Sv(obj, C, invCv)
end

function tangent(obj::ViscousPolyconvex, F, Fn, Cvn)
  C, Cn = Cauchy.((F, Fn))
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  H1 = obj.μ * ∂Cv⁻¹∂C(obj, C, Cn, Cvn)
  H2 = ∂Sv∂C_Cᵥfix(obj, C, invCv)
  H3 = I3 ⊗₁₃²⁴ Sv(obj, C, invCv)
  DCDF = F' ⊗₁₃²⁴ I3 + I3 ⊗₁₄²³ F'
  0.5 * DCDF' · (H1 + H2) · DCDF + H3
end

function dissipation(obj::ViscousPolyconvex, F, Fn, Cvn)
  γ = obj.μ / obj.τ
  Τ = obj.τ / obj.Δt[]
  C, Cn = Cauchy.((F, Fn))
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  Cv = inv(invCv)
  λ_algo = 1 / (det(invC + Τ*inv(Cvn))^(1/3) - Τ)  # λ = 3 / (Cv ⊙ invC)
  -0.5γ * (C -λ_algo*Cv) ⊙ (invC - (1/λ_algo)*invCv)
end

# --- Return mapping and derivatives for the underlying neo-Hookean ---

function Cv⁻¹(obj::ViscousPolyconvex, C, Cn, Cvn)
  Τ = obj.Δt[] / obj.τ
  B = Τ * inv(C) + inv(Cvn)
  det(B)^(-1/3) * B
end

function ∂Cv⁻¹∂C(obj::ViscousPolyconvex, C, Cn, Cvn)
  Τ = obj.Δt[] / obj.τ
  invC = inv(C)
  B = Τ * invC + inv(Cvn)
  ∂invCv∂B = det(B)^(-1/3) * (IIsym(I3) - (1/3) * (B ⊗ inv(B)))
  ∂B∂C = -Τ * IIsym(invC)
  ∂invCv∂B · ∂B∂C
end

function return_mapping(obj::ViscousPolyconvex, C, Cn, Cvn)
  invCv = Cv⁻¹(obj, C, Cn, Cvn)
  inv(invCv)
end

function return_mapping(obj::ViscousPolyconvex)
  (A, F, Fn) -> begin
    C, Cn = Cauchy.((F, Fn))
    Cv = return_mapping(obj, C, Cn, A)
    (true, Cv)
  end
end
