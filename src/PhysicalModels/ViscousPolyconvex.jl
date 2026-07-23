
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

function update_time_step!(obj::ViscousPolyconvex, Δt::Float64)
  obj.Δt[] = Δt
end

function Gridap.CellData.CellState(::ViscousPolyconvex, Cv₀::TensorValue, points::Measure)
  CellState(Cv₀, points)
end

function Gridap.CellData.CellState(obj::ViscousPolyconvex, points::Measure)
  CellState(I3, points)
end

function Gridap.CellData.update_state!(obj::ViscousIncompressible, A, F, Fn)
  update_state!(return_mapping(obj), A, F, Fn)
end

function dissipation(obj::ViscousPolyconvex)
  D(F, Fn, A) = dissipation(obj, F, Fn, A)
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

function energy(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj, C, Cn, A)
  Ψv(obj, C, Cv)
end

function first_piola(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj, C, Cn, Cvn)
  F * Sv(obj, C, Cv)
end

function tangent(obj::ViscousPolyconvex, F, Fn, Cvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj, C, Cn, Cvn)
  H = Hv(obj, C, Cv)
  DCDF = F' ⊗₁₃²⁴ I3 + I3 ⊗₁₄²³ F'
  DCDF' · H · DCDF
end

function dissipation(obj::ViscousPolyconvex, F, Fn, Cvn)
  γ = obj.μ / obj.Δt[]
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Cv = return_mapping(obj, C, Cn, Cvn)
  invC = inv(C)
  λ = 3 / (Cv ⊙ invC)
  -0.5γ * (C -λ*Cv) ⊙ (invC - inv(Cv)/λ)
end

function return_mapping(obj::ViscousPolyconvex, C, Cn, A)
  τ = obj.τ
  Δt = obj.Δt[]
  Cvn = A
  invC = inv(C)
  B = Δt/τ * invC + inv(Cvn)
  invCv = det(B)^(1/3) * B
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
