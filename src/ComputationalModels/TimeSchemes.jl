#*******************************************************************************
#    					 Time schemes: the stage contract
#*******************************************************************************
#
# A time step is a sequence of stages; a stage is a (non)linear problem in a stage unknown y (free
# values). Every time-dependent quantity a weak form may use is affine in y:
#
#     field(ctx, u)   = α₀ y + β₀      the field at the stage (u itself for the implicit schemes)
#     ∂t(ctx, u)      = α₁ y + β₁      the velocity of the stage (the one of the residual)
#     ∂tt(ctx, u)     = α₂ y + β₂      the acceleration of the residual (algorithmic: (1−αm) aₙ₊₁ + αm aₙ)
#     at(ctx, u, :n)  = uₙ             the history
#
# α are scalars of the stage (runtime parameters of the forms), β are FE functions over the preallocated
# history vectors of the model (`StageHistory`), written in place when a stage is built and when a step is
# committed. The variations of the trial direction are δfield(ctx, du) = α₀ du, δ∂t(ctx, du) = α₁ du,
# δ∂tt(ctx, du) = α₂ du. The role (state or trial direction) is explicit in the name because the compiled
# backend captures both as the same probe type.

abstract type TimeScheme end

"Order of accuracy in Δt."
function order end
"Order of the time derivative the scheme integrates: 0 (quasi-static), 1 or 2."
function derivative_order end
"Number of stage solves per step (1 unless Runge–Kutta)."
nstages(::TimeScheme) = 1
"Whether the stage is implicit in the field (α₀ ≠ 0); an explicit stage solves with the mass alone."
implicit(::TimeScheme, stage::Int) = true
"Whether the scheme has a local error estimate (`AdaptiveStepping`)."
has_error_estimate(::TimeScheme) = false
"""
Order of the local error estimate for the step-size controller: the order of the scheme for the estimates of
its own error (Milne, Zienkiewicz–Xie), the order of the embedded solution for an embedded pair (the estimate
is the error of the lower-order solution, the step is advanced with the higher one).
"""
error_order(s::TimeScheme) = order(s)

"""
    QuasiStatic()

No time derivative: the step is the static problem at `t_{n+1}`; the history (`at(ctx, u, :n)`) is for
the internal variables. Order 0; no error estimate.
"""
struct QuasiStatic <: TimeScheme end
order(::QuasiStatic) = 0
derivative_order(::QuasiStatic) = 0

"""
    BDF(k)

Backward differentiation formula of order `k` = 1 (backward Euler) or 2, with variable step
(`ω = Δtₙ/Δtₙ₋₁`): `∂t u = c₀ uₙ₊₁ − h`, `c₀ = (1+2ω)/((1+ω)Δt)`, `h = ((1+ω) uₙ − ω²/(1+ω) uₙ₋₁)/Δt`
(k = 2, started with k = 1); `c₀ = 1/Δt`, `h = uₙ/Δt` (k = 1). The field is evaluated at `uₙ₊₁`. Error
estimate by Milne's device against the explicit predictor of the same order.
"""
struct BDF <: TimeScheme
    k::Int
    function BDF(k::Int)
        k in (1, 2) || error("HyperFEM: BDF(k) needs k = 1 or 2 (got $k)")
        new(k)
    end
end
order(s::BDF) = s.k
derivative_order(::BDF) = 1
has_error_estimate(::BDF) = true

"""
    Theta(θ)

One-leg θ-method: `∂t u = (uₙ₊₁ − uₙ)/Δt` and the field at `u_θ = θ uₙ₊₁ + (1−θ) uₙ`, time `t_θ`
(`field(ctx, u)`). θ = 1 is backward Euler, θ = ½ the midpoint rule (order 2), Gridap's `ThetaMethod`.
Error estimate by Milne's device.
"""
struct Theta <: TimeScheme
    θ::Float64
    function Theta(θ::Real)
        0 < θ <= 1 || error("HyperFEM: Theta(θ) needs 0 < θ ≤ 1 (got $θ)")
        new(Float64(θ))
    end
end
order(s::Theta) = s.θ == 0.5 ? 2 : 1
derivative_order(::Theta) = 1
has_error_estimate(::Theta) = true

"""
    GeneralizedAlpha(ρ∞)
    GeneralizedAlpha(; αm, αf, β, γ)
    Newmark(; β=1/4, γ=1/2);  HHT(α);  Midpoint()

Chung–Hulbert family for second-order problems. Newmark relations `aₙ₊₁ = (uₙ₊₁ − ũ)/(βΔt²)`,
`ũ = uₙ + Δt vₙ + Δt²(½−β) aₙ`, `vₙ₊₁ = vₙ + Δt((1−γ) aₙ + γ aₙ₊₁)`; the residual takes the
acceleration `(1−αm) aₙ₊₁ + αm aₙ` (`∂tt(ctx, u)`), the velocity `(1−αf) vₙ₊₁ + αf vₙ` (`∂t(ctx, u)`)
and the field `(1−αf) uₙ₊₁ + αf uₙ` (`field(ctx, u)`) at `t_{n+1−αf}`. `ρ∞ ∈ [0, 1]` gives
`αm = (2ρ∞−1)/(ρ∞+1)`, `αf = ρ∞/(ρ∞+1)`, `γ = ½ − αm + αf`, `β = ¼(1 − αm + αf)²`. `Newmark` is
`αm = αf = 0`; `HHT(α)` is `αm = 0, αf = α, γ = ½ + α, β = ¼(1+α)²`; `Midpoint()` is `ρ∞ = 1`
(`αm = αf = ½, β = ¼, γ = ½`: `uₙ₊₁ − uₙ = Δt (vₙ + vₙ₊₁)/2` and the forces at `u_{n+½}`). Order 2.
Error estimate of Zienkiewicz–Xie extended to the family, `C Δt²(aₙ₊₁ − aₙ)` with
`C = β − (1−αm)/(6(1−αf))` (`β − 1/6` for Newmark; the schemes with `C = 0` have no estimate).
"""
struct GeneralizedAlpha <: TimeScheme
    αm::Float64
    αf::Float64
    β::Float64
    γ::Float64
    function GeneralizedAlpha(; αm::Real, αf::Real, β::Real, γ::Real)
        β > 0 || error("HyperFEM: GeneralizedAlpha needs β > 0 (got $β)")
        new(Float64(αm), Float64(αf), Float64(β), Float64(γ))
    end
end
function GeneralizedAlpha(ρ∞::Real)
    0 <= ρ∞ <= 1 || error("HyperFEM: GeneralizedAlpha(ρ∞) needs 0 ≤ ρ∞ ≤ 1 (got $ρ∞)")
    αm = (2ρ∞ - 1) / (ρ∞ + 1)
    αf = ρ∞ / (ρ∞ + 1)
    GeneralizedAlpha(; αm, αf, γ=0.5 - αm + αf, β=0.25 * (1 - αm + αf)^2)
end
Newmark(; β::Real=0.25, γ::Real=0.5) = GeneralizedAlpha(; αm=0.0, αf=0.0, β, γ)
HHT(α::Real) = GeneralizedAlpha(; αm=0.0, αf=α, γ=0.5 + α, β=0.25 * (1 + α)^2)
Midpoint() = GeneralizedAlpha(; αm=0.5, αf=0.5, β=0.25, γ=0.5)
order(::GeneralizedAlpha) = 2
derivative_order(::GeneralizedAlpha) = 2
has_error_estimate(s::GeneralizedAlpha) = _zx_constant(s) != 0.0

"""
    CentralDifference()

Explicit central differences for second-order problems: the stage unknown is the acceleration
`aₙ₊₁`, the field is known, `uₙ₊₁ = uₙ + Δt vₙ + ½Δt² aₙ` (`field(ctx, u)`, α₀ = 0), the velocity
`vₙ₊₁ = vₙ + ½Δt(aₙ + aₙ₊₁)` (`∂t(ctx, u)`, implicit in the acceleration only through a damping term)
and `∂tt(ctx, u) = aₙ₊₁`. The stage solve is the mass system (`mass_lumping = true` for a diagonal
one); the stability limit (CFL) is the user's. Order 2; no error estimate.
"""
struct CentralDifference <: TimeScheme end
order(::CentralDifference) = 2
derivative_order(::CentralDifference) = 2
implicit(::CentralDifference, ::Int) = false

"""
    SDIRK(2)

Singly diagonally implicit Runge–Kutta of order 2 (Alexander, L-stable, `γ = 1 − √2/2`) for
first-order problems, two implicit stages: the stage unknown is the stage derivative `kᵢ`,
`field(ctx, u) = uₙ + Δt Σⱼ aᵢⱼ kⱼ` (α₀ = Δt aᵢᵢ), `∂t(ctx, u) = kᵢ`, time `tₙ + cᵢ Δt`;
`uₙ₊₁ = uₙ + Δt Σ bᵢ kᵢ`. Embedded first-order estimate `Δt Σ (bᵢ − b̂ᵢ) kᵢ` with `b̂ = (1, 0)`.
"""
struct SDIRK <: TimeScheme
    A::Matrix{Float64}
    b::Vector{Float64}
    b̂::Vector{Float64}
    c::Vector{Float64}
    function SDIRK(p::Int)
        p == 2 || error("HyperFEM: SDIRK(p) is implemented for p = 2 only (got $p)")
        γ = 1 - sqrt(2) / 2
        new([γ 0.0; 1-γ γ], [1 - γ, γ], [1.0, 0.0], [γ, 1.0])
    end
end
order(::SDIRK) = 2
error_order(::SDIRK) = 1
derivative_order(::SDIRK) = 1
nstages(s::SDIRK) = length(s.b)
has_error_estimate(::SDIRK) = true

"Whether the scheme works with a stage time distinct from the end of the step (loads at `ctx.t`)."
stage_time(s::TimeScheme, t::Float64, Δt::Float64, stage::Int) = t + Δt
stage_time(s::GeneralizedAlpha, t::Float64, Δt::Float64, ::Int) = t + (1 - s.αf) * Δt
stage_time(s::Theta, t::Float64, Δt::Float64, ::Int) = t + s.θ * Δt
stage_time(s::SDIRK, t::Float64, Δt::Float64, i::Int) = t + s.c[i] * Δt

#*******************************************************************************
#    					 Stage coefficients and the time context
#*******************************************************************************

"""
    StageCoefficients(α₀, α₁, α₂, αf, s_K)

Scalars of a stage: the coefficients of the stage unknown in the field, the velocity and the
acceleration of the residual, the weight `αf` of the history in the field (`field = (1−αf) uₙ₊₁ + αf uₙ`
for the schemes evaluating at an intermediate state; 0 otherwise) and the factor `s_K` of the static
tangent in the stage tangent (`∂field/∂y`: `(1−αf)` for the generalized-α, `θ` for `Theta`, `Δt aᵢᵢ`
for the Runge–Kutta stages, 1 otherwise, 0 for an explicit stage).
"""
struct StageCoefficients
    α0::Float64
    α1::Float64
    α2::Float64
    αf::Float64
    s_K::Float64
end

"""
    TimeContext

What the weak forms of a transient model see at a stage: the stage time `t`, the step `Δt`, the
step count `step` (the one being computed, n + 1), the stage `stage`, the coefficients `c`
([`StageCoefficients`](@ref)) and the history `h` ([`StageHistory`](@ref): the FE functions `uₙ`,
`uₙ₋₁`, `vₙ`, `aₙ` and the β functions). `ctx.u⁻`, `ctx.u⁻⁻`, `ctx.v⁻`, `ctx.a⁻` are the history FE
functions; `ctx.history` the user's history object (`history = …` of the model), `nothing` by default.
The functions of the contract: [`field`](@ref), [`∂t`](@ref), [`∂tt`](@ref), [`at`](@ref) and their
variations `δfield`, `δ∂t`, `δ∂tt`. Indexing `ctx[i]` gives the context of field `i` (the same one when
all the fields share the scheme).
"""
struct TimeContext{H,U}
    t::Float64
    Δt::Float64
    step::Int
    stage::Int
    c::StageCoefficients
    h::H
    history::U
end
Base.getindex(ctx::TimeContext, ::Int) = ctx
function Base.getproperty(ctx::TimeContext, s::Symbol)
    s === :u⁻ && return getfield(ctx, :h).un
    s === :u⁻⁻ && return getfield(ctx, :h).unm1
    s === :v⁻ && return getfield(ctx, :h).vn
    s === :a⁻ && return getfield(ctx, :h).an
    getfield(ctx, s)
end

"""
    field(ctx, u);  δfield(ctx, du)

The field at the stage, affine in the stage unknown `u`: `u` itself for the implicit one-step and
multistep schemes, `(1−αf) u + αf uₙ` for the generalized-α (and `θ u + (1−θ) uₙ` for `Theta`), the
known `uₙ₊₁` for an explicit stage (α₀ = 0), `uₙ + Δt Σ aᵢⱼ kⱼ` for a Runge–Kutta stage. `δfield` is its
variation along the trial direction `du`.
"""
field(ctx::TimeContext, u) = _affine(ctx.c.α0, u, ctx.h.β0, ctx.h.use_β0)
δfield(ctx::TimeContext, du) = _scaled(ctx.c.α0, du)

"""
    ∂t(ctx, u);  δ∂t(ctx, du)

The velocity of the residual at the stage, affine in the stage unknown: `c₀ u − h` (BDF),
`(u − uₙ)/Δt` (Theta), `(1−αf) vₙ₊₁ + αf vₙ` with Newmark's `vₙ₊₁` (generalized-α), `vₙ + ½Δt(aₙ + aₙ₊₁)`
(central differences), `kᵢ` (Runge–Kutta). Not defined for `QuasiStatic`.
"""
∂t(ctx::TimeContext, u) = _affine(ctx.c.α1, u, ctx.h.β1, true)
δ∂t(ctx::TimeContext, du) = _scaled(ctx.c.α1, du)

"""
    ∂tt(ctx, u);  δ∂tt(ctx, du)

The acceleration of the residual at the stage, affine in the stage unknown: `(1−αm) aₙ₊₁ + αm aₙ` with
Newmark's `aₙ₊₁` (generalized-α), `aₙ₊₁` itself (central differences). Second-order schemes only.
"""
∂tt(ctx::TimeContext, u) = _affine(ctx.c.α2, u, ctx.h.β2, true)
δ∂tt(ctx::TimeContext, du) = _scaled(ctx.c.α2, du)

"""
    at(ctx, u, point)

The field at a point of the step: `:n` (`uₙ`, the history), `:np1` (`uₙ₊₁`: `u` for the implicit
schemes, the known field for an explicit stage) and `:stage` (the same as `field(ctx, u)`).
`δat(ctx, du, point)` is the variation (zero at `:n`).
"""
function at(ctx::TimeContext, u, point::Symbol)
    point === :n && return ctx.h.un
    point === :stage && return field(ctx, u)
    point === :np1 && return _np1(ctx, u)
    error("HyperFEM: unknown point :$point of `at(ctx, u, point)`; use :n, :np1 or :stage")
end
function δat(ctx::TimeContext, du, point::Symbol)
    point === :n && return _scaled(0.0, du)
    point === :stage && return δfield(ctx, du)
    point === :np1 && return _scaled(ctx.h.α0_np1, du)
    error("HyperFEM: unknown point :$point of `δat(ctx, du, point)`; use :n, :np1 or :stage")
end
# uₙ₊₁ as a function of the stage unknown: u (implicit), the known field (explicit), uₙ + Δt Σ bᵢ kᵢ is not
# available inside a Runge–Kutta stage (the later stages are unknown): the stage field is used.
_np1(ctx::TimeContext, u) = _affine(ctx.h.α0_np1, u, ctx.h.β0_np1, ctx.h.use_β0_np1)

_scaled(α::Float64, u) = AffineCellField(α, u, nothing)
_affine(α::Float64, u, β, useβ::Bool) = useβ ? AffineCellField(α, u, β) : AffineCellField(α, u, nothing)

"""
    AffineCellField(α, f, h)

The cell field `α f + h` (`h = nothing`: `α f`) whose gradient is `α ∇f + ∇h`, formed from the gradients
of `f` and `h` instead of differentiating the expression: the compiled backend reads differential
operators on the unknown, test and trial functions only, not on expressions containing them. Every other
operation delegates to the expression. The affine quantities of the stage contract are of this type.
"""
struct AffineCellField{F,H,E} <: CellField
    α::Float64
    f::F
    h::H
    expr::E
end
_affine_expr(α::Float64, f, ::Nothing) = α == 1.0 ? f : α * f
_affine_expr(α::Float64, f, h) = α == 0.0 ? h : (α == 1.0 ? f + h : α * f + h)
AffineCellField(α::Float64, f, h) = AffineCellField(α, f, h, _affine_expr(α, f, h))
CellData.get_data(a::AffineCellField) = CellData.get_data(a.expr)
CellData.get_triangulation(a::AffineCellField) = CellData.get_triangulation(a.expr)
CellData.DomainStyle(a::AffineCellField) = CellData.DomainStyle(a.expr)
CellData.change_domain(a::AffineCellField, t::Triangulation, d::CellData.DomainStyle) = CellData.change_domain(a.expr, t, d)
CellData.change_domain(a::AffineCellField, s::CellData.DomainStyle, t::Triangulation, d::CellData.DomainStyle) = CellData.change_domain(a.expr, s, t, d)
CellData.change_domain(a::AffineCellField, ::CellData.ReferenceDomain, ::CellData.PhysicalDomain) = CellData.change_domain(a.expr, CellData.ReferenceDomain(), CellData.PhysicalDomain())
CellData.change_domain(a::AffineCellField, ::CellData.PhysicalDomain, ::CellData.ReferenceDomain) = CellData.change_domain(a.expr, CellData.PhysicalDomain(), CellData.ReferenceDomain())
Fields.gradient(a::AffineCellField) = _affine_expr(a.α, Fields.gradient(a.f), a.h === nothing ? nothing : Fields.gradient(a.h))
Base.show(io::IO, a::AffineCellField) = print(io, "AffineCellField(", a.α, ", ", a.h === nothing ? "" : "+h", ")")

#*******************************************************************************
#    					 History of the stages
#*******************************************************************************

"""
    StageHistory

Preallocated vectors of the time loop over the free dofs (`xn`, `xnm1`, `xnm2`, `vn`, `an`, `ẋn`, `ẋnm1`, the
stage derivatives `ks`, the predictor `xp`, the error estimate `e`, the scratch `w`) and their Dirichlet
values per field, the FE functions over them (`un`, `unm1`, `vn`, `an`), and the β functions of the
contract (`β0`, `β1`, `β2` over `b0`, `b1`, `b2`, with the flags saying whether they are in use) plus
the ones of `at(ctx, u, :np1)`. Written in place at every stage and at every commit; never rebuilt.
"""
mutable struct StageHistory{V<:AbstractVector,D,F}
    const xn::V
    const xnm1::V
    const xnm2::V
    const vn::V
    const an::V
    const ẋn::V
    const ẋnm1::V
    const ks::Vector{V}
    const xp::V
    const e::V
    const w::V
    const b0::V
    const b1::V
    const b2::V
    const b0np1::V
    const gd::D          # Dirichlet values of the trial space at the target time (g_{n+1}), per field
    const gdn::D
    const gdnm1::D
    const vdn::D
    const adn::D
    const ẋdn::D
    const ẋdnm1::D
    const b0d::D
    const b1d::D
    const b2d::D
    const b0np1d::D
    const un::F
    const unm1::F
    const vnh::F
    const anh::F
    const β0::F
    const β1::F
    const β2::F
    const β0_np1::F
    use_β0::Bool
    use_β0_np1::Bool
    α0_np1::Float64
    nstored::Int          # accepted steps whose state is in the history (BDF2 startup, Theta(½) predictor)
    Δt⁻::Float64          # increment of the last accepted step (0 before the first), set by the driver
    Δt⁻⁻::Float64         # increment of the step before (shifted at commit)
    c0::Float64           # velocity coefficient of the first-order schemes (their commit reads it)
end

function StageHistory(U, x::AbstractVector, nks::Int)
    z() = (v = similar(x); fill!(v, zero(eltype(x))); v)
    zd() = [zeros(num_dirichlet_dofs(s)) for s in _single_spaces(U)]
    ks = [z() for _ in 1:nks]
    xn, xnm1, xnm2, vn, an, ẋn, ẋnm1, xp, e, w, b0, b1, b2, b0np1 = z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z()
    gd, gdn, gdnm1, vdn, adn, ẋdn, ẋdnm1, b0d, b1d, b2d, b0np1d = zd(), zd(), zd(), zd(), zd(), zd(), zd(), zd(), zd(), zd(), zd()
    fe(v, d) = _fe_function(U, v, d)
    un, unm1, vnh, anh = fe(xn, gdn), fe(xnm1, gdnm1), fe(vn, vdn), fe(an, adn)
    β0, β1, β2, β0_np1 = fe(b0, b0d), fe(b1, b1d), fe(b2, b2d), fe(b0np1, b0np1d)
    StageHistory{typeof(x),typeof(gd),typeof(un)}(xn, xnm1, xnm2, vn, an, ẋn, ẋnm1, ks, xp, e, w, b0, b1, b2, b0np1,
        gd, gdn, gdnm1, vdn, adn, ẋdn, ẋdnm1, b0d, b1d, b2d, b0np1d,
        un, unm1, vnh, anh, β0, β1, β2, β0_np1, false, false, 1.0, 0, 0.0, 0.0, 0.0)
end

_single_spaces(U::MultiFieldFESpace) = [U[i] for i in 1:num_fields(U)]
_single_spaces(U) = [U]
_fe_function(U::MultiFieldFESpace, v, d::Vector) = FEFunction(U, v, d)
_fe_function(U, v, d::Vector) = FEFunction(U, v, d[1])

"Dirichlet values of the trial space, per field (the vectors themselves, written in place by `TrialFESpace!`)."
_dirichlet_values(U) = [get_dirichlet_dof_values(s) for s in _single_spaces(U)]

# in-place linear combinations of the per-field Dirichlet vectors
function _dcopy!(a::Vector, b::Vector)
    for i in eachindex(a); a[i] .= b[i]; end
end
function _daxpby!(α::Float64, a::Vector, β::Float64, b::Vector)   # b ← α a + β b
    for i in eachindex(a); b[i] .= α .* a[i] .+ β .* b[i]; end
end
function _dcombine!(out::Vector, c1::Float64, a::Vector, c2::Float64, b::Vector)   # out ← c1 a + c2 b
    for i in eachindex(out); out[i] .= c1 .* a[i] .+ c2 .* b[i]; end
end
function _dcombine!(out::Vector, c1::Float64, a::Vector, c2::Float64, b::Vector, c3::Float64, c::Vector)
    for i in eachindex(out); out[i] .= c1 .* a[i] .+ c2 .* b[i] .+ c3 .* c[i]; end
end

#*******************************************************************************
#    					 Stages of each scheme
#*******************************************************************************
# `stage_coefficients!(s, h, stage, Δt, Δt⁻) -> StageCoefficients` writes the β vectors of the stage
# (free values and Dirichlet values, the latter from `h.gd` = g_{n+1} and the history) and returns the
# scalars. `predict!(y, s, h, stage, Δt)` writes the initial guess of the stage unknown. `commit_stage!`
# stores what the later stages and the commit need (the stage derivatives, the explicit acceleration).
# `commit!(s, h, x, Δt)` updates the history from the accepted field `x` = uₙ₊₁ (the stage unknown itself
# for the one-stage implicit schemes): velocities, accelerations, the shift uₙ₋₁ ← uₙ ← uₙ₊₁ (free and
# Dirichlet values).
# `error_estimate!(s, h, x, Δt, Δt⁻)` writes the local error estimate into `h.e`.

# --- QuasiStatic -------------------------------------------------------------------------------
function stage_coefficients!(::QuasiStatic, h::StageHistory, ::Int, Δt::Float64, Δt⁻::Float64)
    h.use_β0 = false; h.use_β0_np1 = false; h.α0_np1 = 1.0
    StageCoefficients(1.0, 0.0, 0.0, 0.0, 1.0)
end
predict!(y, ::QuasiStatic, h::StageHistory, ::Int, Δt::Float64) = (y .= h.xn; nothing)
commit_stage!(::TimeScheme, h::StageHistory, ::Int, y, Δt::Float64) = nothing
function commit!(::QuasiStatic, h::StageHistory, y, Δt::Float64)
    _shift_x!(h, y)
end
# uₙ₋₁ ← uₙ, uₙ ← y (free values), the Dirichlet values alike (g_{n+1} is in h.gd)
function _shift_x!(h::StageHistory, y)
    h.xnm2 .= h.xnm1
    h.xnm1 .= h.xn
    h.xn .= y
    h.Δt⁻⁻ = h.Δt⁻
    _dcopy!(h.gdnm1, h.gdn)
    _dcopy!(h.gdn, h.gd)
    h.nstored += 1
    nothing
end

# --- BDF ---------------------------------------------------------------------------------------
function stage_coefficients!(s::BDF, h::StageHistory, ::Int, Δt::Float64, Δt⁻::Float64)
    h.use_β0 = false; h.use_β0_np1 = false; h.α0_np1 = 1.0
    if s.k == 1 || h.nstored < 1
        c0 = 1 / Δt
        h.b1 .= (-c0) .* h.xn
        _dcombine!(h.b1d, -c0, h.gdn, 0.0, h.gdn)
    else
        ω = Δt / Δt⁻
        c0 = (1 + 2ω) / ((1 + ω) * Δt)
        c1 = (1 + ω) / Δt
        c2 = ω^2 / ((1 + ω) * Δt)
        h.b1 .= (-c1) .* h.xn .+ c2 .* h.xnm1
        _dcombine!(h.b1d, -c1, h.gdn, c2, h.gdnm1)
    end
    h.c0 = c0
    StageCoefficients(1.0, c0, 0.0, 0.0, 1.0)
end
# predictor: extrapolation of the past solution values (the Milne device needs a predictor independent of
# the scheme's own velocities): linear through xₙ₋₁, xₙ (order 1), quadratic through xₙ₋₂, xₙ₋₁, xₙ
# (order 2, once two steps are stored); xₙ before the first step
function predict!(y, s::BDF, h::StageHistory, ::Int, Δt::Float64)
    _extrapolate!(y, h, Δt, s.k == 2 && h.nstored >= 2 ? 2 : 1)
end
function _extrapolate!(y, h::StageHistory, Δt::Float64, order::Int)
    if h.nstored < 1
        y .= h.xn
    elseif order == 1 || h.nstored < 2
        ω = Δt / h.Δt⁻
        y .= h.xn .+ ω .* (h.xn .- h.xnm1)
    else
        # Lagrange through (−Δt⁻−Δt⁻⁻, xₙ₋₂), (−Δt⁻, xₙ₋₁), (0, xₙ) at Δt
        r1, r2 = h.Δt⁻, h.Δt⁻⁻
        t1, t2 = -r1, -r1 - r2
        L0 = (Δt - t1) * (Δt - t2) / (t1 * t2)
        L1 = Δt * (Δt - t2) / (t1 * (t1 - t2))
        L2 = Δt * (Δt - t1) / (t2 * (t2 - t1))
        y .= L0 .* h.xn .+ L1 .* h.xnm1 .+ L2 .* h.xnm2
    end
    h.xp .= y
    nothing
end
function commit!(s::BDF, h::StageHistory, y, Δt::Float64)
    # ẋₙ₊₁ = c₀ y + β₁ (the velocity of the scheme), Dirichlet alike
    c = h.c0
    h.ẋnm1 .= h.ẋn
    h.ẋn .= c .* y .+ h.b1
    _dcopy!(h.ẋdnm1, h.ẋdn)
    _dcombine!(h.ẋdn, c, h.gd, 1.0, h.b1d)
    _shift_x!(h, y)
end

# --- Theta (one-leg) ---------------------------------------------------------------------------
function stage_coefficients!(s::Theta, h::StageHistory, ::Int, Δt::Float64, Δt⁻::Float64)
    θ = s.θ
    h.use_β0 = θ != 1.0; h.use_β0_np1 = false; h.α0_np1 = 1.0
    if h.use_β0
        h.b0 .= (1 - θ) .* h.xn
        _dcombine!(h.b0d, 1 - θ, h.gdn, 0.0, h.gdn)
    end
    c0 = 1 / Δt
    h.b1 .= (-c0) .* h.xn
    _dcombine!(h.b1d, -c0, h.gdn, 0.0, h.gdn)
    h.c0 = c0
    StageCoefficients(θ, c0, 0.0, 1 - θ, θ)
end
function predict!(y, s::Theta, h::StageHistory, ::Int, Δt::Float64)
    _extrapolate!(y, h, Δt, s.θ == 0.5 && h.nstored >= 2 ? 2 : 1)
end
function commit!(::Theta, h::StageHistory, y, Δt::Float64)
    c = h.c0
    h.ẋnm1 .= h.ẋn
    h.ẋn .= c .* y .+ h.b1
    _dcopy!(h.ẋdnm1, h.ẋdn)
    _dcombine!(h.ẋdn, c, h.gd, 1.0, h.b1d)
    _shift_x!(h, y)
end

# --- GeneralizedAlpha (Newmark, HHT, Midpoint) -------------------------------------------------
function stage_coefficients!(s::GeneralizedAlpha, h::StageHistory, ::Int, Δt::Float64, Δt⁻::Float64)
    αm, αf, β, γ = s.αm, s.αf, s.β, s.γ
    # ũ = uₙ + Δt vₙ + Δt²(½−β) aₙ  in w;  aₙ₊₁ = (u − ũ)/(βΔt²);  vₙ₊₁ = vₙ + Δt(1−γ) aₙ + γ Δt aₙ₊₁
    c2 = 1 / (β * Δt^2)
    h.w .= h.xn .+ Δt .* h.vn .+ (Δt^2 * (0.5 - β)) .* h.an
    # field: (1−αf) u + αf uₙ
    h.use_β0 = αf != 0.0; h.use_β0_np1 = false; h.α0_np1 = 1.0
    if h.use_β0
        h.b0 .= αf .* h.xn
        _dcombine!(h.b0d, αf, h.gdn, 0.0, h.gdn)
    end
    # acceleration of the residual: (1−αm) aₙ₊₁ + αm aₙ = (1−αm) c2 u − (1−αm) c2 ũ + αm aₙ
    α2 = (1 - αm) * c2
    h.b2 .= (-α2) .* h.w .+ αm .* h.an
    # velocity of the residual: (1−αf) vₙ₊₁ + αf vₙ = (1−αf) γ c2 Δt u + (1−αf)(vₙ + Δt(1−γ) aₙ − γ c2 Δt ũ) + αf vₙ
    α1 = (1 - αf) * γ * c2 * Δt
    h.b1 .= (1 - αf) .* (h.vn .+ (Δt * (1 - γ)) .* h.an .- (γ * c2 * Δt) .* h.w) .+ αf .* h.vn
    # Dirichlet values: ũ_d from gdn, vdn, adn; u_d = g_{n+1} (h.gd)
    for i in eachindex(h.gd)
        ũd = h.gdn[i] .+ Δt .* h.vdn[i] .+ (Δt^2 * (0.5 - β)) .* h.adn[i]
        h.b2d[i] .= (-α2) .* ũd .+ αm .* h.adn[i]
        h.b1d[i] .= (1 - αf) .* (h.vdn[i] .+ (Δt * (1 - γ)) .* h.adn[i] .- (γ * c2 * Δt) .* ũd) .+ αf .* h.vdn[i]
    end
    StageCoefficients(1 - αf, α1, α2, αf, 1 - αf)
end
# predictor: the Newmark predictor ũ (zero-acceleration guess), already in w
function predict!(y, ::GeneralizedAlpha, h::StageHistory, ::Int, Δt::Float64)
    y .= h.w
    h.xp .= y
    nothing
end
function commit!(s::GeneralizedAlpha, h::StageHistory, y, Δt::Float64)
    β, γ = s.β, s.γ
    c2 = 1 / (β * Δt^2)
    # aₙ₊₁ = c2 (y − ũ) with ũ still in w; vₙ₊₁ = vₙ + Δt(1−γ) aₙ + γ Δt aₙ₊₁; the error estimate reads aₙ
    h.xp .= h.an                                       # aₙ (the predictor is consumed; the estimate in h.e survives)
    h.an .= c2 .* (y .- h.w)
    h.vn .= h.vn .+ (Δt * (1 - γ)) .* h.xp .+ (γ * Δt) .* h.an
    for i in eachindex(h.gd)
        ũd = h.gdn[i] .+ Δt .* h.vdn[i] .+ (Δt^2 * (0.5 - β)) .* h.adn[i]
        ad = c2 .* (h.gd[i] .- ũd)
        h.vdn[i] .= h.vdn[i] .+ (Δt * (1 - γ)) .* h.adn[i] .+ (γ * Δt) .* ad
        h.adn[i] .= ad
    end
    _shift_x!(h, y)
end

# --- CentralDifference (explicit) --------------------------------------------------------------
function stage_coefficients!(::CentralDifference, h::StageHistory, ::Int, Δt::Float64, Δt⁻::Float64)
    # uₙ₊₁ = uₙ + Δt vₙ + ½Δt² aₙ (known, in b0 and b0np1); vₙ₊₁ = vₙ + ½Δt aₙ + ½Δt aₙ₊₁; aₙ₊₁ = y
    h.b0 .= h.xn .+ Δt .* h.vn .+ (0.5 * Δt^2) .* h.an
    h.b0np1 .= h.b0
    h.use_β0 = true; h.use_β0_np1 = true; h.α0_np1 = 0.0
    h.b1 .= h.vn .+ (0.5 * Δt) .* h.an
    h.b2 .= 0.0
    for i in eachindex(h.gd)
        h.b0d[i] .= h.gd[i]                            # the Dirichlet values of uₙ₊₁ are the prescribed ones
        h.b0np1d[i] .= h.gd[i]
        ad = (2 / Δt^2) .* (h.gd[i] .- h.gdn[i] .- Δt .* h.vdn[i]) .- h.adn[i]   # consistent with uₙ₊₁ = uₙ + Δt vₙ + ½Δt² aₙ
        h.b1d[i] .= h.vdn[i] .+ (0.5 * Δt) .* h.adn[i] .+ (0.5 * Δt) .* ad
        h.b2d[i] .= ad
    end
    StageCoefficients(0.0, 0.5 * Δt, 1.0, 0.0, 0.0)
end
predict!(y, ::CentralDifference, h::StageHistory, ::Int, Δt::Float64) = (y .= h.an; nothing)
commit_stage!(::CentralDifference, h::StageHistory, ::Int, y, Δt::Float64) = (h.ks[1] .= y; nothing)   # aₙ₊₁
function commit!(::CentralDifference, h::StageHistory, x, Δt::Float64)
    # `x` is the accepted field uₙ₊₁ (= b0); the acceleration of the stage is in ks[1]
    h.vn .= h.b1 .+ (0.5 * Δt) .* h.ks[1]
    h.an .= h.ks[1]
    for i in eachindex(h.gd)
        h.vdn[i] .= h.b1d[i]
        h.adn[i] .= h.b2d[i]
    end
    _shift_x!(h, h.b0)
end

# --- SDIRK -------------------------------------------------------------------------------------
function stage_coefficients!(s::SDIRK, h::StageHistory, i::Int, Δt::Float64, Δt⁻::Float64)
    # field: uₙ + Δt Σⱼ₍ⱼ<ᵢ₎ aᵢⱼ kⱼ + Δt aᵢᵢ y;  ∂t = y
    h.b0 .= h.xn
    for j in 1:i-1
        h.b0 .+= (Δt * s.A[i, j]) .* h.ks[j]
    end
    h.use_β0 = true; h.use_β0_np1 = false; h.α0_np1 = 1.0
    # Dirichlet values of the stage field: the prescribed values at the stage time, taken as the
    # interpolation g(tₙ + cᵢΔt) ≈ (1 − cᵢ) gₙ + cᵢ g_{n+1} (the loads at the stage time are the user's)
    _dcombine!(h.b0d, 1 - s.c[i], h.gdn, s.c[i], h.gd)
    h.b1 .= 0.0
    for d in h.b1d; d .= 0.0; end
    StageCoefficients(Δt * s.A[i, i], 1.0, 0.0, 0.0, Δt * s.A[i, i])
end
function predict!(y, s::SDIRK, h::StageHistory, i::Int, Δt::Float64)
    y .= i == 1 ? h.ẋn : h.ks[i-1]
    nothing
end
function commit_stage!(::SDIRK, h::StageHistory, i::Int, y, Δt::Float64)
    h.ks[i] .= y
    nothing
end
function commit!(s::SDIRK, h::StageHistory, y, Δt::Float64)
    # uₙ₊₁ = uₙ + Δt Σ bᵢ kᵢ into w; ẋₙ₊₁ = k of the last stage (c = 1, stiffly accurate)
    h.w .= h.xn
    for i in eachindex(s.b)
        h.w .+= (Δt * s.b[i]) .* h.ks[i]
    end
    h.ẋnm1 .= h.ẋn
    h.ẋn .= h.ks[end]
    _dcopy!(h.ẋdnm1, h.ẋdn)
    for i in eachindex(h.gd)
        h.ẋdn[i] .= (h.gd[i] .- h.gdn[i]) ./ Δt
    end
    _shift_x!(h, h.w)
end

#*******************************************************************************
#    					 Local error estimates (AdaptiveStepping)
#*******************************************************************************
# `error_estimate!(s, h, y, Δt, Δt⁻)` writes the estimate of the local error of the step into `h.e`
# after the stage solves and before the commit (the history is still the one of step n); `y` is the
# accepted stage unknown (uₙ₊₁ for the one-stage implicit schemes).

# Milne's device: e = C (x − x^p) with C = c_c/(c_c − c_p), the leading local-error coefficients of the
# scheme (c_c) and of the predictor (c_p) in Δt^(p+1) x^(p+1); variable-step values for BDF2 and AB2.
# Milne's device with the extrapolation predictors of `_extrapolate!`: e = C (x − xᵖ), C = c_c/(c_c − c_p) with
# the leading local-error coefficients of the scheme (c_c) and of the predictor (c_p), derived on ẋ = λx with
# variable steps ω = Δt/Δt⁻, ω′ = Δt⁻/Δt⁻⁻ (constant-step values in brackets):
#   linear predictor:    c_p = −(1+ω)/(2ω)                      (in Δt² ẍ)
#   quadratic predictor: c_p = −(1+ω)(ωω′+ω′+1)/(6ω²ω′)          (in Δt³ x‴)
#   BDF1 (c_c = ½):               C = ω/(2ω+1)                   [1/3]
#   Theta(θ) (c_c = θ−½):         C = ω(2θ−1)/(2ωθ+1)            [(2θ−1)/(2θ+1)]
#   BDF2 (c_c = (1+ω)²/(6ω(1+2ω))): C = ωω′(1+ω)/(3ω²ω′+4ωω′+2ω+ω′+1)   [2/11]
#   Theta(½) (c_c = 1/12):        C = ω²ω′/(3ω²ω′+4ωω′+2ω+2ω′+2)         [1/13]
# Before the first accepted step the predictor is xₙ and C = ½ (a conservative first-order estimate); in the
# start-up steps of the order-2 schemes the linear predictor is used with C = ½ as well.
function error_estimate!(s::BDF, h::StageHistory, y, Δt::Float64, Δt⁻::Float64)
    ω = h.nstored >= 1 ? Δt / Δt⁻ : 1.0
    if s.k == 2 && h.nstored >= 2
        ω′ = Δt⁻ / h.Δt⁻⁻
        C = ω * ω′ * (1 + ω) / (3ω^2 * ω′ + 4ω * ω′ + 2ω + ω′ + 1)
    elseif h.nstored >= 1 && s.k == 1
        C = ω / (2ω + 1)
    else
        C = 0.5
    end
    h.e .= C .* (y .- h.xp)
    nothing
end
function error_estimate!(s::Theta, h::StageHistory, y, Δt::Float64, Δt⁻::Float64)
    θ = s.θ
    ω = h.nstored >= 1 ? Δt / Δt⁻ : 1.0
    if θ == 0.5 && h.nstored >= 2
        ω′ = Δt⁻ / h.Δt⁻⁻
        C = ω^2 * ω′ / (3ω^2 * ω′ + 4ω * ω′ + 2ω + 2ω′ + 2)
    elseif θ != 0.5 && h.nstored >= 1
        C = ω * (2θ - 1) / (2ω * θ + 1)
    else
        C = 0.5
    end
    h.e .= C .* (y .- h.xp)
    nothing
end
function error_estimate!(s::GeneralizedAlpha, h::StageHistory, y, Δt::Float64, Δt⁻::Float64)
    # Zienkiewicz–Xie for the generalized-α family: C Δt²(aₙ₊₁ − aₙ), C = β − (1−αm)/(6(1−αf)) (β − 1/6 for
    # Newmark), from the leading local displacement error on ẍ = −ω²x (TimeSchemesTests); aₙ₊₁ = (y − ũ)/(βΔt²), ũ in w
    c2 = 1 / (s.β * Δt^2)
    h.e .= (Δt^2 * _zx_constant(s)) .* (c2 .* (y .- h.w) .- h.an)
    nothing
end
_zx_constant(s::GeneralizedAlpha) = s.β - (1 - s.αm) / (6 * (1 - s.αf))
function error_estimate!(s::SDIRK, h::StageHistory, y, Δt::Float64, Δt⁻::Float64)
    h.e .= 0.0
    for i in eachindex(s.b)
        h.e .+= (Δt * (s.b[i] - s.b̂[i])) .* h.ks[i]
    end
    nothing
end
error_estimate!(s::TimeScheme, h::StageHistory, y, Δt::Float64, Δt⁻::Float64) =
    error("HyperFEM: $(typeof(s)) has no local error estimate; use TimeStepping instead of AdaptiveStepping")
