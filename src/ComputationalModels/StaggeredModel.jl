#*******************************************************************************
#    					 StaggeredModel: partitioned solve of coupled problems
#*******************************************************************************

#---------------------------------- Relaxation ---------------------------------

"""
    Relaxation

Relaxation of the free values of one field of a [`StaggeredModel`](@ref) between the passes of a step.
After the solve of the field in pass `k` the driver sets `x_k = x_{k-1} + ω r_k`, with
`r_k = x̃_k - x_{k-1}` (`x̃_k` the solved value, `x_{k-1}` the value of the previous pass; `x_0` the
state of the last accepted step). `NoRelaxation` is the plain staggered iteration (ω = 1, the state is
not touched); `FixedRelaxation(ω)` a constant factor (written as a `Real` in [`StaggeredStepping`](@ref));
[`Aitken`](@ref) the dynamic one; [`IQNILS`](@ref) a quasi-Newton update built from the passes of the step
(not a scalar factor). Only the free values are relaxed. When the model of the field has an
`injectivity` measure and its Newton uses `InjectivityLS`, ω is cut by the injectivity step of that line
search along `ω r_k`, so that the relaxed state does not invert elements.
"""
abstract type Relaxation end

struct NoRelaxation <: Relaxation end

struct FixedRelaxation <: Relaxation
    ω::Float64
    function FixedRelaxation(ω::Real)
        ω > 0 || error("HyperFEM: FixedRelaxation needs ω > 0 (got $ω)")
        new(Float64(ω))
    end
end

"""
    Aitken(; ω0=1.0, ωmin=0.01, ωmax=1.0)

Aitken Δ² relaxation of a field of a [`StaggeredModel`](@ref): the first pass of every step uses `ω0`;
the pass `k ≥ 2` uses `ω_k = -ω_{k-1} (r_{k-1} ⋅ (r_k - r_{k-1})) / ‖r_k - r_{k-1}‖²`, clamped to
`[ωmin, ωmax]` (when the formula is undefined, `r_k = r_{k-1}`, the previous ω is kept); `ωmax > 1`
allows over-relaxation, which is what a monotonically converging staggered iteration asks for. See
`Relaxation`.
"""
struct Aitken <: Relaxation
    ω0::Float64
    ωmin::Float64
    ωmax::Float64
    function Aitken(; ω0::Real=1.0, ωmin::Real=0.01, ωmax::Real=1.0)
        0 < ωmin <= ωmax || error("HyperFEM: Aitken needs 0 < ωmin ≤ ωmax (got ωmin = $ωmin, ωmax = $ωmax)")
        ωmin <= ω0 <= ωmax || error("HyperFEM: Aitken needs ωmin ≤ ω0 ≤ ωmax (got ω0 = $ω0)")
        new(Float64(ω0), Float64(ωmin), Float64(ωmax))
    end
end

"""
    IQNILS(; reuse=1, ω0=1.0, εfilter=1e-10, maxcols=typemax(Int))

Interface quasi-Newton with inverse Jacobian from a least-squares model (IQN-ILS, Degroote, Bathe and
Vierendeels 2009) for a field of a [`StaggeredModel`](@ref). The passes of a step are a fixed-point
iteration `x̃_k = G(x_k)` with residual `r_k = x̃_k − x_k`; from the differences between passes,
`V = [r_k − r_{k−1}, …]` and `W = [x̃_k − x̃_{k−1}, …]`, the next input is the quasi-Newton update
`x_{k+1} = x̃_k + W α` with `α = argmin ‖V α + r_k‖` (a low-rank secant model of the Jacobian of the
coupling). The columns of the last `reuse` accepted steps are kept and used together with those of the
current step, so that the first pass of a step is already corrected with the model of the previous ones
(the modes that make the coupling hard persist from step to step; without reuse the first passes of a step
follow the plain, possibly diverging, iteration until the model has enough columns; columns of steps too far
back describe a stale Jacobian and make the first updates overshoot, hence the default of one step). A first pass with no
columns at all is relaxed by `ω0`. Unlike a scalar relaxation ([`Aitken`](@ref)), it converges when the map
has modes of both signs, e.g. a slow mode λ ≈ 0.9 together with a mode λ < −1 (a coupling stiffer than the
field's own stiffness), where no single factor ω works. `εfilter` drops the near-dependent columns of `V`
(pivoted QR, relative to the first pivot; essential with reuse); `maxcols` is the maximum number of columns
used. The log reports ω = 1 for this relaxation. See `Relaxation`.
"""
struct IQNILS <: Relaxation
    reuse::Int
    ω0::Float64
    εfilter::Float64
    maxcols::Int
    function IQNILS(; reuse::Int=1, ω0::Real=1.0, εfilter::Real=1e-10, maxcols::Int=typemax(Int))
        reuse >= 0 || error("HyperFEM: IQNILS needs reuse ≥ 0 (got $reuse)")
        0 < ω0 <= 1 || error("HyperFEM: IQNILS needs 0 < ω0 ≤ 1 (got $ω0)")
        εfilter >= 0 || error("HyperFEM: IQNILS needs εfilter ≥ 0 (got $εfilter)")
        maxcols >= 1 || error("HyperFEM: IQNILS needs maxcols ≥ 1 (got $maxcols)")
        new(reuse, Float64(ω0), Float64(εfilter), maxcols)
    end
end

# History for IQN-ILS: the residuals r_k and the solver outputs x̃_k of the passes of the current step, and the
# difference columns (V, W) of the last accepted steps, most recent step first
struct IQNState{X}
    r::Vector{X}
    x̃::Vector{X}
    oldV::Vector{Vector{X}}
    oldW::Vector{Vector{X}}
end

"Work data of a relaxation for the free values `x` of a field (`nothing` for the scalar relaxations)."
relaxation_state(::Relaxation, x) = nothing
relaxation_state(::IQNILS, x) = IQNState(typeof(x)[], typeof(x)[], Vector{typeof(x)}[], Vector{typeof(x)}[])

# Columns of the current step (differences of consecutive passes, most recent first)
_step_columns(v::Vector) = [v[j+1] - v[j] for j in length(v)-1:-1:1]

"The step was accepted: its columns join the pool of the last `reuse` steps."
accept_step!(::Relaxation, ::Nothing) = nothing
function accept_step!(rl::IQNILS, st::IQNState)
    rl.reuse == 0 && return nothing
    length(st.r) >= 2 || return nothing
    pushfirst!(st.oldV, _step_columns(st.r)); pushfirst!(st.oldW, _step_columns(st.x̃))
    while length(st.oldV) > rl.reuse
        pop!(st.oldV); pop!(st.oldW)
    end
    nothing
end

as_relaxation(r::Relaxation) = r
as_relaxation(ω::Real) = FixedRelaxation(ω)
function as_relaxation(s::Symbol)
    s === :none && return NoRelaxation()
    s === :aitken && return Aitken()
    s === :iqnils && return IQNILS()
    error("HyperFEM: unknown relaxation :$s; use :none, :aitken, :iqnils, a factor ω, an `Aitken(…)` or an `IQNILS(…)`")
end
as_relaxation(r) = error("HyperFEM: `relaxation` must be :none, :aitken, :iqnils, a factor ω, an `Aitken(…)`, an `IQNILS(…)` or a tuple with one of them per field; got $(typeof(r))")

# One relaxation per field (a single value applies to every field)
_relaxations(r::Tuple, N) = (length(r) == N || error("HyperFEM: `relaxation` has $(length(r)) entries for $N fields"); map(as_relaxation, r))
_relaxations(r, N) = ntuple(_ -> as_relaxation(r), N)

# Relaxation factor of pass `k` for the increment `r` of the pass, the increment `r⁻` of the previous
# pass and the factor `ω` of the previous pass (the one of the field at the start of the step for k = 1)
relaxation_factor(::NoRelaxation, k::Int, ω::Float64, r, r⁻) = 1.0
relaxation_factor(f::FixedRelaxation, k::Int, ω::Float64, r, r⁻) = f.ω
function relaxation_factor(a::Aitken, k::Int, ω::Float64, r, r⁻)
    k == 1 && return a.ω0
    # Δ = r - r⁻ through inner products (no work vector): r⁻⋅Δ and ‖Δ‖²
    rr⁻ = dot(r⁻, r)
    r⁻r⁻ = dot(r⁻, r⁻)
    den = dot(r, r) - 2 * rr⁻ + r⁻r⁻
    den > 0 || return ω
    ωk = -ω * (rr⁻ - r⁻r⁻) / den
    isfinite(ωk) || return ω
    return clamp(ωk, a.ωmin, a.ωmax)
end

"""
    relaxation_cut(m::ComputationalModel, x⁻, Δx) -> α

Fraction `α ∈ (0, 1]` of the relaxed increment `Δx = ω r` that can be taken from the free values `x⁻` of
the field of `m`: the injectivity step of the `InjectivityLS` of a [`StaticNonlinearModel`](@ref) with an
`injectivity` measure (`Δx` read as a function of the test space, with zero Dirichlet values); `1.0` for
any other model or line search. The relaxation uses `α ω`.
"""
relaxation_cut(::ComputationalModel, x⁻, Δx) = 1.0
function relaxation_cut(m::StaticNonlinearModel, x⁻, Δx)
    ls = _linesearch(m.caches.nls)
    (ls isa InjectivityLS && m.hooks.injectivity !== nothing) || return 1.0
    U, V = m.spaces[1], m.spaces[2]
    return injectivity_step(m, FEFunction(U, x⁻), FEFunction(V, Δx), ls.β, ls.jmin)
end
_linesearch(nls::NewtonRaphson) = nls.linesearch
_linesearch(::NonlinearSolver) = nothing

# Relaxed state of a field after its solve in pass `k`: `r = x⁺ - x⁻` is the increment of the pass;
# the relaxed value is written in `x⁺` (not for `NoRelaxation`, which leaves the solved state untouched)
# and `r⁻` keeps the increment for the Aitken formula of the next pass (it is the work vector of the
# injectivity cut in between). Returns the factor used.
function relax!(rl::Relaxation, ::Nothing, m::ComputationalModel, k::Int, ω::Float64, x⁺, x⁻, r, r⁻)
    @. r = x⁺ - x⁻
    rl isa NoRelaxation && return 1.0
    ωk = relaxation_factor(rl, k, ω, r, r⁻)
    @. r⁻ = ωk * r
    ωk *= relaxation_cut(m, x⁻, r⁻)
    @. x⁺ = x⁻ + ωk * r
    copyto!(r⁻, r)
    return ωk
end

# IQN-ILS: `x⁻` is the input x_k of the pass, `x⁺` the output x̃_k; the update x_{k+1} = x̃_k + W α is written
# in `x⁺` (cut by the injectivity step along x_{k+1} − x_k when the model has one). `r` keeps r_k = x̃_k − x_k
# (the coupling error of the pass); `r⁻` is the work vector of the increment.
function relax!(rl::IQNILS, st::IQNState, m::ComputationalModel, k::Int, ω::Float64, x⁺, x⁻, r, r⁻)
    @. r = x⁺ - x⁻
    if k == 1
        empty!(st.r); empty!(st.x̃)
    end
    push!(st.r, copy(r)); push!(st.x̃, copy(x⁺))
    cols = (_step_columns(st.r), _step_columns(st.x̃))
    for (V, W) in zip(st.oldV, st.oldW)
        append!(cols[1], V); append!(cols[2], W)
    end
    ncols = min(length(cols[1]), rl.maxcols)
    if ncols == 0
        @. r⁻ = rl.ω0 * r
    else
        V = reduce(hcat, cols[1][1:ncols]); W = reduce(hcat, cols[2][1:ncols])
        α = _lsq_filtered(V, -r, rl.εfilter)
        @. r⁻ = r                                                           # x̃_k − x_k …
        mul!(r⁻, W, α, 1.0, 1.0)                                            # … + W α  = x_{k+1} − x_k
    end
    αinj = relaxation_cut(m, x⁻, r⁻)
    @. x⁺ = x⁻ + αinj * r⁻
    return 1.0
end

# Least squares min ‖V α − b‖ by pivoted QR, dropping the columns whose pivot is below ε times the first one
function _lsq_filtered(V::AbstractMatrix, b::AbstractVector, ε::Real)
    F = qr(V, ColumnNorm())
    d = abs.(diag(F.R))
    rank = count(>=(ε * d[1]), d)
    qb = (F.Q' * b)[1:rank]
    α = zeros(size(V, 2))
    α[F.p[1:rank]] = UpperTriangular(F.R[1:rank, 1:rank]) \ qb
    return α
end

#------------------------------ Coupling residual ------------------------------

"""
    coupling_residual_norm(m::StaticNonlinearModel) -> Float64

Coupling residual of the field of `m` in a pass: the norm of the residual its Newton started from in the
last inner solve (the state of the field from the previous pass with the new state of the other fields;
in the first pass of a step, the residual of the load increment). `norm = :residual` of
[`StaggeredStepping`](@ref) relates it to the one of the first pass.
"""
coupling_residual_norm(m::StaticNonlinearModel) = _initial_residual(m.caches.nls)
coupling_residual_norm(m::ComputationalModel) =
    error("HyperFEM: `norm = :residual` needs StaticNonlinearModel fields in the staggered model (got $(typeof(m)))")
_initial_residual(nls::NewtonRaphson) = first(residual_history(nls))
_initial_residual(nls::NonlinearSolver) = error("HyperFEM: `norm = :residual` needs the residual history of NewtonRaphson (got $(typeof(nls)))")

# Relative error with a zero reference: 0 when the quantity is zero too, ∞ otherwise
_relative(a::Float64, ref::Float64) = ref > 0 ? a / ref : (a > 0 ? Inf : 0.0)

#--------------------------------- Stepping ------------------------------------

"""
    StaggeredStepping(; nsteps=20, max_substeps=1, coupling_tol=nothing, norm=:increment,
                      relaxation=:none, maxbisec=15, grow=1.0, easy=3)

Control of the loop of [`solve!`](@ref) for a [`StaggeredModel`](@ref): the stage τ runs from 0 to 1
in increments Δτ, starting at `Δτ0 = 1/nsteps`; inside every step the fields are solved one after
another (a *pass*), at most `max_substeps` times.

- `coupling_tol = nothing` (default): the **fixed loop**, `nsteps` equal steps of exactly
  `max_substeps` passes each, whatever the convergence of the inner solves (the flag reports a
  failure). This is the behaviour of earlier versions, bit for bit; the `NamedTuple`
  `(nsteps = …, nsubsteps = …, maxbisec = …)` is accepted wherever a `StaggeredStepping` is and selects it.
- `coupling_tol = tol`: the passes of a step stop when every field is converged, `err_i ≤ tol`, and the
  step is accepted; a step whose passes reach `max_substeps` without convergence, or in which an inner
  solve fails, is rejected: the state, the lagged state and the Dirichlet values return to the last
  accepted step and the increment is bisected (`maxbisec` consecutive bisections at most, the count
  restarts at every accepted step). After a step accepted in at most `easy` passes the next increment
  grows, `Δτ ← min(grow Δτ, Δτ0)` (`grow = 1`, the default: never). The first pass of a step counts
  for convergence: a field whose solve did not change it is converged.
- `norm = :increment`: `err_i = ‖x⁺_i(k) − x⁺_i(k−1)‖ / ‖x⁺_i(k)‖`, the relative increment of the free
  values of the field in the pass (the solved increment, before relaxation; a field of zero norm is
  converged when its increment is zero). `norm = :residual`: `err_i = ‖R_i(k)‖ / ‖R_i(1)‖`, the residual
  of the field at the start of its solve in the pass (its state of the previous pass with the new state of
  the other fields), relative to the one of the first pass of the step (needs two passes at least; see
  `coupling_residual_norm`).
- `relaxation`: `:none`, `:aitken`, `:iqnils`, a factor ω, an [`Aitken`](@ref), an [`IQNILS`](@ref), or a
  tuple with one of them per field (see `Relaxation`).
"""
struct StaggeredStepping{R}
    nsteps::Int
    max_substeps::Int
    coupling_tol::Union{Nothing,Float64}
    norm::Symbol
    relaxation::R
    maxbisec::Int
    grow::Float64
    easy::Int
    function StaggeredStepping(; nsteps::Int=20, max_substeps::Int=1, nsubsteps::Union{Nothing,Int}=nothing,
        coupling_tol::Union{Nothing,Real}=nothing, norm::Symbol=:increment, relaxation=:none,
        maxbisec::Int=15, grow::Real=1.0, easy::Int=3)
        nsubsteps === nothing || (max_substeps = nsubsteps)       # the name of the NamedTuple of earlier versions
        nsteps >= 1 || error("HyperFEM: StaggeredStepping needs nsteps ≥ 1 (got $nsteps)")
        max_substeps >= 1 || error("HyperFEM: StaggeredStepping needs max_substeps ≥ 1 (got $max_substeps)")
        coupling_tol === nothing || coupling_tol > 0 || error("HyperFEM: StaggeredStepping needs coupling_tol > 0 (got $coupling_tol)")
        norm in (:increment, :residual) || error("HyperFEM: StaggeredStepping: `norm` must be :increment or :residual (got :$norm)")
        maxbisec >= 0 || error("HyperFEM: StaggeredStepping needs maxbisec ≥ 0 (got $maxbisec)")
        grow >= 1 || error("HyperFEM: StaggeredStepping needs grow ≥ 1 (got $grow)")
        easy >= 1 || error("HyperFEM: StaggeredStepping needs easy ≥ 1 (got $easy)")
        relaxation isa Tuple ? map(as_relaxation, relaxation) : as_relaxation(relaxation)   # validated here; a tuple is matched to the fields in solve!
        tol = coupling_tol === nothing ? nothing : Float64(coupling_tol)
        new{typeof(relaxation)}(nsteps, max_substeps, tol, norm, relaxation, maxbisec, Float64(grow), easy)
    end
end

staggered_stepping(s::StaggeredStepping) = s
staggered_stepping(s::NamedTuple) = StaggeredStepping(; s...)
staggered_stepping(s) = error("HyperFEM: `stepping` of a StaggeredModel must be a StaggeredStepping or a NamedTuple (nsteps = …, nsubsteps = …, maxbisec = …); got $(typeof(s))")

"The fixed loop: `nsteps` equal steps of `max_substeps` passes, no rejection of steps."
fixed_loop(s::StaggeredStepping) = s.coupling_tol === nothing

"""
    StaggeredStep

State of the staggered loop: the stage `τ` of the last accepted step (the start of the attempt while
one is running), the current increment `Δτ` and the initial one `Δτ0`, the accepted steps `nsteps`, the
consecutive bisections `nbisect` and the total ones `nbisect_total`.
"""
mutable struct StaggeredStep
    τ::Float64
    Δτ::Float64
    Δτ0::Float64
    nsteps::Int
    nbisect::Int
    nbisect_total::Int
end
StaggeredStep(s::StaggeredStepping) = StaggeredStep(0.0, 1.0 / s.nsteps, 1.0 / s.nsteps, 0, 0, 0)

finished(st::StaggeredStep, s::StaggeredStepping) =
    fixed_loop(s) ? st.nsteps >= s.nsteps : isapprox(st.τ, 1.0; atol=1e-12)

#------------------------------------ Log --------------------------------------

"""
    StaggeredLog

Record of the last [`solve!`](@ref) of a [`StaggeredModel`](@ref) (`get_log(m)`), one entry per
accepted step: the stage `τ` reached, the increment `Δτ`, the passes `nsub`, the bisections `nbisect`
the step needed, the coupling errors `errors` per field after the last pass (see
[`StaggeredStepping`](@ref) for their definition), the relaxation factors `ω` per field of the last
pass (1 for a field without relaxation) and the time `times` spent in the solves of every field over
the passes of the step. `nattempts` counts every attempt (accepted or rejected), `nbisect_total` the
rejected ones, `t_solve` the wall time of the whole solve.
"""
mutable struct StaggeredLog
    τ::Vector{Float64}
    Δτ::Vector{Float64}
    nsub::Vector{Int}
    nbisect::Vector{Int}
    errors::Vector{Vector{Float64}}
    ω::Vector{Vector{Float64}}
    times::Vector{Vector{Float64}}
    nattempts::Int
    nbisect_total::Int
    t_solve::Float64
end
StaggeredLog() = StaggeredLog(Float64[], Float64[], Int[], Int[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], 0, 0, 0.0)

function reset!(log::StaggeredLog)
    foreach(empty!, (log.τ, log.Δτ, log.nsub, log.nbisect, log.errors, log.ω, log.times))
    log.nattempts = 0
    log.nbisect_total = 0
    log.t_solve = 0.0
    return log
end

function record!(log::StaggeredLog, τ, Δτ, nsub, nbisect, errors, ω, times)
    push!(log.τ, τ)
    push!(log.Δτ, Δτ)
    push!(log.nsub, nsub)
    push!(log.nbisect, nbisect)
    push!(log.errors, copy(errors))
    push!(log.ω, copy(ω))
    push!(log.times, copy(times))
    return log
end

"Accepted steps of the log."
num_steps(log::StaggeredLog) = length(log.τ)
"Passes of the accepted steps: total and mean per step."
num_substeps(log::StaggeredLog) = sum(log.nsub; init=0)
mean_substeps(log::StaggeredLog) = num_steps(log) == 0 ? 0.0 : num_substeps(log) / num_steps(log)

function Base.show(io::IO, log::StaggeredLog)
    n = num_steps(log)
    @printf(io, "StaggeredLog: %d steps, %d passes (%.2f per step), %d bisections, %.2f s", n, num_substeps(log),
        mean_substeps(log), log.nbisect_total, log.t_solve)
end

# Output: `verbose = 1` (or `true`) a header, a row per attempt and a summary
_fmt(v::AbstractVector, f) = join((f(x) for x in v), " ")
function staggered_header(v::Int, s::StaggeredStepping)
    v >= 1 || return
    if fixed_loop(s)
        printstyled(@sprintf("Staggered loop: %d steps × %d passes\n", s.nsteps, s.max_substeps); bold=true)
    else
        printstyled(@sprintf("Staggered loop: Δτ0 = %.4f, coupling tolerance %.1e (%s), up to %d passes\n",
            1.0 / s.nsteps, s.coupling_tol, s.norm, s.max_substeps); bold=true)
    end
end
function staggered_row(v::Int, n::Int, τ, Δτ, nsub::Int, errors, ω, ok::Bool, grown::Bool, t)
    v >= 1 || return
    @printf("stag %4d  τ = %.4f  Δτ = %.4f  passes = %2d  err = %s", n, τ, Δτ, nsub, _fmt(errors, e -> @sprintf("%.2e", e)))
    any(!=(1.0), ω) && @printf("  ω = %s", _fmt(ω, w -> @sprintf("%.2f", w)))
    printstyled(ok ? "  ✓" : "  ✗ → bisect"; color=ok ? :green : :yellow)
    grown && printstyled("  ↑ Δτ"; color=:cyan)
    @printf("  %.2f s\n", t)
end
function staggered_summary(v::Int, st::StaggeredStep, log::StaggeredLog)
    v >= 1 || return
    @printf("τ = %.4f reached: %d steps, %d passes (%.2f per step), %d bisections, %.2f s\n", st.τ, num_steps(log),
        num_substeps(log), mean_substeps(log), log.nbisect_total, log.t_solve)
end

#----------------------------------- Model -------------------------------------

"""
    StaggeredModel(models, state⁺, state⁻)

Staggered (partitioned) solve of the coupled problem formed by `models`, a tuple of computational
models (typically [`StaticNonlinearModel`](@ref)s), one per field. `state⁺` are the `FEFunction`s of
the fields (the states the models solve for, `xh` of each model) and `state⁻` their lagged copies, on
their own trial spaces: the forms and the `Coupling` conditions of a field read the other fields through
them, typically as `uh⁻ + (uh⁺ - uh⁻) * τ` with the stage τ of the inner solve.

[`solve!`](@ref) advances the stage of the loop from 0 to 1 by steps; in every step the models are solved
one after another, each over its own inner loop with the load of the step (one or more *passes*, see
[`StaggeredStepping`](@ref)); after every pass the lagged states are set to the new ones. The model keeps
the free values of the states (`caches.x⁺`, `caches.x⁻`), a copy of the last accepted step
(`caches.xₙ`, for the rejection of a step), the work vectors of the relaxation and the
[`StaggeredLog`](@ref) of the last solve (`get_log`).
"""
struct StaggeredModel{A,B,C,D,E} <: ComputationalModel
    compmodels::A
    state⁺::B
    state⁻::C
    caches::D
    applicators⁻::E                   # Dirichlet applicators of the spaces of state⁻
    log::StaggeredLog

    function StaggeredModel(
        CompModels::Tuple{Vararg{ComputationalModel,N}},
        state⁺::Tuple{Vararg{FEFunction,N}},
        state⁻::Tuple{Vararg{FEFunction,N}}) where N

        x⁺ = map(get_free_dof_values, state⁺)
        x⁻ = map(get_free_dof_values, state⁻)
        xₙ = map(copy, x⁺)                       # last accepted step
        r = map(similar, x⁺)                     # increment of the pass (relaxation)
        r⁻ = map(similar, x⁺)                    # increment of the previous pass (Aitken)
        work = (err=fill(NaN, N), ref=zeros(N), ω=ones(N), t=zeros(N))
        caches = (; x⁺, x⁻, xₙ, r, r⁻, work)
        apps⁻ = map((s, m) -> dirichlet_applicator(get_fe_space(s), m.dirichlet), state⁻, CompModels)
        A, B, C, D, E = typeof(CompModels), typeof(state⁺), typeof(state⁻), typeof(caches), typeof(apps⁻)
        return new{A,B,C,D,E}(CompModels, state⁺, state⁻, caches, apps⁻, StaggeredLog())
    end
end

"Log of the last `solve!` of a staggered model (a [`StaggeredLog`](@ref))."
get_log(m::StaggeredModel) = m.log

_applicator(m::ComputationalModel) = get_applicators(m)[1]

# Dirichlet values of the trial spaces of the models and of the lagged states at the load `Λ`, stage 1
function set_dirichlet!(m::StaggeredModel, Λ)
    foreach(x -> TrialFESpace!(get_trial_space(x), _applicator(x), Λ; stage=1.0), m.compmodels)
    foreach((s, a) -> TrialFESpace!(get_fe_space(s), a, Λ; stage=1.0), m.state⁻, m.applicators⁻)
    nothing
end

_call(::Nothing, args...) = nothing
_call(f, args...) = (f(args...); nothing)

"""
    solve!(m::StaggeredModel; stepping, kargsolve, presolver=nothing, evolτ, verbose=true,
           on_substep=nothing, on_step=nothing) -> (x⁺, flagconv)

Staggered solve of the fields of `m` over the stage τ ∈ [0, 1], controlled by `stepping` (a
[`StaggeredStepping`](@ref), or the `NamedTuple` `(nsteps = …, nsubsteps = …, maxbisec = …)` of the fixed
loop). In a step from τ⁻ to τ⁺ every model is solved with the keywords of its entry in `kargsolve`
(`Dict`s or `NamedTuple`s, one per model) and the load `load(stage) = τ⁻ + (τ⁺ − τ⁻) stage` of its inner
loop; `evolτ` maps the stage of the loop to the load (the identity on [0, 1] by default). Returns the
free values of the fields and `flagconv`: `1` when τ = 1 was reached with every inner solve converged,
`0` otherwise (in the fixed loop the solve goes on after a failed inner solve, as in earlier versions;
with a coupling tolerance, when the consecutive bisections exceeded `maxbisec`). The record of the solve
is in `get_log(m)`.

Hooks, all optional (`nothing`: no call):
- `presolver(τ⁻, Δτ)`, before every attempt of a step, with the stage τ⁻ ∈ [0, 1] of the last accepted
  step and the increment to try (as the `presolver` of [`StaticNonlinearModel`](@ref)); a load ramped
  by the loop reads `τ⁻ + Δτ`. With the `NamedTuple` `stepping` of earlier versions it receives instead
  the 0-based index of the step, `presolver(n, Δτ)` (deprecated: pass a `StaggeredStepping`).
- `on_substep(k, errors)`, after every pass `k` of a step, with the coupling errors per field (a tuple).
- `on_step(τ, Δτ, nsub)`, after every accepted step, with the stage reached, the increment of the next
  step and the passes of the step.

The post-processors and hooks of the inner models run inside every pass, also in the passes of a step
that is rejected afterwards; output per accepted step belongs in `on_step`.
"""
function solve!(m::StaggeredModel;
    stepping=(nsteps=20, nsubsteps=1, maxbisec=15),
    presolver=nothing,
    evolτ=t -> max(min(t, 1.0), 0.0),
    verbose=true,
    kargsolve=ntuple(_ -> (;), length(m.compmodels)),
    on_substep=nothing,
    on_step=nothing)

    N = length(m.compmodels)
    length(kargsolve) == N || error("HyperFEM: `kargsolve` has $(length(kargsolve)) entries for $N models")
    legacy_index = stepping isa NamedTuple        # presolver(n, Δτ) with the index of the step, as in earlier versions
    s = staggered_stepping(stepping)
    relax = _relaxations(s.relaxation, N)
    states = map(relaxation_state, relax, m.caches.x⁺)
    legacy_index && presolver !== nothing && Base.depwarn("solve!(::StaggeredModel): a `presolver` with a NamedTuple `stepping` " *
        "receives the 0-based index of the step, which is deprecated; pass `stepping = StaggeredStepping(…)` to " *
        "receive the stage, `presolver(τ⁻, Δτ)` with τ⁻ ∈ [0, 1]", :solve!)
    v = Int(verbose)
    log = reset!(m.log)
    c = m.caches

    set_dirichlet0!(m)
    foreach(copyto!, c.xₙ, c.x⁺)

    flagconv = 1
    st = StaggeredStep(s)
    t_run = time()
    staggered_header(v, s)
    while !finished(st, s)
        ok = step!(m, st, s, relax, states; legacy_index, presolver, evolτ, kargsolve, on_substep, on_step, verbose=v)
        if fixed_loop(s)
            ok || (flagconv = 0)
        elseif st.nbisect > s.maxbisec
            @warn "Maximum number of bisections reached"
            flagconv = 0
            break
        end
    end
    log.t_solve = time() - t_run
    staggered_summary(v, st, log)

    return c.x⁺, flagconv
end

# Dirichlet values at the start of the loop (load 0, stage 0), as in earlier versions
function set_dirichlet0!(m::StaggeredModel)
    foreach(x -> TrialFESpace!(get_trial_space(x), _applicator(x), 0.0), m.compmodels)
    foreach((s, a) -> TrialFESpace!(get_fe_space(s), a, 0.0), m.state⁻, m.applicators⁻)
    nothing
end

"""
    step!(m::StaggeredModel, st::StaggeredStep, stepping::StaggeredStepping, relaxations; kwargs...) -> Bool

One attempt of the staggered loop: the step from the stage `st.τ` with the increment `st.Δτ` (cut at
τ = 1), its passes until convergence or `max_substeps` and, in the fixed loop, the acceptance of the
step, or with a coupling tolerance, its acceptance or its rejection (state, lagged state and Dirichlet
values back to the last accepted step, increment bisected). Returns whether every inner solve converged
and (with a tolerance) the coupling converged. `relaxations` is the tuple of `Relaxation`s of
the fields. See [`solve!`](@ref) for the keywords.
"""
function step!(m::StaggeredModel, st::StaggeredStep, s::StaggeredStepping, relax::Tuple, states::Tuple=map(relaxation_state, relax, m.caches.x⁺);
    legacy_index::Bool=false, presolver=nothing, evolτ=identity, kargsolve, on_substep=nothing, on_step=nothing, verbose::Int=0)

    c = m.caches
    w = c.work
    fixed = fixed_loop(s)
    # target stage: the fixed loop computes it from the index of the step (bit for bit the stages of the
    # fixed loop of earlier versions); the adaptive one accumulates the increments, cut at τ = 1
    if fixed
        Δτ = st.Δτ0
        τ⁺raw = (st.nsteps + 1) * Δτ
    else
        Δτ = min(st.Δτ, 1.0 - st.τ)
        τ⁺raw = st.τ + Δτ
    end
    _call(presolver, legacy_index ? st.nsteps : st.τ, Δτ)
    τ⁻ = evolτ(st.τ)
    τ⁺ = evolτ(τ⁺raw)
    # Inside the step the inner solves run their stage parameter from 0 to 1; the load they apply is
    # the ramp of the step (the drivers pass both to the conditions, see `Prescription`)
    stevol(stage) = τ⁻ + (τ⁺ - τ⁻) * stage

    fill!(w.t, 0.0)
    fill!(w.err, NaN)
    fill!(w.ω, 1.0)
    log = m.log
    log.nattempts += 1
    t_step = time()
    nsub = 0
    inner_ok = true
    converged = false
    while nsub < s.max_substeps
        nsub += 1
        flags = pass!(m, nsub, stevol, τ⁺, relax, states, kargsolve, s.norm)
        inner_ok &= all(==(1), flags)
        _call(on_substep, nsub, ntuple(i -> w.err[i], Val(length(m.compmodels))))
        fixed && continue                                  # the fixed loop runs every pass, whatever happened
        inner_ok || break
        if all(e -> e <= s.coupling_tol, w.err)
            converged = true
            break
        end
    end
    ok = fixed ? inner_ok : (inner_ok && converged)
    t = time() - t_step

    if fixed || ok
        Δτnext = fixed ? Δτ : grown_increment(s.grow, s.easy, Δτ, st.Δτ0, true, nsub)
        staggered_row(verbose, st.nsteps + 1, τ⁺raw, Δτ, nsub, w.err, w.ω, ok, Δτnext > Δτ, t)
        record!(log, τ⁺raw, Δτ, nsub, st.nbisect, w.err, w.ω, w.t)
        st.nsteps += 1
        st.nbisect = 0
        st.τ = fixed ? st.nsteps * Δτ : τ⁺raw
        st.Δτ = Δτnext
        foreach(copyto!, c.xₙ, c.x⁺)
        foreach(accept_step!, relax, states)
        _call(on_step, st.τ, st.Δτ, nsub)
    else
        staggered_row(verbose, st.nsteps + 1, τ⁺raw, Δτ, nsub, w.err, w.ω, false, false, t)
        @warn (inner_ok ? "Staggered coupling did not converge in $nsub passes at τ = $τ⁺raw" :
                          "An inner solve failed at τ = $τ⁺raw") * ": bisection performed!"
        # back to the last accepted step: states, lagged states and the Dirichlet values of their spaces
        foreach(copyto!, c.x⁺, c.xₙ)
        foreach(copyto!, c.x⁻, c.xₙ)
        set_dirichlet!(m, τ⁻)
        st.Δτ = Δτ / 2
        st.nbisect += 1
        st.nbisect_total += 1
        log.nbisect_total += 1
    end
    return ok
end

# One pass of a step: the trial spaces at the load τ⁺ (stage 1), the solve of every model in order with
# the ramp of the step (the relaxation of its field and its coupling error right after it, so that the next
# models read the relaxed state), then the lagged states (Dirichlet values and free values) set to the new
# ones. Returns the flags of the inner solves.
function pass!(m::StaggeredModel, k::Int, stevol, τ⁺, relax::Tuple, states::Tuple, kargsolve, errnorm::Symbol)
    c = m.caches
    w = c.work
    foreach(x -> TrialFESpace!(get_trial_space(x), _applicator(x), τ⁺; stage=1.0), m.compmodels)
    idx = ntuple(identity, Val(length(m.compmodels)))
    flags = map(idx, m.compmodels, kargsolve, relax, states, c.x⁺, c.x⁻, c.r, c.r⁻) do i, mi, kw, rl, st, x⁺, x⁻, r, r⁻
        t0 = time()
        flag = solve!(mi; load=stevol, kw...)[2]
        w.t[i] += time() - t0
        w.ω[i] = relax!(rl, st, mi, k, w.ω[i], x⁺, x⁻, r, r⁻)
        if errnorm === :residual
            R = coupling_residual_norm(mi)
            k == 1 && (w.ref[i] = R)
            w.err[i] = _relative(R, w.ref[i])
        else
            w.err[i] = _relative(norm(r), norm(x⁺))
        end
        flag
    end
    foreach((s, a) -> TrialFESpace!(get_fe_space(s), a, τ⁺; stage=1.0), m.state⁻, m.applicators⁻)
    foreach(copyto!, c.x⁻, c.x⁺)
    return flags
end

# Getters
get_state(m::StaggeredModel) = m.state⁺
get_spaces(m::StaggeredModel) = map(get_trial_space, m.compmodels)
get_applicators(m::StaggeredModel) = map(_applicator, m.compmodels)
