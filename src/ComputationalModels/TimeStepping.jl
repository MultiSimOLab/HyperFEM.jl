#*******************************************************************************
#    					 Control of the time step of the transient driver
#*******************************************************************************

"""
    grown_increment(grow, easy, Δ, Δ0, ok, iterations) -> Δnext

Increment of the next attempt after an attempt that converged (`ok`) in `iterations`: the increment
grows, `min(grow Δ, Δ0)`, when `grow > 1` and the attempt was easy (at most `easy` iterations); it
stays otherwise. Shared by the load stepping (`LoadStep`, ΔΛ) and the time stepping (`TimeStep`, Δt).
"""
grown_increment(grow::Float64, easy::Int, Δ::Float64, Δ0::Float64, ok::Bool, iterations::Int) =
    ok && grow > 1 && iterations <= easy ? min(grow * Δ, Δ0) : Δ

"""
    TimeStepping(; Δt, t_end, t0=0.0, maxbisec=15, grow=1.0, easy=10, Δtmax=Δt, tstops=())

Control of the time loop of a transient model with a fixed increment: the time runs from `t0` to
`t_end` in increments `Δt` (the last one cut at `t_end`, and any of them cut at the next of the exact
output times `tstops`; the proposed increment is kept after a cut). A failed attempt (the stage solve
did not converge) is bisected from the last accepted step; `maxbisec` is the maximum number of
**consecutive** bisections, the count restarts at every accepted step. After a step accepted in at
most `easy` iterations the increment grows, `Δt ← min(grow Δt, Δt0, Δtmax)`; with `grow = 1` (the
default) a bisected increment is never grown again. The analogue of [`LoadStepping`](@ref) in time.
"""
struct TimeStepping
    Δt::Float64
    t_end::Float64
    t0::Float64
    maxbisec::Int
    grow::Float64
    easy::Int
    Δtmax::Float64
    tstops::Vector{Float64}
    function TimeStepping(; Δt::Real, t_end::Real, t0::Real=0.0, maxbisec::Int=15, grow::Real=1.0, easy::Int=10,
                           Δtmax::Real=Δt, tstops=())
        Δt > 0 || error("HyperFEM: TimeStepping needs Δt > 0 (got $Δt)")
        t_end > t0 || error("HyperFEM: TimeStepping needs t_end > t0 (got t0 = $t0, t_end = $t_end)")
        maxbisec >= 0 || error("HyperFEM: TimeStepping needs maxbisec ≥ 0 (got $maxbisec)")
        grow >= 1 || error("HyperFEM: TimeStepping needs grow ≥ 1 (got $grow)")
        Δtmax >= Δt || error("HyperFEM: TimeStepping needs Δtmax ≥ Δt (got Δt = $Δt, Δtmax = $Δtmax)")
        new(Float64(Δt), Float64(t_end), Float64(t0), maxbisec, Float64(grow), easy, Float64(Δtmax), _tstops(tstops, t0, t_end))
    end
end

"""
    AdaptiveStepping(; Δt, t_end, t0=0.0, rtol=1e-3, atol=1e-6, safety=0.9, fmin=0.2, fmax=2.0,
                     Δtmin=1e-8 (t_end - t0), Δtmax=t_end - t0, controller=:PI, maxreject=20,
                     maxbisec=15, tstops=())

Control of the time loop of a transient model by the local error estimate of its scheme. `Δt` is the
first increment. After a converged attempt the weighted error `err` of the step (the RMS over the
degrees of freedom with a time derivative of the estimate divided by `atol + rtol max(|x|, |xₙ|)`)
decides: the step is accepted when `err ≤ 1` and rejected otherwise (retried with the new increment;
`maxreject` consecutive rejections stop the run). The next increment is `Δt · f` with the factor of
the `controller`, `:I` (`f = safety err^(-1/(p+1))`, `p` the order of the scheme) or `:PI`
(`f = safety err^(-kI) (err_prev/err)^(kP)`, `kI = 0.3/(p+1)`, `kP = 0.4/(p+1)`, Gustafsson; the
`:I` factor on the first step and after a rejection), `f` clipped to `[fmin, fmax]`, the increment to
`[Δtmin, Δtmax]`; the increment is never grown right after a rejection. A failed attempt (the stage
solve did not converge) is bisected as in [`TimeStepping`](@ref), without consulting the controller;
`maxbisec` consecutive bisections stop the run. `tstops` are reached exactly.
"""
struct AdaptiveStepping
    Δt::Float64
    t_end::Float64
    t0::Float64
    rtol::Float64
    atol::Float64
    safety::Float64
    fmin::Float64
    fmax::Float64
    Δtmin::Float64
    Δtmax::Float64
    controller::Symbol
    maxreject::Int
    maxbisec::Int
    tstops::Vector{Float64}
    function AdaptiveStepping(; Δt::Real, t_end::Real, t0::Real=0.0, rtol::Real=1e-3, atol::Real=1e-6, safety::Real=0.9,
                               fmin::Real=0.2, fmax::Real=2.0, Δtmin::Real=1e-8 * (t_end - t0), Δtmax::Real=t_end - t0,
                               controller::Symbol=:PI, maxreject::Int=20, maxbisec::Int=15, tstops=())
        Δt > 0 || error("HyperFEM: AdaptiveStepping needs Δt > 0 (got $Δt)")
        t_end > t0 || error("HyperFEM: AdaptiveStepping needs t_end > t0 (got t0 = $t0, t_end = $t_end)")
        rtol > 0 && atol >= 0 || error("HyperFEM: AdaptiveStepping needs rtol > 0 and atol ≥ 0 (got rtol = $rtol, atol = $atol)")
        0 < safety <= 1 || error("HyperFEM: AdaptiveStepping needs 0 < safety ≤ 1 (got $safety)")
        0 < fmin < 1 < fmax || error("HyperFEM: AdaptiveStepping needs 0 < fmin < 1 < fmax (got fmin = $fmin, fmax = $fmax)")
        0 < Δtmin <= Δt <= Δtmax || error("HyperFEM: AdaptiveStepping needs 0 < Δtmin ≤ Δt ≤ Δtmax (got Δtmin = $Δtmin, Δt = $Δt, Δtmax = $Δtmax)")
        controller in (:I, :PI) || error("HyperFEM: AdaptiveStepping controller must be :I or :PI (got :$controller)")
        maxreject >= 0 && maxbisec >= 0 || error("HyperFEM: AdaptiveStepping needs maxreject ≥ 0 and maxbisec ≥ 0")
        new(Float64(Δt), Float64(t_end), Float64(t0), Float64(rtol), Float64(atol), Float64(safety), Float64(fmin), Float64(fmax),
            Float64(Δtmin), Float64(Δtmax), controller, maxreject, maxbisec, _tstops(tstops, t0, t_end))
    end
end

const TimeControl = Union{TimeStepping,AdaptiveStepping}

function _tstops(tstops, t0, t_end)
    ts = sort!(Float64[t for t in tstops if t0 < t < t_end])
    unique!(ts)
end

as_time_stepping(s::TimeControl) = s
as_time_stepping(s::NamedTuple) = TimeStepping(; s...)
as_time_stepping(s) = error("HyperFEM: `stepping` of a transient model must be a TimeStepping, an AdaptiveStepping or a " *
                            "NamedTuple (Δt = …, t_end = …); got $(typeof(s))")

"""
    TimeStep

State of the time loop: the time `t` of the last accepted step, the proposed increment `Δt` of the next
attempt, the initial one `Δt0`, the increment `Δt_prev` and the error `err_prev` of the last accepted
step (0 and NaN before the first), the accepted steps `nsteps`, the consecutive and total bisections
`nbisect`/`nbisect_total`, the consecutive and total rejections `nreject`/`nreject_total`, whether the
last attempt was rejected (`rejected`, the controller does not grow the increment right after one),
and the running attempt: its target `t_target`, its increment `Δt_attempt` and whether it was `cut`
(less than the proposed increment, at `t_end` or at a `tstop`), set by [`next_target!`](@ref).
"""
mutable struct TimeStep
    t::Float64
    Δt::Float64
    Δt0::Float64
    Δt_prev::Float64
    err_prev::Float64
    nsteps::Int
    nbisect::Int
    nbisect_total::Int
    nreject::Int
    nreject_total::Int
    rejected::Bool
    t_target::Float64
    Δt_attempt::Float64
    cut::Bool
end
TimeStep(s::TimeControl) = TimeStep(s.t0, s.Δt, s.Δt, 0.0, NaN, 0, 0, 0, 0, 0, false, s.t0, s.Δt, false)

# Tolerance of the comparisons in time: a fraction of the initial increment
_ttol(st::TimeStep) = 1e-10 * st.Δt0

finished(st::TimeStep, s::TimeControl) = st.t >= s.t_end - _ttol(st)

"""
    next_target!(st, stepping) -> (t_target, Δt_attempt)

Target time and increment of the next attempt: the proposed increment `st.Δt` from `st.t`, cut at
`t_end` and at the first `tstop` beyond `st.t`; the proposed increment is not changed by a cut.
"""
function next_target!(st::TimeStep, s::TimeControl)
    tol = _ttol(st)
    Δt = st.Δt
    t = st.t + Δt
    cut = false
    if t > s.t_end - tol
        t, Δt, cut = s.t_end, s.t_end - st.t, true
    end
    for ts in s.tstops
        if ts > st.t + tol
            if t > ts - tol
                t, Δt, cut = ts, ts - st.t, true
            end
            break
        end
    end
    st.t_target, st.Δt_attempt, st.cut = t, Δt, cut
    return t, Δt
end

"""
    accept!(st, stepping, iterations, err=NaN)

The running attempt is accepted: `st.t` moves to its target, the counters of consecutive failures
restart and the increment of the next attempt is set — `TimeStepping`: the proposed increment grows
after an easy step (`grown_increment`), bounded by `Δtmax`; `AdaptiveStepping`: the increment of the
controller for the error `err` of the step (`order` is the order of the scheme).
"""
function accept!(st::TimeStep, s::TimeStepping, iterations::Int, err::Float64=NaN)
    Δt = st.cut ? st.Δt : grown_increment(s.grow, s.easy, st.Δt_attempt, st.Δt0, true, iterations)
    _accepted!(st, min(Δt, s.Δtmax), err)
end
function accept!(st::TimeStep, s::AdaptiveStepping, iterations::Int, err::Float64, order::Int)
    Δt = next_increment(s, st.Δt_attempt, err, st.err_prev, order, st.rejected)
    _accepted!(st, Δt, err)
end
function _accepted!(st::TimeStep, Δt::Float64, err::Float64)
    st.nsteps += 1
    st.nbisect = 0
    st.nreject = 0
    st.rejected = false
    st.t = st.t_target
    st.Δt_prev = st.Δt_attempt
    st.err_prev = err
    st.Δt = Δt
    nothing
end

"""
    bisect!(st, stepping)

The running attempt failed (the stage solve did not converge): the next attempt tries half of its
increment from the same time.
"""
function bisect!(st::TimeStep, ::TimeControl)
    st.Δt = st.Δt_attempt / 2
    st.nbisect += 1
    st.nbisect_total += 1
    nothing
end

"""
    reject!(st, stepping, err, order)

The running attempt converged but its error `err > 1`: the next attempt tries the increment of the
`:I` controller from the same time (`AdaptiveStepping` only).
"""
function reject!(st::TimeStep, s::AdaptiveStepping, err::Float64, order::Int)
    st.Δt = next_increment(s, st.Δt_attempt, err, NaN, order, true)
    st.nreject += 1
    st.nreject_total += 1
    st.rejected = true
    nothing
end

"""
    exhausted(st, stepping) -> Bool

Whether the loop must stop: more than `maxbisec` consecutive bisections, more than `maxreject`
consecutive rejections, or an increment below `Δtmin` (the last two with `AdaptiveStepping`).
"""
exhausted(st::TimeStep, s::TimeStepping) = st.nbisect > s.maxbisec
exhausted(st::TimeStep, s::AdaptiveStepping) = st.nbisect > s.maxbisec || st.nreject > s.maxreject || st.Δt < s.Δtmin

"""
    next_increment(stepping::AdaptiveStepping, Δt, err, err_prev, order, after_rejection) -> Δt_new

Increment proposed by the controller after a step of increment `Δt` with weighted error `err`: the
`:I` factor `safety err^(-1/(p+1))` (always after a rejection, when `err_prev` is `NaN`, and with
`controller = :I`) or the `:PI` factor `safety err^(-kI) (err_prev/err)^kP`; the factor is clipped to
`[fmin, fmax]` (`fmax` for a zero error) and to 1 after a rejection; the increment to `[Δtmin, Δtmax]`.
"""
function next_increment(s::AdaptiveStepping, Δt::Float64, err::Float64, err_prev::Float64, p::Int, after_rejection::Bool)
    f = if err <= 0
        s.fmax
    elseif after_rejection || isnan(err_prev) || s.controller === :I
        s.safety * err^(-1 / (p + 1))
    else
        kI, kP = 0.3 / (p + 1), 0.4 / (p + 1)
        s.safety * err^(-kI) * (err_prev / err)^kP
    end
    f = clamp(f, s.fmin, s.fmax)
    after_rejection && (f = min(f, 1.0))
    clamp(Δt * f, s.Δtmin, s.Δtmax)
end

"""
    weighted_error(e, x, xn, rtol, atol, mask) -> err

RMS of `e[i] / (atol + rtol max(|x[i]|, |xn[i]|))` over the entries with `mask[i]` (the degrees of
freedom with a time derivative); 0 when none. One global reduction.
"""
function weighted_error(e::AbstractVector, x::AbstractVector, xn::AbstractVector, rtol::Float64, atol::Float64, mask::AbstractVector{Bool})
    s = 0.0
    n = 0
    @inbounds for i in eachindex(e)
        mask[i] || continue
        w = atol + rtol * max(abs(x[i]), abs(xn[i]))
        s += (e[i] / w)^2
        n += 1
    end
    n == 0 ? 0.0 : sqrt(s / n)
end
