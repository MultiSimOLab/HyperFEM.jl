"""
    NewtonTolerances(; maxiter=100, atol=1e-12, rtol=1e-6, ctol=0.0)

Stopping criteria of [`NewtonRaphson`](@ref): an iterate is accepted when `‖R‖ < atol` or
`‖R‖/‖R₀‖ < rtol` (`R₀` the residual at the start of the solve), and the solve stops without
updating the solution when the curvature `|b·dx|` has dropped below `ctol` times its value at the
first iteration (`b = -R`, `dx` the Newton increment; `ctol = 0` disables the criterion). `maxiter`
is the maximum number of iterations. In the projected mode of [`HybridTangent`](@ref) the criterion
is the absolute one of the strategy.
"""
struct NewtonTolerances
  maxiter::Int
  atol::Float64
  rtol::Float64
  ctol::Float64
  function NewtonTolerances(; maxiter::Integer=100, atol::Real=1e-12, rtol::Real=1e-6, ctol::Real=0.0)
    maxiter >= 0 || throw(ArgumentError("NewtonTolerances: maxiter must be ≥ 0"))
    new(maxiter, atol, rtol, ctol)
  end
end

"""
    NewtonLog(; maxiter, verbose=0, name="Newton-Raphson")

History of a [`NewtonRaphson`](@ref) solve: residual norms `residuals[1:num_iters+1]`
(`residuals[1]` = `‖R₀‖`), step lengths `alphas[1:num_iters]`, the tangent of every iteration
`modes[1:num_iters]` (`:consistent` / `:projected`) and the stop `reason` (`:converged_atol`,
`:converged_rtol`, `:converged_ctol`, `:converged_projected`, `:maxiter`, `:linesearch_failed`,
`:linear_failed`, or `:running` while solving). The storage is allocated once and reused by every
solve; counters over the whole run (`nsolves`, `niters`, `nswitches`, `t_solve`) accumulate until
`reset!`. Verbosity: `0` silent, `1` a line per load step (printed by the drivers), `2` a row per
Newton iteration (step, ‖R‖, ‖R‖/‖R₀‖, α — marked † when limited by the injectivity step —, tangent,
linear solver, time), switches and failures as indented lines and a closing ✓/✗ line, `3` the detail of
every line search. [`print_summary`](@ref) prints the counters of a run.
"""
mutable struct NewtonLog
  const residuals::Vector{Float64}
  const alphas::Vector{Float64}
  const modes::Vector{Symbol}
  num_iters::Int
  reason::Symbol
  const verbose::Int
  const name::String
  # counters over the whole run (reset with `reset!`): solves, iterations, projected iterations, switches,
  # and the time split of the solves: assembly of the tangent, setup of the linear solver (factorisation),
  # linear solves, line searches (with their residuals), total
  nsolves::Int
  niters::Int
  nprojected::Int
  nswitches::Int
  t_assembly::Float64
  t_setup::Float64
  t_linear::Float64
  t_linesearch::Float64
  t_solve::Float64
  function NewtonLog(; maxiter::Integer=100, verbose=0, name::AbstractString="Newton-Raphson")
    v = _verbose_level(verbose)
    new(zeros(maxiter + 1), zeros(maxiter + 1), fill(:consistent, maxiter + 1), 0, :running, v, String(name),
        0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
  end
end

_verbose_level(v::Bool) = v ? 2 : 0
_verbose_level(v::Integer) = Int(v)
_verbose_level(v) = Int(v)   # enums with an integer representation (e.g. GridapSolvers' levels)

"Residual norms of the last solve, `‖R₀‖` first."
residual_history(log::NewtonLog) = view(log.residuals, 1:log.num_iters+1)
"Step lengths of the last solve."
step_history(log::NewtonLog) = view(log.alphas, 1:log.num_iters)
"Tangent of every iteration of the last solve (`:consistent` / `:projected`)."
mode_history(log::NewtonLog) = view(log.modes, 1:log.num_iters)
"Number of iterations of the last solve."
num_iterations(log::NewtonLog) = log.num_iters
"Stop reason of the last solve (see [`NewtonLog`](@ref))."
stop_reason(log::NewtonLog) = log.reason
"`true` if the last solve converged (by any of the criteria)."
converged(log::NewtonLog) = log.reason in (:converged_atol, :converged_rtol, :converged_ctol, :converged_projected)
verbosity(log::NewtonLog) = log.verbose

function reset!(log::NewtonLog)
  log.nsolves = 0
  log.niters = 0
  log.nprojected = 0
  log.nswitches = 0
  log.t_assembly = 0.0
  log.t_setup = 0.0
  log.t_linear = 0.0
  log.t_linesearch = 0.0
  log.t_solve = 0.0
  log
end

function _start!(log::NewtonLog, r0::Float64)
  log.num_iters = 0
  log.reason = :running
  log.residuals[1] = r0
  log.nsolves += 1
  log
end

@inline function _record!(log::NewtonLog, r::Float64, α::Float64, mode::Symbol)
  n = log.num_iters + 1
  if n + 1 > length(log.residuals)
    resize!(log.residuals, 2n)
    resize!(log.alphas, 2n)
    resize!(log.modes, 2n)
  end
  log.residuals[n+1] = r
  log.alphas[n] = α
  log.modes[n] = mode
  log.num_iters = n
  log.niters += 1
  mode === :projected && (log.nprojected += 1)
  log
end

# Convergence test of the consistent mode: ‖R‖ < atol or ‖R‖/‖R₀‖ < rtol.
@inline function _check(tols::NewtonTolerances, r::Float64, r0::Float64)
  r < tols.atol && return :converged_atol
  r < tols.rtol * r0 && return :converged_rtol
  :running
end

# Printing (verbose ≥ 2): a row per iteration, switches and failures as indented lines, a closing line
# with the stop reason. Colours only on a terminal (`printstyled` is plain text on files and in CI).
_mark(ok::Bool) = ok ? "✓" : "✗"
_styled(s, color) = (printstyled(s; color=color); nothing)

function _print_header(log::NewtonLog, r0)
  log.verbose >= 2 || return
  @printf("  %4s  %11s  %11s  %9s  %-10s  %-12s  %8s\n", "it", "‖R‖", "‖R‖/‖R₀‖", "α", "tangent", "linear", "t[s]")
  @printf("  %4d  %11.3e  %11.3e\n", 0, r0, 1.0)
end

function _print_row(log::NewtonLog, r, r0, α, limited::Bool, mode, linear::String, t)
  log.verbose >= 2 || return
  αs = @sprintf("%8.4f", α) * (limited ? "†" : " ")
  @printf("  %4d  %11.3e  %11.3e  %9s  %-10s  %-12s  %8.3f\n", log.num_iters, r, r / r0, αs, mode, linear, t)
end

function _print_detail(log::NewtonLog, ls)
  log.verbose >= 3 || return
  info = linesearch_info(ls)
  isempty(info) || println("        line search: ", info)
end

function _print_switch(log::NewtonLog, to::Symbol, r)
  log.verbose >= 2 || return
  _styled("        → $(to) tangent (‖R‖ = $(@sprintf("%.3e", r)))\n", :yellow)
end

const _REASON_TEXT = Dict(
  :converged_atol => "converged: ‖R‖ < atol",
  :converged_rtol => "converged: ‖R‖/‖R₀‖ < rtol",
  :converged_ctol => "converged: curvature |b·dx| < ctol",
  :converged_projected => "converged: ‖R‖ < atol_projected (projected tangent)",
  :maxiter => "maximum number of iterations",
  :linesearch_failed => "line search failed",
  :linear_failed => "linear solver did not converge",
)

reason_text(reason::Symbol) = get(_REASON_TEXT, reason, string(reason))

function _print_end(log::NewtonLog)
  log.verbose >= 2 || return
  ok = converged(log)
  _styled("  $(_mark(ok)) ", ok ? :green : :red)
  println("$(log.name): $(reason_text(log.reason)) after $(log.num_iters) iterations, ‖R‖ = $(@sprintf("%.3e", log.residuals[log.num_iters+1]))")
end

"""
    print_summary(log::NewtonLog; io=stdout)

Counters of the run since the last `reset!`: solves, iterations (share with the projected tangent,
switches) and the time split of the solves (assembly, factorisation / setup, linear solves, line
searches). The drivers print it at the end of `solve!` with `verbose ≥ 1`.
"""
function print_summary(log::NewtonLog; io=stdout)
  n = max(log.niters, 1)
  pproj = 100 * log.nprojected / n
  rest = log.t_solve - log.t_assembly - log.t_setup - log.t_linear - log.t_linesearch
  println(io, "  $(log.name): $(log.nsolves) solves, $(log.niters) iterations ($(@sprintf("%.0f", pproj)) % projected, $(log.nswitches) switches)")
  @printf(io, "  time %.2f s: assembly %.2f, setup %.2f, linear solve %.2f, line search %.2f, other %.2f\n",
          log.t_solve, log.t_assembly, log.t_setup, log.t_linear, log.t_linesearch, max(rest, 0.0))
end

# Name of the linear solver in the rows (plus the iterations and residual of an iterative one, `linear_info`)
_linear_name(ls::Algebra.LinearSolver) = string(nameof(typeof(ls)))
_linear_name(::Algebra.LUSolver) = "LU"
_linear_name(::Algebra.BackslashSolver) = "backslash"

"""
    linear_info(ns) -> String

Iterations and residual of the last solve of an iterative linear solver, for the verbose output
(`""` for direct solvers; specialised by the iterative solvers' setups).
"""
linear_info(::Algebra.NumericalSetup) = ""

"""
    linesearch_info(ls) -> String

Detail of the last search of a line search for the verbose output (`verbose = 3`); `""` by default.
"""
linesearch_info(::AbstractLineSearch) = ""

"""
    injectivity_limit(ls) -> Float64

Injectivity step the last search started from (`NaN` for line searches without an injectivity test),
to mark in the output the steps limited by it.
"""
injectivity_limit(::AbstractLineSearch) = NaN


# ------------------------------------------------------------------------------------------------
# Tangent strategies
# ------------------------------------------------------------------------------------------------

"""
Strategy of [`NewtonRaphson`](@ref) deciding which tangent is assembled and what to do when the line
search fails: [`ConsistentTangent`](@ref) (default) or [`HybridTangent`](@ref).
"""
abstract type TangentStrategy end

"""
    ConsistentTangent()

Always the consistent Jacobian; a failed line search fails the solve (`:linesearch_failed`) and the
driver bisects the load step.
"""
struct ConsistentTangent <: TangentStrategy end

"""
    HybridTangent(; hybback=1e-2, atol_projected)

Hybrid Newton: the consistent tangent while the line search works; when it fails, the solve switches
to the projected tangent (the eigenvalues of the material tangent clamped from below at every
quadrature point: `HessianRegularization(model; δ, switch)` sharing one `TangentSwitch`, registered in
the driver with `tangent_switch = switch`) and records `‖R‖` at the switch; it returns to the
consistent tangent once `‖R‖ < hybback · ‖R‖_switch`, to finish with quadratic convergence. Every
solve starts with the consistent tangent. In projected mode the convergence criterion is the absolute
`‖R‖ < atol_projected`, a number on the scale of the reactions given by the user (the relative
criterion is unreliable there). A failed line search in projected mode fails the solve.
"""
struct HybridTangent <: TangentStrategy
  hybback::Float64
  atol_projected::Float64
  function HybridTangent(; hybback::Real=1e-2, atol_projected::Real)
    0 < hybback < 1 || throw(ArgumentError("HybridTangent: hybback must be in (0, 1)"))
    atol_projected > 0 || throw(ArgumentError("HybridTangent: atol_projected must be > 0"))
    new(hybback, atol_projected)
  end
end

# Mode at the start of a solve
_start_mode(::TangentStrategy) = :consistent

# Convergence test in the current mode
@inline _check(::ConsistentTangent, tols, mode, r, r0) = _check(tols, r, r0)
@inline function _check(t::HybridTangent, tols, mode, r, r0)
  mode === :projected && return r < t.atol_projected ? :converged_projected : :running
  _check(tols, r, r0)
end

# Mode for the next tangent, decided after the linear solve and before the line search
@inline _next_mode(::ConsistentTangent, mode, r, r_sw) = mode
@inline _next_mode(t::HybridTangent, mode, r, r_sw) = (mode === :projected && r < t.hybback * r_sw) ? :consistent : mode

# Plan B after a failed line search: the mode to switch to, or `nothing` to stop
_on_failure(::ConsistentTangent, mode) = nothing
_on_failure(::HybridTangent, mode) = mode === :consistent ? :projected : nothing


"""
    NewtonRaphson(ls; maxiter=100, atol=1e-12, rtol=1e-6, ctol=0.0, verbose=0, name="Newton-Raphson",
                  linesearch=FullStepLS(), tangent=ConsistentTangent())

Newton–Raphson solver of HyperFEM, a Gridap `NonlinearSolver`. `ls` is the linear solver of the
tangent systems (any Gridap `LinearSolver`; see [`numerical_setup_state!`](@ref) for solvers whose setup
depends on the current state `x`).
The stopping criteria are those of [`NewtonTolerances`](@ref); `linesearch` chooses the step length
along the Newton direction ([`FullStepLS`](@ref), [`BacktrackingLS`](@ref), [`InjectivityLS`](@ref));
`tangent` decides which Jacobian is assembled and what happens when the line search fails
([`ConsistentTangent`](@ref): the solve fails and the driver bisects; [`HybridTangent`](@ref): switch
to the projected tangent); `verbose` as in [`NewtonLog`](@ref).

The former name `Newton_RaphsonSolver` is deprecated and kept as an alias for one release.
"""
struct NewtonRaphson{L<:Algebra.LinearSolver,S<:AbstractLineSearch,T<:TangentStrategy} <: Algebra.NonlinearSolver
  ls::L
  tols::NewtonTolerances
  log::NewtonLog
  linesearch::S
  tangent::T
end

function NewtonRaphson(ls::Algebra.LinearSolver; maxiter=100, atol=1e-12, rtol=1.e-6, ctol=0.0, verbose=0,
                       name="Newton-Raphson", linesearch::AbstractLineSearch=FullStepLS(),
                       tangent::TangentStrategy=ConsistentTangent())
  tols = NewtonTolerances(; maxiter, atol, rtol, ctol)
  log = NewtonLog(; maxiter, verbose, name)
  NewtonRaphson(ls, tols, log, linesearch, tangent)
end

NewtonRaphson(ls, tols, log, linesearch) = NewtonRaphson(ls, tols, log, linesearch, ConsistentTangent())

AbstractTrees.children(s::NewtonRaphson) = [s.ls]

converged(nls::NewtonRaphson) = converged(nls.log)
residual_history(nls::NewtonRaphson) = residual_history(nls.log)
step_history(nls::NewtonRaphson) = step_history(nls.log)
mode_history(nls::NewtonRaphson) = mode_history(nls.log)
num_iterations(nls::NewtonRaphson) = num_iterations(nls.log)
stop_reason(nls::NewtonRaphson) = stop_reason(nls.log)
verbosity(nls::NewtonRaphson) = verbosity(nls.log)
get_tolerances(nls::NewtonRaphson) = nls.tols

"""
    NewtonCache(A, b, dx, ns[, w])

Work data of a [`NewtonRaphson`](@ref) solve: tangent `A`, residual `b`, increment `dx`, the
numerical setup `ns` of the linear solver and the work vector `w` of the line searches. Parametric
so that the matrix may be any operator a linear solver accepts (e.g. a matrix-free Jacobian).
"""
struct NewtonCache{TA,TB,TNS<:Algebra.NumericalSetup}
  A::TA
  b::TB
  dx::TB
  ns::TNS
  w::TB
end

NewtonCache(A, b, dx, ns) = NewtonCache(A, b, dx, ns, similar(dx))

function NewtonCache(x::AbstractVector, nls::NewtonRaphson, op::Algebra.NonlinearOperator)
  set_tangent!(op, :consistent)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup_state(ss, A, x)
  NewtonCache(A, b, dx, ns, similar(dx))
end

function Algebra.solve!(x::AbstractVector, nls::NewtonRaphson, op::Algebra.NonlinearOperator, ::Nothing)
  cache = NewtonCache(x, nls, op)
  _solve_nr!(x, cache, nls, op)
  return cache
end

function Algebra.solve!(x::AbstractVector, nls::NewtonRaphson, op::Algebra.NonlinearOperator, cache::NewtonCache)
  A, b, ns = cache.A, cache.b, cache.ns
  log = nls.log
  t1 = time()
  residual!(b, op, x)
  set_tangent!(op, :consistent)
  jacobian!(A, op, x)
  t2 = time()
  numerical_setup_state!(ns, A, x)
  t3 = time()
  log.t_assembly += t2 - t1
  log.t_setup += t3 - t2
  _solve_nr!(x, cache, nls, op)
  log.t_solve += t3 - t1
  return cache
end

"""
    numerical_setup_state(ss, A, x) -> ns
    numerical_setup_state!(ns, A, x)

Numerical setup of the linear solver for the tangent `A` at the current state `x` of the Newton
iteration. By default `x` is dropped (Gridap's `numerical_setup(ss, A)` / `numerical_setup!(ns, A)`);
linear solvers whose setup depends on the state (preconditioners built from the state, matrix-free
operators) specialise these two methods for their setup types.
"""
numerical_setup_state(ss::Algebra.SymbolicSetup, A, x) = numerical_setup(ss, A)
numerical_setup_state!(ns::Algebra.NumericalSetup, A, x) = numerical_setup!(ns, A)

"""
    linear_converged(ns) -> Bool

Whether the last `solve!` of the numerical setup `ns` converged. `true` by default (direct
solvers); iterative solvers (e.g. the PETSc extension) specialise it so that a failed linear solve
fails the Newton iteration like a failed line search.
"""
linear_converged(::Algebra.NumericalSetup) = true

function _solve_nr!(x, cache::NewtonCache, nls::NewtonRaphson, op)
  A, b, dx, ns, w = cache.A, cache.b, cache.dx, cache.ns, cache.w
  log, tols, linesearch, tangent = nls.log, nls.tols, nls.linesearch, nls.tangent
  t0 = time()
  lname = log.verbose >= 2 ? _linear_name(nls.ls) : ""

  mode = _start_mode(tangent)           # the tangent in A was assembled in this mode
  r_sw = 0.0                            # ‖R‖ at the last switch to the projected tangent
  r = r0 = norm(b)
  _start!(log, r0)
  _print_header(log, r0)
  reason = _check(tangent, tols, mode, r0, r0)
  if reason === :running && tols.maxiter == 0
    reason = :maxiter
  end
  c0 = 0.0   # curvature |b·dx| of the first iteration

  while reason === :running
    # linearised problem: A dx = -R
    rmul!(b, -1)
    t1 = time()
    solve!(dx, ns, b)
    log.t_linear += time() - t1
    if !linear_converged(ns)
      reason = :linear_failed
      break
    end

    # curvature criterion, relative to the first iteration (only computed when enabled)
    if tols.ctol > 0
      c = abs(dot(b, dx))
      if log.num_iters == 0
        c0 = c
      elseif c < tols.ctol * c0
        reason = :converged_ctol
        break
      end
    end

    # back to the consistent tangent once the residual has dropped enough in projected mode
    next = _next_mode(tangent, mode, r, r_sw)
    if next !== mode
      _print_switch(log, next, r)
      mode = next
    end

    t1 = time()
    α = search!(linesearch, w, op, x, dx, b)    # takes the step: x ← x + α dx, b = R(x); α = 0 is a failure
    log.t_linesearch += time() - t1
    r = norm(b)
    _record!(log, r, α, mode)
    if log.verbose >= 2
      _print_row(log, r, r0, α, α == injectivity_limit(linesearch), mode, lname * linear_info(ns), time() - t0)
      _print_detail(log, linesearch)
    end

    if α == 0
      planb = _on_failure(tangent, mode)
      if planb === nothing
        reason = :linesearch_failed
        break
      end
      mode = planb                    # switch to the projected tangent at the same state
      r_sw = r
      log.nswitches += 1
      _print_switch(log, mode, r)
    else
      reason = _check(tangent, tols, mode, r, r0)
    end
    if reason === :running && log.num_iters >= tols.maxiter
      reason = :maxiter
    end
    if reason === :running
      set_tangent!(op, mode)
      t1 = time()
      jacobian!(A, op, x)
      t2 = time()
      numerical_setup_state!(ns, A, x)
      log.t_assembly += t2 - t1
      log.t_setup += time() - t2
    end
  end

  mode === :projected && set_tangent!(op, :consistent)   # the model is left with the consistent tangent
  log.reason = reason
  log.t_solve += time() - t0
  _print_end(log)
  return x
end
