# Time integration

Transient problems are solved by `TransientModel` on the same incremental loop as the static
driver: one attempt per time step with `step!`, bisection when a solve fails, growth or
error-controlled adaptation of the step, internal variables committed when the step is accepted,
and the Newton–Raphson solver, line searches and backends of `StaticNonlinearModel`. This page states
the contract the time schemes follow and how a transient problem is written.

## The stage contract

A time step is a sequence of **stages**. Every stage is a (non)linear problem in a *stage unknown*
`y` (free values). The weak forms do not know the scheme: they receive a context `ctx` and write the
field, its time derivatives and its evaluation points through four functions, each of them **affine in
`y`**:

```julia
u                    # the field: α₀ y + β₀
∂t(ctx, u)           # first time derivative: α₁ y + β₁
∂tt(ctx, u)          # second time derivative: α₂ y + β₂
at(ctx, u, :mid)     # an evaluation point of the scheme (:n, :mid, :np1, :nf): αₚ y + βₚ
```

`α` are scalars of the stage (runtime parameters of the forms: no recompilation when Δt changes); `β`
are FE functions over the **history vectors** of the model, preallocated once and written in place when
a step is committed. The Jacobian uses the same calls on the trial direction, `∂t(ctx, du) = α₁ du`.
For a multi-field problem `ctx` holds one context per field, so every field may have its own scheme
(`scheme = (u = Newmark(), θ = BDF(2))`).

The contract covers implicit one-step and multistep schemes, explicit schemes (the stage unknown is the
acceleration and the solve is a mass solve), Runge–Kutta stages (the stage unknown is the stage
derivative), energy–momentum schemes (through `at(ctx, u, :n)` and `:mid`; the algorithmic stress is a
constitutive model) and user-defined history objects.

### Schemes

| Scheme | Order | Unknown | Notes |
|:--|:-:|:--|:--|
| `QuasiStatic()` | — | `u_{n+1}` | no time derivative; the history (`ctx.u⁻`) for internal variables |
| `BDF(1)`, `BDF(2)` | 1, 2 | `u_{n+1}` | variable step; BDF2 starts with BDF1 |
| `Theta(θ)` | 1 (2 for θ = ½) | `u_{n+1}` | θ = 1 backward Euler, θ = ½ Crank–Nicolson |
| `GeneralizedAlpha(ρ∞)` | 2 | `u_{n+1}` | Chung–Hulbert; `Newmark(β, γ)`, `HHT(α)` and `Midpoint()` are members |
| `CentralDifference()` | 2 | `a_{n+1}` | explicit; `mass_lumping = true` for a diagonal solve |
| `SDIRK(2)` | 2 | stage derivatives | two implicit stages; embedded error estimate |

`Midpoint()` is `GeneralizedAlpha` with ρ∞ = 1: `u_{n+1} − u_n = Δt (v_n + v_{n+1})/2` and the internal
forces at `u_{n+½}`.

## Writing a transient problem

### General level

The forms take the context. A viscous solid with internal variables, quasi-static in time:

```julia
res(ctx) = (u, v)     -> residual(cons_model, k, u, v, dΩ, ctx.t, F∘∇(ctx.u⁻)', A...)
jac(ctx) = (u, du, v) -> jacobian(cons_model, k, u, du, v, dΩ, ctx.t, F∘∇(ctx.u⁻)', A...)
m = TransientModel(res, jac, Uu, Vu, D_bc; scheme=QuasiStatic(), xh=uh,
                   physics=cons_model, internal=(xh, xh⁻) -> update_state!(cons_model, A, F∘∇(xh)', F∘∇(xh⁻)'; backend),
                   nls, backend)
solve!(m; stepping=TimeStepping(Δt=0.05, t_end=15.0, maxbisec=4), post, ProjectDirichlet=true)
```

Navier–Stokes with stabilisation, where the time derivative also appears inside the SUPG and PSPG
terms, and the problem is linear in the unknown once the advecting velocity is taken from the history
(Oseen):

```julia
function res(ctx)
  ((u, p), (v, q)) -> begin
    w = 2.0 * ctx.u⁻ - 1.0 * ctx.u⁻⁻               # advecting velocity, extrapolated
    Ru = ρ * ∂t(ctx, u) + ρ * (∇(u)' ⋅ w) + ∇(p)
    ∫(ρ * (∂t(ctx, u) ⋅ v))dΩ + … + ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ Ru))dΩ + …
  end
end
m = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), tangent_update=:step, backend)
```

### Structured level: the mass form

For solids the inertia can be moved out of the forms: the user writes the *static* forms `res(t)`,
`jac(t)` as for `StaticNonlinearModel` and gives the bilinear **mass form**; the model assembles the mass
matrix once (or once per step with `mass_update = :step`) and adds the inertia to the residual and the
tangent algebraically. This is the fastest path and the one that gives the Armijo line search and the
hybrid tangent their incremental energy automatically.

```julia
mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ          # or mass_form(cons_model, dΩ) from the density of the model
m = TransientModel(res, jac, U, V, D_bc; scheme=GeneralizedAlpha(0.9), mass, xh=u0h, vh=v0h,
                   energy, nls, backend)
solve!(m; stepping=AdaptiveStepping(Δt=1e-3, t_end=2.0, rtol=1e-4), post)
```

The mass form of a multi-field problem is written on the tuples and is zero on the fields without
inertia (electric potential, pressure); those fields are evaluated at the same point of the step as the
rest and do not enter the error estimate.

## Tangent update

`tangent_update` tells the stage solve how the tangent depends on the unknown:

- `:iteration` (default): nonlinear problem, the tangent is re-assembled at every Newton iteration;
- `:step`: linear in the unknown with a tangent that depends on the history (Oseen): assembled once per
  attempt and solved once from the zero guess, no Newton iteration and no second residual;
- `:Δt`: linear with a tangent that depends on Δt only (heat conduction): assembled and factorised only
  when Δt changes.

Explicit schemes solve with the mass matrix only.

## Step control

`TimeStepping` is the analogue of `LoadStepping`: fixed Δt, bisection when a solve fails
(`maxbisec` consecutive bisections), growth `Δt ← min(grow Δt, Δt0, Δtmax)` after a step converged in
at most `easy` iterations, exact output times `tstops`.

`AdaptiveStepping` controls Δt with the local error estimate of the scheme (Milne's device for
the first-order schemes, Zienkiewicz–Xie for the Newmark family, the embedded pair for SDIRK): the
weighted RMS error `err` over the free dofs with a time derivative, with `rtol` and `atol`, must be at
most 1 for the step to be accepted; otherwise the step is rejected and retried with the Δt of the
controller (PI by default, I on the first step and after a rejection), within `[Δtmin, Δtmax]` and
`[fmin, fmax]` times the current Δt. A solver failure is not an error estimate: it bisects Δt as in
`TimeStepping`. `AdaptiveStepping` is an error for schemes without an estimate (`QuasiStatic`,
`CentralDifference`).

In both policies a failed or rejected attempt restores the state, the velocities and accelerations, the
time, the Dirichlet values and the internal variables of the last accepted step.

## Internal variables and Δt

`physics = cons_model` makes the driver call `update_time_step!(cons_model, Δt)` before every attempt
with the real Δt (also after a bisection or a rejection). `internal = (xh, xh⁻) -> …` is called when a
step is accepted, with the new state and the state of the previous step, and is where the user updates
the internal variables (`update_state!`, with `CellState` or `KernelState`, through the engine when
`backend = :compiled`). Post-processing (`post`) runs after the internal variables are updated.

## Output

With `verbose = 1` one line per time step: step, time, Δt, Newton iterations, residual, the error estimate
when adaptive, the kinetic energy `K = ½ vᵀ M v`, the stored energy `Π` of the `energy` hook, their sum,
the balance `W_ext − ΔK − ΔΠ` accumulated since the start (numerical and physical dissipation plus the
work of loads not given through `neumann`), convergence and time. With `verbose = 2` the header of every
step and the energies of every Newton iterate. The final summary adds the balance and, when adaptive, the
accepted, rejected and bisected steps and the range of Δt.
