#*******************************************************************************
#    					 Incremental loop of the static nonlinear driver
#*******************************************************************************

"""
    LoadStepping(; nsteps=20, maxbisec=15, grow=1.0, easy=10)

Control of the incremental loop of [`solve!`](@ref) for a [`StaticNonlinearModel`](@ref): the stage
parameter Λ runs from 0 to 1 in increments ΔΛ, starting at `ΔΛ0 = 1/nsteps` (an increment may be cut by
the safe load factor of the Dirichlet projection). A failed increment is bisected (ΔΛ/2 from the last
converged state); `maxbisec` is the maximum number of **consecutive** bisections, the count restarts at
every converged increment. After an increment converged in at most `easy` Newton iterations the next one
grows, `ΔΛ ← min(grow ΔΛ, ΔΛ0)`; with `grow = 1` (the default) the increment is never grown again after a
bisection or a cut. A `NamedTuple` `(nsteps = …, maxbisec = …)` is accepted wherever a `LoadStepping` is.
"""
struct LoadStepping
    nsteps::Int
    maxbisec::Int
    grow::Float64
    easy::Int
    function LoadStepping(; nsteps::Int=20, maxbisec::Int=15, grow::Real=1.0, easy::Int=10)
        nsteps >= 1 || error("HyperFEM: LoadStepping needs nsteps ≥ 1 (got $nsteps)")
        maxbisec >= 0 || error("HyperFEM: LoadStepping needs maxbisec ≥ 0 (got $maxbisec)")
        grow >= 1 || error("HyperFEM: LoadStepping needs grow ≥ 1 (got $grow)")
        new(nsteps, maxbisec, Float64(grow), easy)
    end
end

as_stepping(s::LoadStepping) = s
as_stepping(s::NamedTuple) = LoadStepping(; s...)
as_stepping(s) = error("HyperFEM: `stepping` must be a LoadStepping or a NamedTuple (nsteps = …, maxbisec = …); got $(typeof(s))")

"""
    LoadStep

State of the incremental loop: the stage `Λ` of the last converged increment (the target of the attempt
while one is running), the current increment `ΔΛ` and the initial one `ΔΛ0`, the converged increments
`nsteps`, the consecutive bisections `nbisect` and the total ones `nbisect_total`.
"""
mutable struct LoadStep
    Λ::Float64
    ΔΛ::Float64
    ΔΛ0::Float64
    nsteps::Int
    nbisect::Int
    nbisect_total::Int
end
LoadStep(stepping::LoadStepping) = LoadStep(0.0, 1.0 / stepping.nsteps, 1.0 / stepping.nsteps, 0, 0, 0)

finished(st::LoadStep) = isapprox(st.Λ, 1.0; atol=1e-12)

"""
    step!(m::StaticNonlinearModel, st::LoadStep, stepping::LoadStepping;
          ProjectDirichlet, load, post, presolver, on_bisection) -> Bool

One attempt of the incremental loop: the next increment of `st` from the last converged state of `m`
(`presolver(Λ, ΔΛ)` is called first with the stage of the last converged increment and the increment to
try; then the Dirichlet projection when `ProjectDirichlet`, which may cut the increment by its safe load
factor), the Newton solve at the target stage and, on convergence, the update of `st` and of the model
(`post` is called at the new stage; the increment grows according to `stepping`); on failure the state,
the stage and the Dirichlet values of the trial space return to the last converged increment, the increment
is bisected and `on_bisection(Λ, ΔΛ)` is called with that stage and the halved increment. Returns whether
the attempt converged. The hooks are `nothing` (no call) or functions of `(Λ, ΔΛ)`.
"""
function step!(m::StaticNonlinearModel, st::LoadStep, stepping::LoadStepping;
    ProjectDirichlet::Bool=false, load=identity, post=PostProcessor(), presolver=nothing, on_bisection=nothing)

    U, V, ∆U, Un = m.spaces
    app_U, _, app_Un = m.applicators
    c = m.caches
    nls, nls_cache, x, x⁻, assem_U = c.nls, c.nls_cache, c.x, c.x⁻, c.assem
    verbose = nls_verbosity(nls)

    # target stage, cut at Λ = 1
    Λ, ∆Λ = st.Λ, st.ΔΛ
    Λ += ∆Λ
    if Λ > 1.0
        ∆Λ = 1.0 - (Λ - ∆Λ)
        Λ = min(1.0, Λ)
    end
    _hook(presolver, st.Λ, ∆Λ)
    α = ProjectDirichlet ? dirichlet_preconditioning!(x, m, Λ, ∆Λ, nls, load) : 1.0
    # a zero safe load factor would leave the stage where it is forever
    α > 0 || error("HyperFEM: the Dirichlet projection at Λ = $(st.Λ) gives a zero safe load factor (α = $α): the " *
                   "state already violates the injectivity bound of the line search along the Dirichlet increment")
    Λ -= ∆Λ
    ∆Λ = α * ∆Λ
    Un === U || TrialFESpace!(Un, app_Un, load(Λ); stage=Λ)      # the same space is set at the target stage below
    Λ += ∆Λ
    TrialFESpace!(U, app_U, load(Λ); stage=Λ)

    res = with_neumann(m.res(Λ), m.neumann, load(Λ))
    jac = m.jac(Λ)
    op = build_operator(c.backend, res, jac, U, V, assem_U, c.op)
    c.op = op
    step_header(verbose, st.nsteps + 1, Λ, ∆Λ)
    t_step = time()
    Algebra.solve!(x, nls, model_operator(m, op, Λ, load(Λ)), nls_cache)
    ok = nls_converged(nls)
    # growth of the increment after an easy step (grow = 1: never; shared with the time stepping)
    ∆Λnext = grown_increment(stepping.grow, stepping.easy, ∆Λ, st.ΔΛ0, ok, nls_iterations(nls))
    step_line(verbose, nls, st.nsteps + 1, Λ, ∆Λ, ok, time() - t_step, ∆Λnext > ∆Λ)
    if ok
        st.nsteps += 1
        st.nbisect = 0
        post(Λ)
        x⁻ .= x
        ∆Λ = ∆Λnext
    else
        @warn "Bisection performed!"
        # back to the last converged step: state, stage (∆Λ already includes α) and the Dirichlet values of
        # U, so that the next attempt (and its Dirichlet projection) starts from a consistent state
        x .= x⁻
        Λ -= ∆Λ
        ∆Λ = ∆Λ / 2
        TrialFESpace!(U, app_U, load(Λ); stage=Λ)
        st.nbisect += 1
        st.nbisect_total += 1
        _hook(on_bisection, Λ, ∆Λ)
    end
    st.Λ, st.ΔΛ = Λ, ∆Λ
    return ok
end

_hook(::Nothing, Λ, ΔΛ) = nothing
_hook(f, Λ, ΔΛ) = (f(Λ, ΔΛ); nothing)

"""
    solve!(m::StaticNonlinearModel; stepping, RestartState, ProjectDirichlet, post, load=identity,
           presolver=nothing, on_bisection=nothing) -> (x, flagconv)

Incremental solve with the stage parameter Λ from 0 to 1 by [`step!`](@ref), controlled by `stepping`
(a [`LoadStepping`](@ref) or a `NamedTuple` `(nsteps = …, maxbisec = …)`). The conditions and the
Neumann term are applied at the load `load(Λ)` (`identity` by default; `StaggeredModel` passes the ramp of
its step). `presolver(Λ, ΔΛ)` runs before every attempt and `on_bisection(Λ, ΔΛ)` after every bisection
(see `step!`). `flagconv` is 1 when Λ = 1 was reached and 0 when the consecutive bisections exceeded
`maxbisec`.
"""
function solve!(m::StaticNonlinearModel;
    stepping=LoadStepping(), RestartState::Bool=false, ProjectDirichlet::Bool=false,
    post=PostProcessor(), load=identity, presolver=nothing, on_bisection=nothing)

    stepping = as_stepping(stepping)
    reset!(post)
    flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
    U = m.spaces[1]
    app_U = m.applicators[1]
    TrialFESpace!(U, app_U, load(0.0); stage=0.0)
    c = m.caches
    nls, x, x⁻ = c.nls, c.x, c.x⁻
    verbose = nls_verbosity(nls)
    nls_reset!(nls)
    t_run = time()

    st = LoadStep(stepping)
    RestartState && fill!(x, zero(eltype(x)))     # not `x .*= 0`: a NaN of a diverged run would survive it
    x⁻ .= x

    while !finished(st)
        step!(m, st, stepping; ProjectDirichlet, load, post, presolver, on_bisection)
        if st.nbisect > stepping.maxbisec
            @warn "Maximum number of bisections reached"
            flagconv = 0
            break
        end
    end

    run_summary(verbose, nls, st.Λ, st.nsteps, st.nbisect_total, time() - t_run)
    vtk_save(post)

    return x, flagconv
end
