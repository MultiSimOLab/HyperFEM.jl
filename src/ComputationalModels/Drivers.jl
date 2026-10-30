abstract type ComputationalModel end

get_state(::ComputationalModel) = @abstractmethod
get_measure(::ComputationalModel) = @abstractmethod
get_spaces(::ComputationalModel) = @abstractmethod
"Dirichlet applicators of the spaces a model updates (see `dirichlet_applicator`)."
get_applicators(::ComputationalModel) = @abstractmethod
get_trial_space(m::ComputationalModel) = get_spaces(m)[1]
get_test_space(m::ComputationalModel) = get_spaces(m)[2]

function solve!(::ComputationalModel, ::Float64)
    @abstractmethod
end

# Convergence, verbosity and step messages of the nonlinear solver of a model. HyperFEM's `NewtonRaphson`
# answers through its log; other solvers add their own methods (GridapSolvers' `NewtonSolver` through the
# extension HyperFEMGridapSolversExt, deprecated).
nls_converged(nls::NewtonRaphson) = Solvers.converged(nls)
nls_message(nls::NewtonRaphson, msg) = Solvers.verbosity(nls) >= 1 && print(msg)
nls_verbosity(nls::NewtonRaphson) = Solvers.verbosity(nls)
nls_converged(nls::NonlinearSolver) = error("HyperFEM: the drivers cannot tell whether $(typeof(nls)) converged; " *
    "use NewtonRaphson or add a method of HyperFEM.ComputationalModels.nls_converged")
nls_message(::NonlinearSolver, msg) = nothing
nls_verbosity(::NonlinearSolver) = 0
nls_reset!(nls::NewtonRaphson) = Solvers.reset!(nls.log)
nls_reset!(::NonlinearSolver) = nothing
"Iterations of the last solve (the growth of the increment asks for it; unknown solvers never grow it)."
nls_iterations(nls::NewtonRaphson) = num_iterations(nls)
nls_iterations(::NonlinearSolver) = typemax(Int)

# Output of the incremental drivers: `verbose = 1` one line per attempt and a summary of the run;
# `verbose ≥ 2` a header per step before solving (the solver prints its iterations) and the summary.
function step_header(v::Int, n::Int, Λ, ΔΛ)
    v >= 2 || return
    printstyled(@sprintf("━━ Step %d ━━ Λ = %.4f  ΔΛ = %.4f ━━\n", n, Λ, ΔΛ); bold=true)
end
function step_line(v::Int, nls::NewtonRaphson, n::Int, Λ, ΔΛ, ok::Bool, t, grown::Bool=false)
    v == 1 || return
    r = residual_history(nls)[end]
    np = count(==(:projected), mode_history(nls))
    @printf("step %4d  Λ = %.4f  ΔΛ = %.4f  it = %3d  ‖R‖ = %.3e  ", n, Λ, ΔΛ, num_iterations(nls), r)
    printstyled(ok ? "✓" : "✗ → bisect"; color=ok ? :green : :yellow)
    np > 0 && print("  (projected $np)")
    grown && printstyled("  ↑ ΔΛ"; color=:cyan)
    @printf("  %.2f s\n", t)
end
step_line(v::Int, ::NonlinearSolver, n, Λ, ΔΛ, ok, t, grown=false) = nothing
function run_summary(v::Int, nls::NewtonRaphson, Λ, nsteps::Int, nbisect::Int, t)
    v >= 1 || return
    @printf("Λ = %.4f reached: %d steps, %d bisections, %.2f s\n", Λ, nsteps, nbisect, t)
    print_summary(nls.log)
end
run_summary(v::Int, ::NonlinearSolver, Λ, nsteps, nbisect, t) = nothing


#*******************************************************************************	
#    					 StaggeredModel
#*******************************************************************************	


struct StaggeredModel{A,B,C,D,E} <: ComputationalModel
    compmodels::A
    state⁺::B
    state⁻::C
    caches::D
    applicators⁻::E                   # Dirichlet applicators of the spaces of state⁻

    function StaggeredModel(
        CompModels::Tuple{Vararg{ComputationalModel,N}},
        state⁺::Tuple{Vararg{FEFunction,N}},
        state⁻::Tuple{Vararg{FEFunction,N}}) where N

        x⁺ = map((x) -> get_free_dof_values(x), state⁺)
        x⁻ = map((x) -> get_free_dof_values(x), state⁻)

        caches = (x⁺, x⁻)
        apps⁻ = map((s, m) -> dirichlet_applicator(get_fe_space(s), m.dirichlet), state⁻, CompModels)
        A, B, C, D, E = typeof(CompModels), typeof(state⁺), typeof(state⁻), typeof(caches), typeof(apps⁻)
        return new{A,B,C,D,E}(CompModels, state⁺, state⁻, caches, apps⁻)
    end

end


"""
    solve!(m::StaggeredModel; stepping, presolver, evolτ, verbose, kargsolve) -> (x⁺, flagconv)

Staggered solve of the physics of `m` over `nsteps` steps of the load ramp `evolτ` (each physics with
the keywords of its entry in `kargsolve`). Returns the free values of every physics and `flagconv`,
`1` when every inner solve of every step converged and `0` otherwise (the flag of the inner drivers).
"""
function solve!(m::StaggeredModel;
    stepping=(nsteps=20, nsubsteps=1, maxbisec=15),
    presolver=(τ, ∆τ) -> nothing,
    evolτ=t-> max(min(t, 1.0), 0.0),
    verbose=true,
    kargsolve)

    nsubsteps = stepping[:nsubsteps]
    nsteps = stepping[:nsteps]

    x⁺, x⁻ = m.caches
    map((x) -> TrialFESpace!(x.spaces[1], get_applicators(x)[1], 0.0), m.compmodels)
    map((s, a) -> TrialFESpace!(get_fe_space(s), a, 0.0), m.state⁻, m.applicators⁻)

    flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
    ∆τ = 1.0 / nsteps
    for τ in 0:nsteps-1
        if verbose
            println("*******************************************")
            println("           Staggered Step: $τ              ")
            println("*******************************************")
        end
        presolver(τ, ∆τ)
        τ⁻=evolτ(τ*∆τ)
        τ⁺=evolτ((τ+1)*∆τ)
        # Inside the step the inner solves run their stage parameter from 0 to 1; the load they apply is
        # the ramp of the step (the drivers pass both to the conditions, see `Prescription`)
        stevol(stage) = τ⁻ + (τ⁺ - τ⁻) * stage
        for τ_inner in 1:nsubsteps
            map((x) -> TrialFESpace!(x.spaces[1], get_applicators(x)[1], τ⁺; stage=1.0), m.compmodels)
            flags = map((x, y) -> solve!(x; load=stevol, y...)[2], m.compmodels, kargsolve)
            flagconv = min(flagconv, all(==(1), flags) ? 1 : 0)
            map((s, a) -> TrialFESpace!(get_fe_space(s), a, τ⁺; stage=1.0), m.state⁻, m.applicators⁻)
            map((x, y) -> x .= y, x⁻, x⁺)
        end
    end

    return x⁺, flagconv

end


#*******************************************************************************
#    					 Caches of the drivers
#*******************************************************************************

"""
    NonlinearStepCache(nls, nls_cache, x, x⁻, assem, backend, op)

Work data of the incremental nonlinear drivers, typed so that the hot path of the drivers has no
dynamic dispatch: the nonlinear solver `nls` and its cache, the free values `x` of the state and
`x⁻` of the last converged step, the assembler `assem` of the lazy backend (`nothing` for the
others), the `backend` and its operator `op` of the current step (rebuilt or updated by
[`build_operator`](@ref) at every step, hence a field of a fixed type), and the backend operator of
the Dirichlet projection, `projection` (`nothing` until the first projection and for the lazy
backend; the kernel operator on `∆U` of the compiled one, reused with `update`; its type depends on
the backend, so it is untyped — one dynamic dispatch per projected step). The projection system
itself lives in the tangent, residual and increment of `nls_cache` ([`dirichlet_projection!`](@ref)).
"""
mutable struct NonlinearStepCache{N<:NonlinearSolver,C,X<:AbstractVector,A,B<:AbstractBackend,O}
    const nls::N
    const nls_cache::C
    const x::X
    const x⁻::X
    const assem::A
    const backend::B
    op::O
    projection::Any
end

NonlinearStepCache(nls, nls_cache, x, x⁻, assem, backend, op) =
    NonlinearStepCache{typeof(nls),typeof(nls_cache),typeof(x),typeof(assem),typeof(backend),typeof(op)}(
        nls, nls_cache, x, x⁻, assem, backend, op, nothing)

"""
    LinearSystemCache(ns, K, b, x, assem, backend, op)

Work data of [`StaticLinearModel`](@ref): the numerical setup `ns` of the linear solver, the matrix
`K`, the right-hand side `b`, the solution `x`, the assembler `assem` of the lazy backend, the
`backend` and its operator `op` (`nothing` for the lazy backend; the operator of the last assembly
for the others, whose form changes between the system, the matrix alone and the load vectors, hence
untyped). `stale` records that `K` was re-assembled since the last numerical setup: the solves update
the setup (the factorisation of a direct solver) only then (`numerical_setup!(cache)`); code that
changes `K` by other means sets it.
"""
mutable struct LinearSystemCache{NS,TK,TB,TX,A,B<:AbstractBackend}
    const ns::NS
    const K::TK
    const b::TB
    const x::TX
    const assem::A
    const backend::B
    op::Any
    stale::Bool
end

LinearSystemCache(ns, K, b, x, assem, backend, op) =
    LinearSystemCache{typeof(ns),typeof(K),typeof(b),typeof(x),typeof(assem),typeof(backend)}(ns, K, b, x, assem, backend, op, false)

"Numerical setup of the linear solver for `K`, only if `K` changed since the last one."
function Algebra.numerical_setup!(c::LinearSystemCache)
    if c.stale
        numerical_setup!(c.ns, c.K)
        c.stale = false
    end
    return c.ns
end



"""
    assembler(backend, assem, U, V)

Assembler of the drivers for `backend`: `LazyBackend` assembles with the Gridap assembler `assem`
(`SparseMatrixAssembler(U, V)` when `assem === nothing`, the default of the drivers); the other
backends assemble with their own kernels and accept no assembler (`assem_U` must be left to its
default), failing with a clear error otherwise.
"""
assembler(::LazyBackend, ::Nothing, U, V) = SparseMatrixAssembler(U, V)
assembler(::LazyBackend, assem, U, V) = assem
assembler(::AbstractBackend, ::Nothing, U, V) = nothing
assembler(b::AbstractBackend, assem, U, V) =
    error("HyperFEM: backend :$(Backends.backend_name(b)) assembles with its own kernels and takes no assembler " *
          "(got $(typeof(assem))); custom assemblers are only supported by backend = :lazy")

#*******************************************************************************	
#    					 StaticNonlinearModel
#*******************************************************************************	

"""
    build_operator(backend, res, jac, U, V, assem, previous)

Algebraic operator used by the drivers. `LazyBackend` (default) builds Gridap's `FEOperator`;
`CompiledBackend` is provided by the HyperFEMKernels extension (`using HyperFEMKernels`).
`previous` is the operator of the previous load step (or `nothing`), so that a backend can reuse
its structure.
"""
build_operator(::LazyBackend, res, jac, U, V, assem, previous) =
    get_algebraic_operator(FEOperator(res, jac, U, V, assem))
build_operator(b::AbstractBackend, res, jac, U, V, assem, previous) = Backends.not_available(b, "the drivers")

"""
    dirichlet_projection!(backend, A, b, res, jac, uh, ∆U, V, assem, previous) -> op

Linear system of the Dirichlet projection (`solve!` with `ProjectDirichlet=true`), assembled into
the tangent `A` and the residual `b` of the Newton cache: `A du = b` is `jac(uh, du, v) = -res(uh, v)`
for `du` in `∆U`, whose Dirichlet values hold the increment of the conditions of the step and are
lifted to `b`. `∆U` shares the free dofs of the trial space of the model, so `A` has the pattern of
the tangent (the backend must assemble into it; the system is then solved with the numerical setup
of the cache, no second matrix nor factorisation structure per step). Returns the operator used,
handed back as `previous` in the next step so that a backend can reuse its structure (`nothing` the
first time). `LazyBackend` assembles with the Gridap assembler `assem` of the model;
`CompiledBackend` is provided by the HyperFEMKernels extension.
"""
function dirichlet_projection!(::LazyBackend, A, b, res, jac, uh, ∆U, V, assem, previous)
    l(v) = -1.0 * res(uh, v)
    a(du, v) = jac(uh, du, v)
    du = get_trial_fe_basis(∆U)
    v = get_fe_basis(V)
    data = collect_cell_matrix_and_vector(∆U, V, a(du, v), l(v), zero(∆U))
    assemble_matrix_and_vector!(A, b, assem, data)
    nothing
end
dirichlet_projection!(bk::AbstractBackend, A, b, res, jac, uh, ∆U, V, assem, previous) = Backends.not_available(bk, "the Dirichlet projection")

# Tangent, residual and increment of a Newton cache for the projection, and its numerical setup
# (HyperFEM's `NewtonCache`; Gridap's `NewtonRaphsonCache` and GridapSolvers' cache have the same fields)
projection_system(c) = (c.A, c.b, c.dx)
projection_setup!(c::NewtonCache, x) = numerical_setup!(c, x)
projection_setup!(c, x) = numerical_setup!(c.ns, c.A)

"""
    StaticNonlinearModel(res, jac, U, V, dirichlet; neumann=nothing, nls, xh, xh⁻, backend=:lazy,
                         injectivity=nothing, energy=nothing, tangent_switch=nothing)

`res(τ)` and `jac(τ)` are functions of the stage parameter τ of the solver. `dirichlet` are the
`BoundaryConditions` of `U`; `neumann`, if any, the [`NeumannConditions`](@ref) added to the residual by
the driver at the load of every step. The model owns the Dirichlet applicators of its spaces
(`get_applicators`: those of `U`, `∆U` and the space of `xh⁻`). `backend` (`:lazy`, `:compiled` or an
[`AbstractBackend`](@ref)) selects who evaluates and assembles the forms (see `build_operator`);
`assem_U` is a custom Gridap assembler for the lazy backend only (see [`assembler`](@ref)).

`injectivity` is the measure whose quadrature points must keep `det F > 0` (`dΩ`, or `(dΩ, i)` when
the displacement is the field `i` of a multi-field space); it gives the injectivity step of
[`InjectivityLS`](@ref) and the safe load factor of the Dirichlet projection (see
[`injectivity_step`](@ref)). `energy(τ) = u -> ∫(…)dΩ + …` is the total potential energy of the
model whose first variation is `res(τ)`: the stored energy of every subdomain (`energy(model, km, u, dΩ)`
of `WeakForms`) plus the regularisation terms; the driver adds the work of the Neumann loads. It is
the energy of the Armijo line search (`InjectivityLS(armijo = true)`), for minimisation problems
only. `tangent_switch` is the `TangentSwitch` shared by the `HessianRegularization` wrappers of the
models in `jac`, which the hybrid Newton (`HybridTangent`) flips between the consistent and the
projected tangent. The solver receives the operator of the backend wrapped in a
[`ModelOperator`](@ref) with these hooks.
"""
struct StaticNonlinearModel{A,B,C,D,E,F,G,H} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    neumann::E
    applicators::F
    caches::G
    hooks::H

    function StaticNonlinearModel(
        res::Function, jac::Function, U, V, dirbc;
        neumann=nothing,
        assem_U=nothing,
        nls::NonlinearSolver=NewtonRaphson(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)),
        xh⁻::FEFunction=FEFunction(U, zero_free_values(U)),
        backend=:lazy,
        injectivity=nothing,
        energy=nothing,
        tangent_switch=nothing)

        ∆U = TrialFESpace(U, dirbc, 0.0)
        Un = get_fe_space(xh⁻)
        app_U = dirichlet_applicator(U, dirbc)
        applicators = (app_U, dirichlet_applicator(∆U, dirbc), Un === U ? app_U : dirichlet_applicator(Un, dirbc))
        TrialFESpace!(U, app_U, 0.0)
        spaces = (U, V, ∆U, Un)
        x = get_free_dof_values(xh)
        x⁻ = get_free_dof_values(xh⁻)
        hooks = (injectivity=_injectivity_hook(injectivity), energy=energy, tangent_switch=tangent_switch)
        _res = with_neumann(res(1.0), neumann, 1.0)
        _jac = jac(1.0)
        bk = as_backend(backend)
        assem = assembler(bk, assem_U, U, V)
        op = build_operator(bk, _res, _jac, U, V, assem, nothing)
        nls_cache = instantiate_caches(x, nls, model_operator(hooks, spaces, neumann, op, 1.0, 1.0))
        caches = NonlinearStepCache(nls, nls_cache, x, x⁻, assem, bk, op)

        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E, F, G, H = typeof(dirbc), typeof(neumann), typeof(applicators), typeof(caches), typeof(hooks)
        return new{A,B,C,D,E,F,G,H}(res, jac, spaces, dirbc, neumann, applicators, caches, hooks)
    end
end

_injectivity_hook(::Nothing) = nothing
_injectivity_hook(dΩ::Measure) = (dΩ, nothing)
_injectivity_hook(inj::Tuple{Measure,Int}) = inj
_injectivity_hook(inj) = error("HyperFEM: `injectivity` must be a Measure or a tuple (Measure, field index); got $(typeof(inj))")

_field(uh, ::Nothing) = uh
_field(uh, i::Int) = uh[i]

"""
    injectivity_step(m::StaticNonlinearModel, xh, dxh, β, jmin) -> α

Injectivity step of the model (`injectivity = …`) from the state `xh` along the increment `dxh`.
"""
function Solvers.injectivity_step(m::StaticNonlinearModel, xh, dxh, β, jmin)
    inj = m.hooks.injectivity
    inj === nothing && error("HyperFEM: the model has no `injectivity = …` measure for the injectivity step")
    injectivity_step(inj[1], _field(xh, inj[2]), _field(dxh, inj[2]), β, jmin)
end

# The operator handed to the Newton: the operator of the backend with the hooks of the model at the
# stage Λ and the load loadΛ of the step
function model_operator(hooks, spaces, neumann, op, Λ, loadΛ)
    U, V = spaces[1], spaces[2]
    inj = hooks.injectivity
    ms = inj === nothing ? nothing :
        (x, dx, β, jmin) -> injectivity_step(inj[1], _field(FEFunction(U, x), inj[2]), _field(FEFunction(V, dx), inj[2]), β, jmin)
    en = hooks.energy === nothing ? nothing : _energy_function(with_neumann_energy(hooks.energy(Λ), neumann, loadΛ), U)
    ModelOperator(op; max_step=ms, energy=en, tangent_switch=hooks.tangent_switch)
end
model_operator(m::StaticNonlinearModel, op, Λ=1.0, loadΛ=1.0) = model_operator(m.hooks, m.spaces, m.neumann, op, Λ, loadΛ)

# Energy of the model at the free values x (Gridap's lazy evaluation for every backend; the engine's
# scalar functionals are a follow-up of HyperFEMKernels)
_energy_function(e, U) = x -> sum(e(FEFunction(U, x)))

# Getters
get_state(m::StaticNonlinearModel) = FEFunction(get_trial_space(m), m.caches.x)
get_measure(m::StaticNonlinearModel) = m.res.dΩ
get_spaces(m::StaticNonlinearModel) = m.spaces
get_applicators(m::StaticNonlinearModel) = m.applicators
get_assemblers(m::StaticNonlinearModel) = m.caches.assem



function dirichlet_preconditioning!(x::Vector{Float64}, m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64, nls::NewtonRaphson, load=identity)
    duh = get_dirichlet_preconditioner(m, Λ, ∆Λ, load)
    # safe load factor: the injectivity step of the line search along the Dirichlet increment
    α = dirichlet_step(nls.linesearch, m, get_state(m), duh)
    x .+= α .* get_free_dof_values(duh)
    return α
end

# Safe load factor of the Dirichlet projection for each line search
dirichlet_step(ls::Solvers.AbstractLineSearch, m, xh, dxh) = update_cellstate!(ls, xh, dxh)
dirichlet_step(ls::InjectivityLS, m, xh, dxh) = injectivity_step(m, xh, dxh, ls.β, ls.jmin)

function dirichlet_preconditioning!(x::Vector{Float64}, m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64, ::Any, load=identity)
    duh = get_dirichlet_preconditioner(m, Λ, ∆Λ, load)
    x .+= get_free_dof_values(duh)
    return 1.0
end


"""
    get_dirichlet_preconditioner(m, Λ, ∆Λ, load=identity) -> duh

Increment of the Dirichlet projection from the state of `m` at the stage `Λ - ∆Λ` to the stage `Λ`
(`dirichlet_projection!` on the Newton cache of the model). `duh` is an `FEFunction` of `∆U` over the
increment vector of the Newton cache: it is valid until the next Newton solve or projection.
"""
function get_dirichlet_preconditioner(m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64, load=identity)
    _, V, ∆U = m.spaces
    uh = get_state(m)
    # increment of the Dirichlet values between the stages Λ - ∆Λ and Λ (loads load(Λ - ∆Λ) and load(Λ))
    TrialFESpace!(∆U, m.applicators[2], load(Λ), load(Λ) - load(Λ - ∆Λ); stage=(Λ, ∆Λ))
    res = with_neumann(m.res(Λ - ∆Λ), m.neumann, load(Λ - ∆Λ))
    jac = m.jac(Λ - ∆Λ)
    c = m.caches
    A, b, dx = projection_system(c.nls_cache)
    c.projection = dirichlet_projection!(c.backend, A, b, res, jac, uh, ∆U, V, c.assem, c.projection)
    ns = projection_setup!(c.nls_cache, c.x)
    Algebra.solve!(dx, ns, b)
    return FEFunction(∆U, dx)
end
#*******************************************************************************	
#    					 DynamicNonlinearModel
#*******************************************************************************	


"""
    DynamicNonlinearModel(res, jac, U, Uold, V, dirichlet, velocity; assem_U, nls, xh)

Transient driver with the midpoint update of the velocity (`solve!(m; stepping = (nsteps, Δt), post)`).

!!! warning "Deprecated"
    `DynamicNonlinearModel` is deprecated and will be removed in the next release, replaced by new transient
    models built on the incremental loop of `step!`. It is not covered by the test suite: its time loop
    updates the velocity from the free values instead of the FE functions and fails.
"""
struct DynamicNonlinearModel{A,B,C,D,E,F,G,H} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    velocity::E
    applicators::F                    # of U and Uold
    caches::G
    state⁻::H                         # FE function of `Uold` over `caches.x⁻`

    function DynamicNonlinearModel(
        res::Function, jac::Function, U, Uold, V, dirbc, velocity::TimedependentCondition;
        assem_U=nothing,
        nls::NonlinearSolver=NewtonRaphson(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        Base.depwarn("`DynamicNonlinearModel` is deprecated and will be removed in the next release; " *
                     "new transient models replace it", :DynamicNonlinearModel)
        spaces = (U, V, Uold)
        applicators = (dirichlet_applicator(U, dirbc), dirichlet_applicator(Uold, dirbc))
        x = get_free_dof_values(xh)
        x⁻ = zero_free_values(Uold)
        uh⁻ = FEFunction(Uold, x⁻)
        vuh = velocity()
        _res = res(1.0, uh⁻, vuh)
        _jac = jac(1.0, uh⁻, vuh)
        bk = LazyBackend()
        assem = assembler(bk, assem_U, U, V)
        op = build_operator(bk, _res, _jac, U, V, assem, nothing)
        nls_cache = instantiate_caches(x, nls, op)
        caches = NonlinearStepCache(nls, nls_cache, x, x⁻, assem, bk, op)
        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E, F, G, H = typeof(dirbc), typeof(velocity), typeof(applicators), typeof(caches), typeof(uh⁻)
        return new{A,B,C,D,E,F,G,H}(res, jac, spaces, dirbc, velocity, applicators, caches, uh⁻)
    end
end

# Getters
get_state(m::DynamicNonlinearModel) = FEFunction(get_trial_space(m), m.caches.x)
get_measure(m::DynamicNonlinearModel) = m.res.dΩ
get_spaces(m::DynamicNonlinearModel) = m.spaces
get_applicators(m::DynamicNonlinearModel) = m.applicators
get_assemblers(m::DynamicNonlinearModel) = m.caches.assem


function midpoint_update!(v, x⁺, x⁻, Δt)
  v .*= -1.0
  v .-= (2.0 / Δt) * x⁻
  v .+= (2.0 / Δt) * x⁺
end


"""
    update_velocity!

Update the velocity field `vh` based on the current displacement field `xh⁺`
and the previous displacement `xh⁻` using a midpoint time-stepping scheme.
The velocity is updated in place.
"""
function update_velocity!(vh, xh⁺, xh⁻, Δt)
  # TODO: Once moved to Gridap 0.20, a dispatch get_dirichlet_dof_values(::CellField) will be available
  midpoint_update!(get_free_dof_values(vh), get_free_dof_values(xh⁺), get_free_dof_values(xh⁻), Δt)
  midpoint_update!(get_dirichlet_dof_values(get_fe_space(vh)), get_dirichlet_dof_values(get_fe_space(xh⁺)), get_dirichlet_dof_values(get_fe_space(xh⁻)), Δt)
  return vh
end


"""
    update_displacements!

Update the old displacement field `xh⁻` with the new displacement field `xh⁺`.
The update is performed in place, modifying `xh⁻` to match both the free dof values
and the dirichlet dof values of `xh⁺`.
"""
function update_displacements!(xh⁻, xh⁺)
  # TODO: Once moved to Gridap 0.20, a dispatch get_dirichlet_dof_values(::CellField) will be available
  x = get_free_dof_values(xh⁻)
  x .= get_free_dof_values(xh⁺)
  x_dir = get_dirichlet_dof_values(get_fe_space(xh⁻))
  x_dir .= get_dirichlet_dof_values(get_fe_space(xh⁺))
  return xh⁻
end


function solve!(m::DynamicNonlinearModel;
    stepping=(nsteps=10, Δt=0.1),
    post=PostProcessor())

    reset!(post)
    U, V, U⁻ = m.spaces
    t = 0.0
    Δt = stepping[:Δt]
    nsteps = stepping[:nsteps]
    c = m.caches
    nls, nls_cache, x, x⁻, assem_U, uh⁻ = c.nls, c.nls_cache, c.x, c.x⁻, c.assem, m.state⁻

    for itime in 1:nsteps
        t += Δt
        vuh = m.velocity()
        _res = m.res(t, uh⁻, vuh)
        _jac = m.jac(t, uh⁻, vuh)
        TrialFESpace!(U, m.applicators[1], t)
        TrialFESpace!(U⁻, m.applicators[2], t - Δt)

        op = build_operator(c.backend, _res, _jac, U, V, assem_U, c.op)
        c.op = op
        Algebra.solve!(x, nls, op, nls_cache)

        update_velocity!(m.velocity.vh, x, x⁻, Δt)
        x⁻ .= x
        nls_message(nls, "\nStep: $itime, Time: $t\n")
        post(t)
        GC.gc()
    end
    vtk_save(post)

    return x
end





#*******************************************************************************
#    					 StaticLinearModel
#*******************************************************************************

"""
    linear_system(backend, bf, lf, U, V, assem) -> (K, b, op)

Matrix `K` of the bilinear form `bf` and vector `b` of the linear form `lf`, with the Dirichlet lift of
`U` (the Dirichlet values of `U` are moved to the right-hand side), as Gridap's `AffineFEOperator`.
With `lf === nothing` only `K` is assembled and `b` is allocated and zeroed. `op` is the operator
of the backend (`nothing` for `:lazy`), handed back to `linear_system!` and
`load_vector!` so that a backend can reuse its structure. `CompiledBackend` is provided by
the HyperFEMKernels extension (`using HyperFEMKernels`).
"""
function linear_system(::LazyBackend, bf, lf, U, V, assem)
    if lf === nothing
        K = assemble_matrix(bf, assem, U, V)
        b = allocate_in_range(K)
        fill!(b, zero(eltype(b)))
    else
        op = AffineFEOperator(bf, lf, U, V, assem)
        K, b = get_matrix(op), get_vector(op)
    end
    K, b, nothing
end
linear_system(b::AbstractBackend, args...) = Backends.not_available(b, "the linear drivers")

"""
    linear_system!(backend, K, b, bf, lf, U, V, assem, op) -> op

Re-assembles in place the system of `linear_system`: `K` and `b` with the current Dirichlet
values of `U`, or only `K` when `lf === nothing`. Returns the operator to hand back in the next call.
"""
function linear_system!(::LazyBackend, K, b, bf, lf, U, V, assem, op)
    if lf === nothing
        assemble_matrix!(bf, K, assem, U, V)
    else
        u = get_trial_fe_basis(U)
        v = get_fe_basis(V)
        data = collect_cell_matrix_and_vector(U, V, bf(u, v), lf(v), zero(U))
        assemble_matrix_and_vector!(K, b, assem, data)
    end
    nothing
end
linear_system!(b::AbstractBackend, args...) = Backends.not_available(b, "the linear drivers")

"""
    load_vector!(backend, b, lf, U, V, assem, op) -> op

Vector of the linear form `lf` alone (no Dirichlet lift), as Gridap's `assemble_vector!`; used for
the right-hand sides of a [`StaticLinearModel`](@ref) with several linear forms. Returns the operator
to hand back in the next call.
"""
function load_vector!(::LazyBackend, b, lf, U, V, assem, op)
    assemble_vector!(lf, b, assem, V)
    nothing
end
load_vector!(b::AbstractBackend, args...) = Backends.not_available(b, "the linear drivers")

"""
    StaticLinearModel(lf, bf, U, V, dirichlet; assem_U, ls, xh, backend=:lazy)
    StaticLinearModel(bf, U, V, dirichlet; assem_U, ls, xh, backend=:lazy)
    StaticLinearModel(lfs::Vector, bf, U0, V0, dirichlet; assem_U, ls, backend=:lazy)

Linear problem `bf(u, v) = lf(v)`. With a linear form `lf` the matrix and the vector are assembled at
construction; without one (`lf = nothing`) only the matrix (the right-hand side is given to `solve!`
or built by the model's functors); with a vector of linear forms `lfs` one matrix is solved for
several right-hand sides on the repeated spaces of `U0`, `V0`. The matrix is factorised (numerical
setup of `ls`) at construction and again only when it is re-assembled (`Assembly = true`).
`backend` selects who assembles (see `linear_system`): `:lazy` (default), `:compiled`
(HyperFEMKernels extension, `using HyperFEMKernels`) or an [`AbstractBackend`](@ref); `assem_U`
is a custom Gridap assembler for the lazy backend only (see [`assembler`](@ref)).
"""
struct StaticLinearModel{A,B,C,D,E} <: ComputationalModel
    lf::A
    bf::B
    spaces::C
    dirichlet::D
    caches::E

    function StaticLinearModel(
        lf::Union{Nothing,Function,AbstractVector{<:Function}}, bf::Function, U0, V0, dirbc;
        assem_U=nothing,
        ls::LinearSolver=LUSolver(),
        xh::Union{Nothing,FEFunction}=nothing,
        backend=:lazy)

        multi = lf isa AbstractVector
        spaces = multi ? repeat_spaces(length(lf), U0, V0) : (U0, V0)
        bk = as_backend(backend)
        assem = assembler(bk, assem_U, U0, V0)
        K, b, op = linear_system(bk, bf, multi ? nothing : lf, U0, V0, assem)
        x = multi ? repeated_allocate_in_domain(length(lf), K) :
            get_free_dof_values(xh === nothing ? FEFunction(U0, zero_free_values(U0)) : xh)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K)
        caches = LinearSystemCache(ns, K, b, x, assem, bk, op)

        A, B, C = typeof(lf), typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(lf, bf, spaces, dirbc, caches)
    end

    function (m::StaticLinearModel)(x::AbstractVector{Float64}; Assembly=false)
        U, V = m.spaces
        c = m.caches
        c.b .= x
        Assembly && assemble_stiffness!(m, m.bf, U, V, c.assem)
        Algebra.solve!(x, numerical_setup!(c), c.b)
        return x
    end

    # L2 projection of `xh`: the load vector ∫(xh v) dMeasure assembled by the backend of the model (no Dirichlet lift)
    function (m::StaticLinearModel)(xh::SingleFieldFEFunction; Measure=nothing, Assembly=false, kwargs...)
        x_ = get_free_dof_values(xh)
        U, V = m.spaces
        c = m.caches
        Assembly && assemble_stiffness!(m, m.bf, U, V, c.assem)
        c.op = load_vector!(c.backend, c.b, (v) -> ∫(xh * v)Measure, U, V, c.assem, c.op)
        Algebra.solve!(x_, numerical_setup!(c), c.b)
        return x_
    end
end

StaticLinearModel(bf::Function, U, V, dirbc; kwargs...) = StaticLinearModel(nothing, bf, U, V, dirbc; kwargs...)

# Backend of the model
get_backend(m::StaticLinearModel) = m.caches.backend

# Matrix only, with the backend of the model
function assemble_stiffness!(m::StaticLinearModel, bf, U, V, assem)
    c = m.caches
    c.op = linear_system!(c.backend, c.K, c.b, bf, nothing, U, V, assem, c.op)
    c.stale = true
    nothing
end


"""
    solve!(m::StaticLinearModel; Assembly=true, post) -> x
    solve!(m::StaticLinearModel, b; Assembly=true, post) -> x

Solve of the linear problem with the right-hand side of the model (or the vector `b`). With
`Assembly = true` the system is re-assembled with the current Dirichlet values (the matrix alone
when the model has no linear form or `b` is given) and the numerical setup updated; with
`Assembly = false` the matrix and its setup of the last assembly are reused.
"""
function solve!(m::StaticLinearModel; Assembly=true, post=PostProcessor())

    reset!(post)
    U, V = m.spaces
    c = m.caches
    if Assembly
        c.op = linear_system!(c.backend, c.K, c.b, m.bf, m.lf, U, V, c.assem, c.op)
        c.stale = true
    end
    Algebra.solve!(c.x, numerical_setup!(c), c.b)
    post(1.0)
    vtk_save(post)

    return c.x
end



function solve!(m::StaticLinearModel, b::Vector{Float64}; Assembly=true, post=PostProcessor())

    reset!(post)
    U, V = m.spaces
    c = m.caches
    Assembly && assemble_stiffness!(m, m.bf, U, V, c.assem)
    Algebra.solve!(c.x, numerical_setup!(c), b)
    post(1.0)
    vtk_save(post)
    return c.x
end



function solve!(m::StaticLinearModel{<:AbstractVector}; Assembly=true, post=PostProcessor())
    reset!(post)
    U, V = m.spaces
    U0 = U[1]
    V0 = V[1]
    c = m.caches
    Assembly && assemble_stiffness!(m, m.bf, U0, V0, c.assem)
    ns = numerical_setup!(c)
    map(blocks(c.x), m.lf) do xi, li
        c.op = load_vector!(c.backend, c.b, li, U0, V0, c.assem, c.op)
        Algebra.solve!(xi, ns, c.b)
    end
    post(1.0)
    vtk_save(post)
    return c.x
end

# Getters
get_state(m::StaticLinearModel) = FEFunction(get_trial_space(m), m.caches.x)
get_spaces(m::StaticLinearModel) = m.spaces
get_assemblers(m::StaticLinearModel) = m.caches.assem
