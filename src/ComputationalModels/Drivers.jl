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

# Convergence and step messages of the nonlinear solver of a model. HyperFEM's `NewtonRaphson` answers
# through its log; other solvers add their own methods (GridapSolvers' `NewtonSolver` through the
# extension HyperFEMGridapSolversExt, deprecated).
nls_converged(nls::NewtonRaphson) = Solvers.converged(nls)
nls_message(nls::NewtonRaphson, msg) = Solvers.verbosity(nls) >= 1 && print(msg)
nls_converged(nls::NonlinearSolver) = error("HyperFEM: the drivers cannot tell whether $(typeof(nls)) converged; " *
    "use NewtonRaphson or add a method of HyperFEM.ComputationalModels.nls_converged")
nls_message(::NonlinearSolver, msg) = nothing


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
            _, flagconv = map((x, y) -> solve!(x; load=stevol, y...), m.compmodels, kargsolve)
            map((s, a) -> TrialFESpace!(get_fe_space(s), a, τ⁺; stage=1.0), m.state⁻, m.applicators⁻)
            map((x, y) -> x .= y, x⁻, x⁺)
        end
    end

    return x⁺, flagconv

end

#*******************************************************************************	
#    					 StaticNonlinearModel
#*******************************************************************************	

"""
    build_operator(backend, res, jac, U, V, assem, previous)

Algebraic operator used by the drivers. `Val(:lazy)` (default) builds Gridap's `FEOperator`;
`Val(:compiled)` is provided by the HyperFEMKernels extension (`using HyperFEMKernels`).
`previous` is the operator of the previous load step (or `nothing`), so that a backend can reuse
its structure.
"""
build_operator(::Val{:lazy}, res, jac, U, V, assem, previous) =
    get_algebraic_operator(FEOperator(res, jac, U, V, assem))
build_operator(::Val{B}, res, jac, U, V, assem, previous) where {B} =
    error("HyperFEM: backend :$B is not available; for :compiled load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    dirichlet_projection(backend, res, jac, uh, ∆U, V, ls, previous) -> (duh, op)

Linearised problem of the Dirichlet projection (`solve!` with `ProjectDirichlet=true`): solves
`jac(uh, du, v) = -res(uh, v)` for `du` in `∆U`, whose Dirichlet values hold the increment of the
conditions of the step, with the linear solver `ls`. Returns the increment `duh` (an `FEFunction` of
`∆U`) and the operator used, handed back as `previous` in the next step so that a backend can reuse
its structure (`nothing` the first time). `Val(:lazy)` assembles an `AffineFEOperator`;
`Val(:compiled)` is provided by the HyperFEMKernels extension.
"""
function dirichlet_projection(::Val{:lazy}, res, jac, uh, ∆U, V, ls, previous)
    l(v) = -1.0 * res(uh, v)
    a(du, v) = jac(uh, du, v)
    op = AffineFEOperator(a, l, ∆U, V)
    solve(ls, op), nothing
end
dirichlet_projection(::Val{B}, res, jac, uh, ∆U, V, ls, previous) where {B} =
    error("HyperFEM: backend :$B is not available; for :compiled load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    StaticNonlinearModel(res, jac, U, V, dirichlet; neumann=nothing, nls, xh, xh⁻, backend=:lazy,
                         injectivity=nothing, energy=nothing)

`res(τ)` and `jac(τ)` are functions of the stage parameter τ of the solver. `dirichlet` are the
`BoundaryConditions` of `U`; `neumann`, if any, the [`NeumannConditions`](@ref) added to the residual by
the driver at the load of every step. The model owns the Dirichlet applicators of its spaces
(`get_applicators`: those of `U`, `∆U` and the space of `xh⁻`). `backend` selects the algebraic operator
(see `build_operator`).

`injectivity` is the measure whose quadrature points must keep `det F > 0` (`dΩ`, or `(dΩ, i)` when
the displacement is the field `i` of a multi-field space); it gives the injectivity step of
[`InjectivityLS`](@ref) and the safe load factor of the Dirichlet projection (see
[`injectivity_step`](@ref)). `energy(τ) = u -> ∫(…)dΩ + …` is the total potential energy of the
model whose first variation is `res(τ)`: the stored energy of every subdomain (`energy(model, km, u, dΩ)`
of `WeakForms`) plus the regularisation terms; the driver adds the work of the Neumann loads. It is
the energy of the Armijo line search (`InjectivityLS(armijo = true)`), for minimisation problems
only. The solver receives the operator of the backend wrapped in a [`ModelOperator`](@ref) with
these hooks.
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
        assem_U=SparseMatrixAssembler(U, V),
        nls::NonlinearSolver=NewtonRaphson(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)),
        xh⁻::FEFunction=FEFunction(U, zero_free_values(U)),
        backend::Symbol=:lazy,
        injectivity=nothing,
        energy=nothing)

        ∆U = TrialFESpace(U, dirbc, 0.0)
        Un = get_fe_space(xh⁻)
        app_U = dirichlet_applicator(U, dirbc)
        applicators = (app_U, dirichlet_applicator(∆U, dirbc), Un === U ? app_U : dirichlet_applicator(Un, dirbc))
        TrialFESpace!(U, app_U, 0.0)
        spaces = (U, V, ∆U, Un)
        x = get_free_dof_values(xh)
        x⁻ = get_free_dof_values(xh⁻)
        hooks = (injectivity=_injectivity_hook(injectivity), energy=energy)
        _res = with_neumann(res(1.0), neumann, 1.0)
        _jac = jac(1.0)
        op = build_operator(Val(backend), _res, _jac, U, V, assem_U, nothing)
        nls_cache = instantiate_caches(x, nls, model_operator(hooks, spaces, neumann, op, 1.0, 1.0))
        caches = (nls, nls_cache, x, x⁻, assem_U, Val(backend), Ref{Any}(op), Ref{Any}(nothing))

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
    ModelOperator(op; max_step=ms, energy=en)
end
model_operator(m::StaticNonlinearModel, op, Λ=1.0, loadΛ=1.0) = model_operator(m.hooks, m.spaces, m.neumann, op, Λ, loadΛ)

# Energy of the model at the free values x (Gridap's lazy evaluation for every backend; the engine's
# scalar functionals are a follow-up of HyperFEMKernels)
_energy_function(e, U) = x -> sum(e(FEFunction(U, x)))

# Getters
get_state(m::StaticNonlinearModel) = FEFunction(get_trial_space(m), m.caches[3])
get_measure(m::StaticNonlinearModel) = m.res.dΩ
get_spaces(m::StaticNonlinearModel) = m.spaces
get_applicators(m::StaticNonlinearModel) = m.applicators
get_assemblers(m::StaticNonlinearModel) = (m.caches[4])


# is_vtk=true,
#   filePath=datadir("sims", "Temp"),
#   vtk::WriteVTK.CollectionFile=paraview_collection(datadir("sims", "Temp") * "/Results", append=false)

"""
    solve!(m::StaticNonlinearModel; stepping, RestartState, ProjectDirichlet, post, load=identity)

Incremental solve with the stage parameter Λ from 0 to 1. The conditions and the Neumann term are
applied at the load `load(Λ)` (`identity` by default; `StaggeredModel` passes the ramp of its step).
"""
function solve!(m::StaticNonlinearModel;
    stepping=(nsteps=20, maxbisec=15), RestartState::Bool=false, ProjectDirichlet::Bool=false,
    post=PostProcessor(), load=identity)

    reset!(post)
    flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
    U, V, ∆U, Un = m.spaces
    app_U, _, app_Un = m.applicators
    TrialFESpace!(U, app_U, load(0.0); stage=0.0)
    nls, nls_cache, x, x⁻, assem_U = m.caches[1:5]

    Λ = 0.0
    ∆Λ = 1.0 / stepping[:nsteps]
    nbisect = 0
    Λ_ = 0
    if RestartState
        x .*= 0.0
    end
    x⁻ .= x
    α = 1.0

    while !isapprox(Λ, 1.0; atol=1e-12)
        Λ += ∆Λ
        if Λ > 1.0
            ∆Λ = 1.0 - (Λ - ∆Λ)
            Λ = min(1.0, Λ)
        end
        if ProjectDirichlet
            α = dirichlet_preconditioning!(x, m, Λ, ∆Λ, nls, load)
        end
        #@show α
        Λ -= ∆Λ
        ∆Λ = α * ∆Λ
        TrialFESpace!(Un, app_Un, load(Λ); stage=Λ)
        Λ += ∆Λ

        # U.dirichlet_values .+= α*∆U.dirichlet_values

        TrialFESpace!(U, app_U, load(Λ); stage=Λ)

        res = with_neumann(m.res(Λ), m.neumann, load(Λ))
        jac = m.jac(Λ)
        op = build_operator(m.caches[6], res, jac, U, V, assem_U, m.caches[7][])
        m.caches[7][] = op
        Algebra.solve!(x, nls, model_operator(m, op, Λ, load(Λ)), nls_cache)
        if !nls_converged(nls)
            @warn "Bisection performed!"
            # back to the last converged step: state, load factor (∆Λ already includes α) and the Dirichlet
            # values of U, so that the next attempt (and its Dirichlet projection) starts from a consistent state
            x .= x⁻
            Λ -= ∆Λ
            ∆Λ = ∆Λ / 2
            TrialFESpace!(U, app_U, load(Λ); stage=Λ)
            nbisect += 1
        else
            nls_message(nls, "\nSTEP: $Λ_, Λ: $Λ\n")
            Λ_ += 1
            # Write to PVD
            post(Λ)
            x⁻ .= x
        end
        #  GC.gc()

        if nbisect > stepping[:maxbisec]
            @warn "Maximum number of bisections reached"
            flagconv = 0
            break
        end

    end

    vtk_save(post)

    return x, flagconv
end

function post_solve!(pvd, x, Λ, Λ_, m, filePath)

    Ω = get_triangulation(get_spaces(m)[1])
    xh = FEFunction(get_trial_space(m), x)

    Λstring = replace(string(round(Λ, digits=2)), "." => "_")
    pvd[Λ_] = createvtk(Ω,
        filePath * "/_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["u" => xh],
    )

    return pvd
end

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


function get_dirichlet_preconditioner(m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64, load=identity)
    _, V, ∆U = m.spaces
    uh = get_state(m)
    # increment of the Dirichlet values between the stages Λ - ∆Λ and Λ (loads load(Λ - ∆Λ) and load(Λ))
    TrialFESpace!(∆U, m.applicators[2], load(Λ), load(Λ) - load(Λ - ∆Λ); stage=(Λ, ∆Λ))
    res = with_neumann(m.res(Λ - ∆Λ), m.neumann, load(Λ - ∆Λ))
    jac = m.jac(Λ - ∆Λ)
    duh, op = dirichlet_projection(m.caches[6], res, jac, uh, ∆U, V, m.caches[1].ls, m.caches[8][])
    m.caches[8][] = op
    return duh
end
#*******************************************************************************	
#    					 DynamicNonlinearModel
#*******************************************************************************	


struct DynamicNonlinearModel{A,B,C,D,E,F,G} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    velocity::E
    applicators::F                    # of U and Uold
    caches::G

    function DynamicNonlinearModel(
        res::Function, jac::Function, U, Uold, V, dirbc, velocity::TimedependentCondition;
        assem_U=SparseMatrixAssembler(U, V),
        nls::NonlinearSolver=NewtonRaphson(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V, Uold)
        applicators = (dirichlet_applicator(U, dirbc), dirichlet_applicator(Uold, dirbc))
        x = get_free_dof_values(xh)
        x⁻ = zero_free_values(Uold)
        uh⁻ = FEFunction(Uold, x⁻)
        vuh = velocity()
        _res = res(1.0, uh⁻, vuh)
        _jac = jac(1.0, uh⁻, vuh)
        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        nls_cache = instantiate_caches(x, nls, op)
        caches = (nls, nls_cache, x, x⁻, assem_U, uh⁻)
        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E, F, G = typeof(dirbc), typeof(velocity), typeof(applicators), typeof(caches)
        return new{A,B,C,D,E,F,G}(res, jac, spaces, dirbc, velocity, applicators, caches)
    end
end

# Getters
get_state(m::DynamicNonlinearModel) = FEFunction(get_trial_space(m), m.caches[3])
get_measure(m::DynamicNonlinearModel) = m.res.dΩ
get_spaces(m::DynamicNonlinearModel) = m.spaces
get_applicators(m::DynamicNonlinearModel) = m.applicators
get_assemblers(m::DynamicNonlinearModel) = (m.caches[4])


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
    nls, nls_cache, x, x⁻, assem_U, uh⁻ = m.caches
    itime = 0
    KE = zeros(Float64, nsteps)
    EE = zeros(Float64, nsteps)

    uh = FEFunction(U, x)

    for itime in 1:nsteps
        t += Δt
        vuh = m.velocity()
        _res = m.res(t, uh⁻, vuh)
        _jac = m.jac(t, uh⁻, vuh)
        TrialFESpace!(U, m.applicators[1], t)
        TrialFESpace!(U⁻, m.applicators[2], t - Δt)

        # @show(    ∑( ∫(uh⁻[1]⋅uh⁻[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uh⁻[2]⋅uh⁻[2])m.res.dΩ   ) )


        op = get_algebraic_operator(FEOperator(_res, _jac, U, V, assem_U))
        Algebra.solve!(x, nls, op, nls_cache)

        # @show(    ∑( ∫(uu[1]⋅uu[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uu[2]⋅uu[2])m.res.dΩ   ) )
        # @show(    ∑( ∫(uh⁻[1]⋅uh⁻[1])m.res.dΩ  )  )
        # @show(    ∑( ∫(uh⁻[2]⋅uh⁻[2])m.res.dΩ   ) )

        update_velocity!(m.velocity.vh, x, x⁻, Δt)
        x⁻ .= x
        nls_message(nls, "\nStep: $itime, Time: $t\n")
        post(t)


        # # Kinetic Energy
        #  KE[itime] = ∑(∫(0.5  * (m.velocity.vh ⋅ m.velocity.vh))m.res.dΩ )
        # # Elastic Energy
        #  EE[itime] = ∑(∫(Ψ ∘ (∇(uh[1])',∇(uh[2])'))m.res.dΩ )
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
`load_vector!` so that a backend can reuse its structure. `Val(:compiled)` is provided by
the HyperFEMKernels extension (`using HyperFEMKernels`).
"""
function linear_system(::Val{:lazy}, bf, lf, U, V, assem)
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
linear_system(::Val{B}, bf, lf, U, V, assem) where {B} =
    error("HyperFEM: backend :$B is not available; for :compiled load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    linear_system!(backend, K, b, bf, lf, U, V, assem, op) -> op

Re-assembles in place the system of `linear_system`: `K` and `b` with the current Dirichlet
values of `U`, or only `K` when `lf === nothing`. Returns the operator to hand back in the next call.
"""
function linear_system!(::Val{:lazy}, K, b, bf, lf, U, V, assem, op)
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
linear_system!(::Val{B}, K, b, bf, lf, U, V, assem, op) where {B} =
    error("HyperFEM: backend :$B is not available; for :compiled load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    load_vector!(backend, b, lf, U, V, assem, op) -> op

Vector of the linear form `lf` alone (no Dirichlet lift), as Gridap's `assemble_vector!`; used for
the right-hand sides of a [`StaticLinearModel`](@ref) with several linear forms. Returns the operator
to hand back in the next call.
"""
function load_vector!(::Val{:lazy}, b, lf, U, V, assem, op)
    assemble_vector!(lf, b, assem, V)
    nothing
end
load_vector!(::Val{B}, b, lf, U, V, assem, op) where {B} =
    error("HyperFEM: backend :$B is not available; for :compiled load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    StaticLinearModel(lf, bf, U, V, dirichlet; assem_U, ls, xh, backend=:lazy)
    StaticLinearModel(bf, U, V, dirichlet; assem_U, ls, xh, backend=:lazy)
    StaticLinearModel(lfs::Vector, bf, U0, V0, dirichlet; assem_U0, ls, backend=:lazy)

Linear problem `bf(u, v) = lf(v)`. The first form assembles matrix and vector at construction; the
second only the matrix (the right-hand side is given to `solve!` or built by the model's functors);
the third solves one matrix for several right-hand sides `lfs` on the repeated spaces of `U0`, `V0`.
`backend` selects who assembles (see `linear_system`): `:lazy` (default) or `:compiled`
(HyperFEMKernels extension, `using HyperFEMKernels`).
"""
struct StaticLinearModel{A,B,C,D,E} <: ComputationalModel
    lf::A
    bf::B
    spaces::C
    dirichlet::D
    caches::E

    function StaticLinearModel(
        lf::Function, bf::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        ls::LinearSolver=LUSolver(),
        xh::FEFunction=FEFunction(U, zero_free_values(U)),
        backend::Symbol=:lazy)

        spaces = (U, V)
        K, b, op = linear_system(Val(backend), bf, lf, U, V, assem_U)
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K)
        caches = (ns, K, b, x, assem_U, Val(backend), Ref{Any}(op))

        A, B, C = typeof(lf), typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(lf, bf, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        lf::Vector{<:Function}, bf::Function, U0, V0, dirbc;
        assem_U0=SparseMatrixAssembler(U0, V0),
        ls::LinearSolver=LUSolver(),
        backend::Symbol=:lazy)

        nblocks = length(lf)
        U, V = repeat_spaces(nblocks, U0, V0)
        spaces = (U, V)
        ##  cache
        K, b, op = linear_system(Val(backend), bf, nothing, U0, V0, assem_U0) # 1D
        x = repeated_allocate_in_domain(nblocks, K)
        fill!(x, zero(eltype(x))) # nD
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U0, Val(backend), Ref{Any}(op))

        A, B, C = Vector{<:Function}, typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(lf, bf, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        bf::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        ls::LinearSolver=LUSolver(),
        xh::FEFunction=FEFunction(U, zero_free_values(U)),
        backend::Symbol=:lazy)

        spaces = (U, V)
        K, b, op = linear_system(Val(backend), bf, nothing, U, V, assem_U)
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U, Val(backend), Ref{Any}(op))

        A, B, C = Nothing, typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(nothing, bf, spaces, dirbc, caches)
    end



    function (m::StaticLinearModel)(x::AbstractVector{Float64}; Assembly=false)
        U, V = m.spaces
        bf = m.bf
        ns, K, b, _, assem_U = m.caches
        b .= x
        if Assembly
            assemble_stiffness!(m, bf, U, V, assem_U)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x, ns, b)
        return x
    end

    function (m::StaticLinearModel)(xh::SingleFieldFEFunction; Measure=nothing, Assembly=false, kwargs...)
        x_ = get_free_dof_values(xh)
        U, V = m.spaces
        bf = m.bf
        ns, K, b, _, assem_U = m.caches
        assemble_vector!((v) -> ∫(xh * v)Measure, b, assem_U, V)
        if Assembly
            assemble_stiffness!(m, bf, U, V, assem_U)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x_, ns, b)
        return x_
    end



end

# Backend of the model and its operator (`caches[6]`, `caches[7]`)
get_backend(m::StaticLinearModel) = m.caches[6]

# Matrix only, with the backend of the model
function assemble_stiffness!(m::StaticLinearModel, bf, U, V, assem)
    op = m.caches[7]
    op[] = linear_system!(get_backend(m), m.caches[2], m.caches[3], bf, nothing, U, V, assem, op[])
end


function solve!(m::StaticLinearModel; Assembly=true, post=PostProcessor())

    reset!(post)
    U, V = m.spaces
    bf = m.bf
    lf = m.lf
    ns, K, b, x, assem_U, backend, op = m.caches
    if Assembly
        op[] = linear_system!(backend, K, b, bf, lf, U, V, assem_U, op[])
    end
    numerical_setup!(ns, K)
    Algebra.solve!(x, ns, b)
    post(1.0)
    vtk_save(post)

    return x
end



function solve!(m::StaticLinearModel, b::Vector{Float64}; Assembly=true, post=PostProcessor())

    reset!(post)
    U, V = m.spaces
    bf = m.bf
    ns, K, _, x, assem_U = m.caches
    if Assembly
        assemble_stiffness!(m, bf, U, V, assem_U)
    end
    numerical_setup!(ns, K)
    Algebra.solve!(x, ns, b)
    post(1.0)
    vtk_save(post)
    return x
end



function solve!(m::StaticLinearModel{Vector{<:Function},<:Any,<:Any,<:Any,<:Any}; Assembly=true, post=PostProcessor())
    reset!(post)
    U, V = m.spaces
    U0 = U[1]
    V0 = V[1]
    bf = m.bf
    lf = m.lf
    ns, K, b, x, assem_U0, backend, op = m.caches

    if Assembly
        assemble_stiffness!(m, bf, U0, V0, assem_U0)
    end

    numerical_setup!(ns, K)

    map(blocks(x), lf) do xi, li
        op[] = load_vector!(backend, b, li, U0, V0, assem_U0, op[])
        Algebra.solve!(xi, ns, b)
    end
    post(1.0)
    vtk_save(post)
    return x
end

# Getters
get_state(m::StaticLinearModel) = FEFunction(get_trial_space(m), m.caches[4])
get_measure(m::StaticLinearModel) = m.biform.dΩ
get_spaces(m::StaticLinearModel) = m.spaces
get_assemblers(m::StaticLinearModel) = (m.caches[5], m.plb_caches[2], m.adj_caches[4])
