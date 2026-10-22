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

Algebraic operator used by the drivers. `Val(:gridap)` (default) builds Gridap's `FEOperator`;
`Val(:kernel)` is provided by the HyperFEMKernels extension (`using HyperFEMKernels`).
`previous` is the operator of the previous load step (or `nothing`), so that a backend can reuse
its structure.
"""
build_operator(::Val{:gridap}, res, jac, U, V, assem, previous) =
    get_algebraic_operator(FEOperator(res, jac, U, V, assem))
build_operator(::Val{B}, res, jac, U, V, assem, previous) where {B} =
    error("HyperFEM: backend :$B is not available; for :kernel load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    dirichlet_projection(backend, res, jac, uh, ∆U, V, ls, previous) -> (duh, op)

Linearised problem of the Dirichlet projection (`solve!` with `ProjectDirichlet=true`): solves
`jac(uh, du, v) = -res(uh, v)` for `du` in `∆U`, whose Dirichlet values hold the increment of the
conditions of the step, with the linear solver `ls`. Returns the increment `duh` (an `FEFunction` of
`∆U`) and the operator used, handed back as `previous` in the next step so that a backend can reuse
its structure (`nothing` the first time). `Val(:gridap)` assembles an `AffineFEOperator`;
`Val(:kernel)` is provided by the HyperFEMKernels extension.
"""
function dirichlet_projection(::Val{:gridap}, res, jac, uh, ∆U, V, ls, previous)
    l(v) = -1.0 * res(uh, v)
    a(du, v) = jac(uh, du, v)
    op = AffineFEOperator(a, l, ∆U, V)
    solve(ls, op), nothing
end
dirichlet_projection(::Val{B}, res, jac, uh, ∆U, V, ls, previous) where {B} =
    error("HyperFEM: backend :$B is not available; for :kernel load HyperFEMKernels (`using HyperFEMKernels`)")

"""
    StaticNonlinearModel(res, jac, U, V, dirichlet; neumann=nothing, nls, xh, xh⁻, backend=:gridap)

`res(τ)` and `jac(τ)` are functions of the stage parameter τ of the solver. `dirichlet` are the
`BoundaryConditions` of `U`; `neumann`, if any, the [`NeumannConditions`](@ref) added to the residual by
the driver at the load of every step. The model owns the Dirichlet applicators of its spaces
(`get_applicators`: those of `U`, `∆U` and the space of `xh⁻`). `backend` selects the algebraic operator
(see [`build_operator`](@ref)).
"""
struct StaticNonlinearModel{A,B,C,D,E,F,G} <: ComputationalModel
    res::A
    jac::B
    spaces::C
    dirichlet::D
    neumann::E
    applicators::F
    caches::G

    function StaticNonlinearModel(
        res::Function, jac::Function, U, V, dirbc;
        neumann=nothing,
        assem_U=SparseMatrixAssembler(U, V),
        nls::NonlinearSolver=NewtonSolver(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
        xh::FEFunction=FEFunction(U, zero_free_values(U)),
        xh⁻::FEFunction=FEFunction(U, zero_free_values(U)),
        backend::Symbol=:gridap)

        ∆U = TrialFESpace(U, dirbc, 0.0)
        Un = get_fe_space(xh⁻)
        app_U = dirichlet_applicator(U, dirbc)
        applicators = (app_U, dirichlet_applicator(∆U, dirbc), Un === U ? app_U : dirichlet_applicator(Un, dirbc))
        TrialFESpace!(U, app_U, 0.0)
        spaces = (U, V, ∆U, Un)
        x = get_free_dof_values(xh)
        x⁻ = get_free_dof_values(xh⁻)
        # x⁻ = zero_free_values(U)
        _res = with_neumann(res(1.0), neumann, 1.0)
        _jac = jac(1.0)
        op = build_operator(Val(backend), _res, _jac, U, V, assem_U, nothing)
        nls_cache = instantiate_caches(x, nls, op)
        caches = (nls, nls_cache, x, x⁻, assem_U, Val(backend), Ref{Any}(op), Ref{Any}(nothing))

        A, B, C = typeof(res), typeof(jac), typeof(spaces)
        D, E, F, G = typeof(dirbc), typeof(neumann), typeof(applicators), typeof(caches)
        return new{A,B,C,D,E,F,G}(res, jac, spaces, dirbc, neumann, applicators, caches)
    end
end

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
        Algebra.solve!(x, nls, op, nls_cache)
        r_abs = nls.log.residuals[nls.log.num_iters+1]
        r_rel = r_abs / nls.log.residuals[1]
        if !converged(nls.log.tols, nls.log.num_iters, r_abs, r_rel)
            @warn "Bisection performed!"
            x .= x⁻
            Λ -= α * ∆Λ
            ∆Λ = ∆Λ / 2
            nbisect += 1
            # @assert(nbisect <= stepping[:maxbisec], "Maximum number of bisections reached")
        else
            print_message(nls.log, "\nSTEP: $Λ_, Λ: $Λ\n")
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

function dirichlet_preconditioning!(x::Vector{Float64}, m::StaticNonlinearModel, Λ::Float64, ∆Λ::Float64, ::Newton_RaphsonSolver, load=identity)
    duh = get_dirichlet_preconditioner(m, Λ, ∆Λ, load)
    # update step
    α = update_cellstate!(m.caches[1].linesearch, get_state(m), duh)
    x .+= α * get_free_dof_values(duh)
    return α
end

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
        nls::NonlinearSolver=NewtonSolver(LUSolver(); maxiter=10, rtol=1.e-8, verbose=true),
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
        print_message(nls.log, "\nStep: $itime, Time: $t\n")
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
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V)
        op = AffineFEOperator(bf, lf, U, V, assem_U)
        K, b = get_matrix(op), get_vector(op)
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K)
        caches = (ns, K, b, x, assem_U)


        A, B, C = typeof(lf), typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(lf, bf, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        lf::Vector{<:Function}, bf::Function, U0, V0, dirbc;
        assem_U0=SparseMatrixAssembler(U0, V0),
        ls::LinearSolver=LUSolver())

        nblocks = length(lf)
        U, V = repeat_spaces(nblocks, U0, V0)
        spaces = (U, V)
        ##  cache
        K = assemble_matrix(bf, assem_U0, U0, V0) # 1D
        b = allocate_in_range(K)
        fill!(b, zero(eltype(b))) # 1D
        x = repeated_allocate_in_domain(nblocks, K)
        fill!(x, zero(eltype(x))) # nD
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U0)

        A, B, C = Vector{<:Function}, typeof(bf), typeof(spaces)
        D, E = typeof(dirbc), typeof(caches)
        return new{A,B,C,D,E}(lf, bf, spaces, dirbc, caches)
    end



    function StaticLinearModel(
        bf::Function, U, V, dirbc;
        assem_U=SparseMatrixAssembler(U, V),
        ls::LinearSolver=LUSolver(),
        xh::FEFunction=FEFunction(U, zero_free_values(U)))

        spaces = (U, V)
        K = assemble_matrix(bf, assem_U, U, V)
        b = allocate_in_range(K)
        fill!(b, zero(eltype(b)))
        x = get_free_dof_values(xh)
        # x = allocate_in_domain(K)
        fill!(x, zero(eltype(x)))
        ns = numerical_setup(symbolic_setup(ls, K), K) # 1D
        caches = (ns, K, b, x, assem_U)

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
            assemble_matrix!(bf, K, assem_U, U, V)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x, ns, b)
        return x
    end

    function (m::StaticLinearModel)(xh::SingleFieldFEFunction; Measure=nothing, Assembly=false, kwargs...)
        x_ = get_free_dof_values(xh)
        _, V = m.spaces
        bf = m.bf
        ns, K, b, _, assem_U = m.caches
        assemble_vector!((v) -> ∫(xh * v)Measure, b, assem_U, V)
        if Assembly
            assemble_matrix!(bf, K, assem_U, U, V)
        end
        numerical_setup!(ns, K)
        Algebra.solve!(x_, ns, b)
        return x_
    end



end


function solve!(m::StaticLinearModel; Assembly=true, post=PostProcessor())

    reset!(post)
    U, V = m.spaces
    bf = m.bf
    lf = m.lf
    ns, K, b, x, assem_U = m.caches
    if Assembly
        u = get_trial_fe_basis(U)
        v = get_fe_basis(V)
        uhd = zero(U)
        matcontribs = bf(u, v)
        veccontribs = lf(v)
        data = collect_cell_matrix_and_vector(U, V, matcontribs, veccontribs, uhd)
        assemble_matrix_and_vector!(K, b, assem_U, data)
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
        assemble_matrix!(bf, K, assem_U, U, V)
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
    ns, K, b, x, assem_U0 = m.caches

    if Assembly
        assemble_matrix!(bf, K, assem_U0, U0, V0)
    end

    numerical_setup!(ns, K)

    map(blocks(x), lf) do xi, li
        assemble_vector!(li, b, assem_U0, V0)
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


