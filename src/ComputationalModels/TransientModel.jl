#*******************************************************************************
#    					 TransientModel: the time loop on the stages of a scheme
#*******************************************************************************

"""
    TransientStepOperator

Operator of one stage handed to the solver: the backend operator `inner` of the stage forms plus, on
the structured level (a `mass` form), the inertia added algebraically — the residual
`M (α₂ y + β₂)` (second order) or `M (α₁ y + β₁)` (first order), the damping `C (α₁ y + β₁)`, the
inertia of the moving Dirichlet values `bd` and the tangent `s_K K(field) + c_M M + c_C C`. The field at
which the inner forms are evaluated is `x_eval = α₀ y + b0` (the stage field; `y` itself on the general
level, where the forms carry the contract). `map_M`/`map_C` give the position in `nonzeros(A)` of every
stored entry of `M`/`C` (the compiled backend assembles into its own pattern only, so the mass keeps the
pattern of its form). Allocation-free once built.
"""
mutable struct TransientStepOperator{O<:NonlinearOperator,M,D,X<:AbstractVector} <: NonlinearOperator
    inner::O
    structured::Bool
    M::M                      # mass matrix (structured level) or nothing
    C::D                      # damping matrix or nothing
    const map_M::Vector{Int}
    const map_C::Vector{Int}
    const x_eval::X           # stage field for the inner forms
    const w::X                # scratch: α y + β
    const bd::X               # inertia of the Dirichlet values (structured level)
    use_bd::Bool
    c::StageCoefficients
    const b0::X               # β₀ free values (the history's b0)
    const b1::X
    const b2::X
    use_b0::Bool
    dorder::Int               # 1 or 2: which derivative the mass multiplies
    const zfree::X            # zero free values of the FE function of the Dirichlet derivative
    const ad::Vector{Vector{Float64}}   # Dirichlet values of that function
    opbd::Any
    opbdC::Any
    energy::Any               # functional of the stage (structured: Π + Neumann work at the stage time; general: the user's Φ)
    const w2::X               # scratch of the stage energy
end

function TransientStepOperator(inner, structured::Bool, M, C, h::StageHistory, x::AbstractVector, dorder::Int)
    z() = (v = similar(x); fill!(v, zero(eltype(x))); v)
    ad = [zeros(length(d)) for d in h.gd]
    TransientStepOperator{typeof(inner),typeof(M),typeof(C),typeof(x)}(inner, structured, M, C, Int[], Int[], z(), z(), z(), false,
        StageCoefficients(1.0, 0.0, 0.0, 0.0, 1.0), h.b0, h.b1, h.b2, false, dorder, z(), ad, nothing, nothing, nothing, z())
end

Algebra.allocate_residual(o::TransientStepOperator, x::AbstractVector) = allocate_residual(o.inner, x)
Algebra.allocate_jacobian(o::TransientStepOperator, x::AbstractVector) = allocate_jacobian(o.inner, x)
Algebra.zero_initial_guess(o::TransientStepOperator) = zero_initial_guess(o.inner)
Algebra.residual(o::TransientStepOperator, x::AbstractVector) = (b = allocate_residual(o, x); residual!(b, o, x); b)
Algebra.jacobian(o::TransientStepOperator, x::AbstractVector) = (A = allocate_jacobian(o, x); jacobian!(A, o, x); A)

# the stage field for the inner forms: y on the general level, α₀ y + β₀ on the structured one
function _stage_field!(o::TransientStepOperator, y::AbstractVector)
    o.structured || return y
    c = o.c
    if o.use_b0
        o.x_eval .= c.α0 .* y .+ o.b0
    elseif c.α0 == 1.0
        o.x_eval .= y
    else
        o.x_eval .= c.α0 .* y
    end
    o.x_eval
end

function Algebra.residual!(b::AbstractVector, o::TransientStepOperator, y::AbstractVector)
    xe = _stage_field!(o, y)
    residual!(b, o.inner, xe)
    o.structured || return b
    c = o.c
    if o.dorder == 2
        o.w .= c.α2 .* y .+ o.b2
        mul!(b, o.M, o.w, 1.0, 1.0)
        if o.C !== nothing
            o.w .= c.α1 .* y .+ o.b1
            mul!(b, o.C, o.w, 1.0, 1.0)
        end
    elseif o.dorder == 1
        o.w .= c.α1 .* y .+ o.b1
        mul!(b, o.M, o.w, 1.0, 1.0)
    end
    o.use_bd && (b .+= o.bd)
    b
end

function Algebra.jacobian!(A, o::TransientStepOperator, y::AbstractVector)
    xe = _stage_field!(o, y)
    c = o.c
    if o.structured && c.s_K == 0.0
        fill!(nonzeros(A), zero(eltype(A)))
    else
        jacobian!(A, o.inner, xe)
        o.structured && c.s_K != 1.0 && (nonzeros(A) .*= c.s_K)
    end
    o.structured || return A
    nzA = nonzeros(A)
    if o.dorder == 2
        _add_mapped!(nzA, c.α2, nonzeros(o.M), o.map_M)
        o.C !== nothing && _add_mapped!(nzA, c.α1, nonzeros(o.C), o.map_C)
    elseif o.dorder == 1
        _add_mapped!(nzA, c.α1, nonzeros(o.M), o.map_M)
    end
    A
end
Algebra.jacobian!(A::AbstractMatrix, o::TransientStepOperator, y::AbstractVector) = invoke(jacobian!, Tuple{Any,TransientStepOperator,AbstractVector}, A, o, y)

function Algebra.residual_and_jacobian!(b, A, o::TransientStepOperator, y::AbstractVector)
    residual!(b, o, y)
    jacobian!(A, o, y)
    b, A
end
Algebra.residual_and_jacobian!(b::AbstractVector, A::AbstractMatrix, o::TransientStepOperator, y::AbstractVector) =
    invoke(residual_and_jacobian!, Tuple{Any,Any,TransientStepOperator,AbstractVector}, b, A, o, y)

"""
    stage_energy(o::TransientStepOperator, U, y) -> Float64

Incremental functional of the stage whose gradient in the stage unknown `y` is the stage residual (the
energy of the Armijo line search and of the hybrid tangent). Structured level:
`Φ(y) = yᵀ M (½ α_K y + β_K) + yᵀ C (½ α₁ y + β₁) + yᵀ bd + (Π + W_N)(α₀ y + β₀) / α₀`, with `α_K`, `β_K`
the coefficients of the derivative the mass multiplies, `Π` the model's `energy(t)` and `W_N` the work
of the Neumann loads at the stage time — valid when `r = ∇Π` and `M`, `C` are symmetric (solids); not
defined for an explicit stage (`α₀ = 0`). General level: the user's `energy(ctx)` on the stage FE function.
"""
function stage_energy(o::TransientStepOperator, U, y::AbstractVector)
    e = o.energy
    o.structured || return sum(e(FEFunction(U, y)))
    c = o.c
    c.α0 == 0.0 && error("HyperFEM: the stage energy of an explicit stage is not defined (no line search on explicit stages)")
    Φ = 0.0
    if o.dorder == 2
        o.w .= (0.5 * c.α2) .* y .+ o.b2
        mul!(o.w2, o.M, o.w)
        Φ += dot(y, o.w2)
        if o.C !== nothing
            o.w .= (0.5 * c.α1) .* y .+ o.b1
            mul!(o.w2, o.C, o.w)
            Φ += dot(y, o.w2)
        end
    elseif o.dorder == 1
        o.w .= (0.5 * c.α1) .* y .+ o.b1
        mul!(o.w2, o.M, o.w)
        Φ += dot(y, o.w2)
    end
    o.use_bd && (Φ += dot(y, o.bd))
    xe = _stage_field!(o, y)
    Φ + sum(e(FEFunction(U, xe))) / c.α0
end

function _add_mapped!(nzA::AbstractVector, c::Float64, nzM::AbstractVector, map::Vector{Int})
    c == 0.0 && return
    @inbounds for k in eachindex(map)
        nzA[map[k]] += c * nzM[k]
    end
    nothing
end

"""
    nonzero_map(M, A) -> Vector{Int}

Position in `nonzeros(A)` of every stored entry of `M` (both CSC over the same dofs); an error when an
entry of `M` is outside the pattern of `A` (the mass couples fields the stiffness does not).
"""
function nonzero_map(M::SparseMatrixCSC, A::SparseMatrixCSC)
    size(M) == size(A) || error("HyperFEM: the mass matrix $(size(M)) and the tangent $(size(A)) have different sizes")
    map = Vector{Int}(undef, nnz(M))
    rA, cA = rowvals(A), getcolptr(A)
    rM, cM = rowvals(M), getcolptr(M)
    for j in 1:size(M, 2)
        lo, hi = cA[j], cA[j+1] - 1
        for k in cM[j]:cM[j+1]-1
            i = rM[k]
            p = searchsortedfirst(view(rA, lo:hi), i)
            (p <= hi - lo + 1 && rA[lo+p-1] == i) ||
                error("HyperFEM: the entry ($i, $j) of the mass matrix is outside the sparsity pattern of the tangent; " *
                      "the mass form couples degrees of freedom the stiffness form does not")
            map[k] = lo + p - 1
        end
    end
    map
end

#*******************************************************************************
#    					 The model
#*******************************************************************************

"""
    TransientModel(res, jac, U, V, dirichlet; scheme, mass=nothing, damping=nothing, mass_update=:constant,
                   mass_lumping=false, tangent_update=:iteration, physics=nothing, internal=nothing,
                   history=nothing, xh, vh=nothing, initial_acceleration=:zero, nls, backend=:lazy,
                   neumann=nothing, injectivity=nothing, energy=nothing, tangent_switch=nothing,
                   kinetic=nothing, stage_energy=nothing)

Transient driver on the loop of [`step!`](@ref) with the time `scheme` (`QuasiStatic`, `BDF`, `Theta`,
`GeneralizedAlpha` and its members, `CentralDifference`, `SDIRK`), solved with `TimeStepping` or
`AdaptiveStepping` by [`solve!`](@ref).

Two levels. **General**: `res(ctx)`, `jac(ctx)` take the [`TimeContext`](@ref) of the stage and write
the field, its time derivatives and the history with `field`, `∂t`, `∂tt`, `at` (and `δfield`, `δ∂t`,
`δ∂tt`, `δat` on the trial direction). **Structured** (`mass` given): `res(t)`, `jac(t)` are the static
forms of `StaticNonlinearModel`; the model assembles the mass matrix of the bilinear `mass` form once
(`mass_update = :constant`) or at every attempt (`:step`), lumps it (`mass_lumping`), and adds the inertia
(and the `damping` form on the velocity, second order) to the residual and the tangent of the stage; the
inner forms are evaluated at the stage field (`(1−αf) uₙ₊₁ + αf uₙ` for the generalized-α, `u_θ` for
`Theta`, the known `uₙ₊₁` for `CentralDifference`) and at the stage time, where the `neumann` loads and
the Dirichlet values are taken.

`tangent_update`: `:iteration` (Newton with the consistent tangent), `:step` (linear in the stage
unknown with a tangent that depends on the history — assembled once per attempt and solved once from the
zero guess, no Newton), `:Δt` (linear with a tangent that depends on Δt only — assembled and factorised
only when Δt changes). Explicit stages use `:Δt` (the mass alone).

`physics` receives `update_time_step!(physics, Δt)` before every attempt; `internal(xh, xh⁻)` is called
when a step is accepted, with the new state and the state of the previous step, after `post` (the
post-processing sees the internal variables the residual of the step was evaluated with); `history`
is a user object available as `ctx.history` and committed with `commit!(history, ctx, xh)` when defined.
`xh` is the initial state (and the FE function of the state after every accepted step, over the model's
`x`), `vh` the initial velocity (second order), `initial_acceleration` `:zero` or `:solve` (`M a₀ = −r(0, u₀)`,
structured level, pure mechanics). `injectivity`, `energy` and `tangent_switch` are the hooks of
`StaticNonlinearModel`: `energy(t) = u -> ∫(Ψ…)dΩ` is the stored energy of the model, reported per step
when the solver is verbose ([`EnergyLog`](@ref), with the kinetic energy of the `mass` form — or of the
`kinetic(vh)` functional on the velocity — and the work of the `neumann` loads) and, on the structured
level, turned into the incremental functional of the stage whose gradient is the stage residual
([`stage_energy`](@ref)) for the Armijo line search and the hybrid tangent; on the general level that
functional is the user's `stage_energy(ctx) = y -> …` on the stage FE function.
"""
struct TransientModel{R,J,S<:TimeScheme,SP,D,N,AP,H<:StageHistory,C,HK,MF,PH,IN,UH} <: ComputationalModel
    res::R
    jac::J
    scheme::S
    spaces::SP
    dirichlet::D
    neumann::N
    applicators::AP
    hist::H
    caches::C
    hooks::HK
    mass::MF
    damping::Any
    mass_update::Symbol
    mass_lumping::Bool
    tangent_update::Symbol
    physics::PH
    internal::IN
    history::UH
    initial_acceleration::Symbol
    velocity_given::Bool                  # `vh` was given: its Dirichlet values are the initial Dirichlet velocity

    function TransientModel(res::Function, jac::Function, U, V, dirbc;
        scheme::TimeScheme,
        mass=nothing, damping=nothing, mass_update::Symbol=:constant, mass_lumping::Bool=false,
        tangent_update::Symbol=:iteration,
        physics=nothing, internal=nothing, history=nothing,
        xh::FEFunction=FEFunction(U, zero_free_values(U)), vh=nothing, initial_acceleration::Symbol=:zero,
        nls::NonlinearSolver=NewtonRaphson(LUSolver(); maxiter=10, rtol=1.e-8, verbose=0),
        backend=:lazy, assem_U=nothing, neumann=nothing, injectivity=nothing, energy=nothing, tangent_switch=nothing,
        kinetic=nothing, stage_energy=nothing)

        mass_update in (:constant, :step) || error("HyperFEM: mass_update must be :constant or :step (got :$mass_update)")
        tangent_update in (:iteration, :step, :Δt) || error("HyperFEM: tangent_update must be :iteration, :step or :Δt (got :$tangent_update)")
        initial_acceleration in (:zero, :solve) || error("HyperFEM: initial_acceleration must be :zero or :solve (got :$initial_acceleration)")
        structured = mass !== nothing
        dorder = derivative_order(scheme)
        dorder == 0 && structured && error("HyperFEM: QuasiStatic takes no mass form")
        dorder > 0 && !structured && damping !== nothing && error("HyperFEM: a damping form needs a mass form (structured level)")
        dorder < 2 && damping !== nothing && error("HyperFEM: the damping form is for second-order schemes")
        implicit(scheme, 1) || structured || error("HyperFEM: an explicit scheme needs the mass form (structured level)")
        initial_acceleration === :solve && !structured && error("HyperFEM: initial_acceleration = :solve needs the mass form")

        ∆U = TrialFESpace(U, dirbc, 0.0)
        app_U = dirichlet_applicator(U, dirbc)
        applicators = (app_U, dirichlet_applicator(∆U, dirbc))
        TrialFESpace!(U, app_U, 0.0)
        spaces = (U, V, ∆U)
        x = get_free_dof_values(xh)
        hist = StageHistory(U, x, nstages(scheme))
        hist.xn .= x
        _dcopy!(hist.gdn, _dirichlet_values(U))
        if vh !== nothing
            hist.vn .= get_free_dof_values(vh)
            _dcopy!(hist.vdn, _dirichlet_values(get_fe_space(vh)))
        end
        bk = as_backend(backend)
        assem = assembler(bk, assem_U, U, V)
        # inner operator of the first stage at t = 0 (its structure is reused by every stage)
        c = stage_coefficients!(scheme, hist, 1, 1.0, 1.0)
        ctx = TimeContext(0.0, 1.0, 1, 1, c, hist, history)
        r0, j0 = _stage_forms(res, jac, structured, ctx, neumann, 0.0, identity)
        op = build_operator(bk, r0, j0, U, V, assem, nothing)
        M, C, opM, opC = structured ? _mass_matrices(bk, mass, damping, U, V, assem, mass_lumping) : (nothing, nothing, nothing, nothing)
        y = structured ? similar(x) : x                 # stage unknown: the state itself on the general level
        structured && fill!(y, zero(eltype(y)))
        stepop = TransientStepOperator(op, structured, M, C, hist, x, dorder)
        # the Dirichlet inertia vectors are assembled by the operators of the mass and damping systems (the
        # compiled backend assembles a load vector only through an operator built by `linear_system`)
        stepop.opbd, stepop.opbdC = opM, opC
        kinetic !== nothing && derivative_order(scheme) != 2 && error("HyperFEM: the `kinetic` functional takes the velocity of the history, which the first-order schemes do not carry")
        structured && stage_energy !== nothing && error("HyperFEM: the stage energy is derived from `energy` on the structured level; `stage_energy` is for the general level")
        !structured && energy !== nothing && stage_energy === nothing && @warn "HyperFEM: on the general level the stored energy is reported only; the energy of the line searches needs `stage_energy(ctx)`"
        hooks = (injectivity=_injectivity_hook(injectivity), energy=energy, stage_energy=stage_energy, kinetic=kinetic, tangent_switch=tangent_switch)
        mop = _model_operator(hooks, spaces, stepop, structured)
        nls_cache = instantiate_caches(y, nls, mop)
        if structured
            A = nls_cache.A
            append!(stepop.map_M, nonzero_map(M, A))
            C === nothing || append!(stepop.map_C, nonzero_map(C, A))
        end
        caches = TransientStepCache(nls, nls_cache, x, y, assem, bk, stepop, opM, opC, EnergyLog(x, hist.gd))

        new{typeof(res),typeof(jac),typeof(scheme),typeof(spaces),typeof(dirbc),typeof(neumann),typeof(applicators),
            typeof(hist),typeof(caches),typeof(hooks),typeof(mass),typeof(physics),typeof(internal),typeof(history)}(
            res, jac, scheme, spaces, dirbc, neumann, applicators, hist, caches, hooks, mass, damping, mass_update,
            mass_lumping, tangent_update, physics, internal, history, initial_acceleration, vh !== nothing)
    end
end

"""
    EnergyLog

Energies of the run (tracked when the solver is verbose; `get_energies`): the kinetic energy `K`
(`½ ∫ρ v⋅v`, from the `mass` form on the velocity of the history — second-order schemes — or from the
`kinetic` hook), the stored energy `Π` (the model's `energy(t)` at the state), the work `W` of the
`neumann` loads (trapezoidal rule in time; loads written inside `res` and the work of the Dirichlet
reactions are not tracked) and the balance `W − ΔK − ΔΠ` from the start of the run. `NaN` when not
available.
"""
mutable struct EnergyLog{X<:AbstractVector,D}
    K::Float64
    Π::Float64
    W::Float64
    K0::Float64
    Π0::Float64
    tracked::Bool
    const dx::X                 # increment of the step (free dofs) for the work of the loads
    const dg::D                 # and its Dirichlet values
end
EnergyLog(x::AbstractVector, gd) = EnergyLog{typeof(x),typeof(gd)}(NaN, NaN, 0.0, NaN, NaN, false, similar(x), deepcopy(gd))
balance(e::EnergyLog) = e.W - (e.K - e.K0) - (e.Π - e.Π0)

"""
    TransientStepCache

Work data of [`TransientModel`](@ref): the nonlinear solver and its cache (tangent, residual, increment,
numerical setup), the state `x` (uₙ₊₁ after every accepted step), the stage unknown `y` (`x` itself on the
general level), the assembler of the lazy backend, the backend, the [`TransientStepOperator`](@ref) of
the stage (its inner operator rebuilt or updated by `build_operator` at every stage) and the backend
operators of the mass and damping forms (reused by `linear_system!` when `mass_update = :step`).
`Δt_assembled` and `coef_assembled` are the increment and the stage coefficients of the last tangent
assembly (`tangent_update = :Δt` re-assembles when either changes, e.g. the BDF2 start-up); `projection` the
operator of the Dirichlet projection (as in `NonlinearStepCache`).
"""
mutable struct TransientStepCache{N<:NonlinearSolver,NC,X<:AbstractVector,A,B<:AbstractBackend,O<:TransientStepOperator}
    const nls::N
    const nls_cache::NC
    const x::X
    const y::X
    const assem::A
    const backend::B
    const op::O
    opM::Any
    opC::Any
    Δt_assembled::Float64
    coef_assembled::StageCoefficients
    projection::Any
    const energies::EnergyLog
    const inertia_mask::Vector{Bool}       # dofs with a time derivative (rows of M with entries); all on the general level
end
TransientStepCache(nls, nls_cache, x, y, assem, backend, op, opM, opC, energies) =
    TransientStepCache{typeof(nls),typeof(nls_cache),typeof(x),typeof(assem),typeof(backend),typeof(op)}(
        nls, nls_cache, x, y, assem, backend, op, opM, opC, NaN, StageCoefficients(NaN, NaN, NaN, NaN, NaN), nothing, energies,
        _inertia_mask(op, length(x)))


# the forms of a stage: the user's forms of the context (general level) or the static forms at the stage
# time with the Neumann loads (structured level)
function _stage_forms(res, jac, structured::Bool, ctx, neumann, t::Float64, load)
    structured || return res(ctx), jac(ctx)
    with_neumann(res(t), neumann, load(t)), jac(t)
end

# mass and damping matrices of the structured level (the pattern of their own forms)
function _mass_matrices(bk, mass, damping, U, V, assem, lumping::Bool)
    M, _, opM = linear_system(bk, mass, nothing, U, V, assem)
    lumping && (M = _lump(M))
    if damping === nothing
        return M, nothing, opM, nothing
    end
    C, _, opC = linear_system(bk, damping, nothing, U, V, assem)
    M, C, opM, opC
end
_lump(M::SparseMatrixCSC) = sparse(Diagonal(vec(sum(M; dims=2))))

# the ModelOperator handed to the Newton: injectivity along the stage field, energy of the stage (§6)
function _model_operator(hooks, spaces, stepop::TransientStepOperator, structured::Bool)
    U, V = spaces[1], spaces[2]
    inj = hooks.injectivity
    ms = inj === nothing ? nothing :
        (y, dy, β, jmin) -> begin
            xe = _stage_field!(stepop, y)
            α0 = structured ? stepop.c.α0 : 1.0
            injectivity_step(inj[1], _field(FEFunction(U, xe), inj[2]), _field(FEFunction(V, α0 == 1.0 ? dy : α0 .* dy), inj[2]), β, jmin)
        end
    has = structured ? hooks.energy !== nothing : hooks.stage_energy !== nothing
    en = has ? (y -> stage_energy(stepop, U, y)) : nothing
    ModelOperator(stepop; max_step=ms, energy=en, tangent_switch=hooks.tangent_switch)
end

# Getters
get_state(m::TransientModel) = FEFunction(get_trial_space(m), m.caches.x)
get_spaces(m::TransientModel) = m.spaces
get_applicators(m::TransientModel) = m.applicators
get_assemblers(m::TransientModel) = m.caches.assem
get_measure(m::TransientModel) = m.res.dΩ
"Velocity and acceleration of the last accepted step as FE functions (second-order schemes)."
get_velocity(m::TransientModel) = m.hist.vnh
get_acceleration(m::TransientModel) = m.hist.anh
"""
    get_previous_state(m::TransientModel)

The FE function over the history `uₙ` (`ctx.u⁻` of the forms): during `post` (before the commit of the
step) the state the step started from; after `step!` the last accepted state (the history is committed).
"""
get_previous_state(m::TransientModel) = m.hist.un

#*******************************************************************************
#    					 Energies and verbose output
#*******************************************************************************

"Energies of the run as a named tuple `(; K, Π, W, balance)` (`NaN` when not tracked; see [`EnergyLog`](@ref))."
get_energies(m::TransientModel) = (e = m.caches.energies; (; K=e.K, Π=e.Π, W=e.W, balance=balance(e)))

# kinetic and stored energies at the state of the history (after the commit of a step, or at the start)
function _energies!(e::EnergyLog, m::TransientModel, t::Float64)
    h = m.hist
    U = m.spaces[1]
    kin = m.hooks.kinetic
    if kin !== nothing
        e.K = sum(kin(h.vnh))
    elseif m.mass !== nothing && derivative_order(m.scheme) == 2
        e.K = 0.5 * sum(m.mass(h.vnh, h.vnh))
    end
    en = m.hooks.energy
    en === nothing || (e.Π = sum(en(t)(FEFunction(U, m.caches.x))))
    nothing
end
# work of the Neumann loads over the step (trapezoidal rule): ½ (f(tₙ) + f(tₙ₊₁)) ⋅ Δu, Dirichlet dofs included
function _work_increment!(e::EnergyLog, m::TransientModel, tn::Float64, t::Float64)
    m.neumann === nothing && return
    h = m.hist
    U = m.spaces[1]
    e.dx .= m.caches.x .- h.xn
    for i in eachindex(e.dg)
        e.dg[i] .= h.gd[i] .- h.gdn[i]
    end
    e.W -= sum(residual_Neumann(m.neumann.bc, _fe_function(U, e.dx, e.dg), m.neumann.dΓ, t, tn))
    nothing
end
function _start_energies!(e::EnergyLog, m::TransientModel, t0::Float64, verbose::Int)
    e.tracked = verbose >= 1
    e.K = NaN; e.Π = NaN; e.W = 0.0
    e.tracked || return
    _energies!(e, m, t0)
    e.K0, e.Π0 = e.K, e.Π
    nothing
end

function step_header_t(v::Int, n::Int, t, Δt)
    v >= 2 || return
    printstyled(@sprintf("━━ Step %d ━━ t = %.4e  Δt = %.3e ━━\n", n, t, Δt); bold=true)
end
function step_line_t(v::Int, m::TransientModel, n::Int, t, Δt, its::Int, ok::Bool, err::Float64, time)
    v >= 1 || return
    nls = m.caches.nls
    e = m.caches.energies
    @printf("step %4d  t = %.4e  Δt = %.3e  it = %3d", n, t, Δt, its)
    if nls isa NewtonRaphson && m.tangent_update === :iteration && !isempty(residual_history(nls))
        @printf("  ‖R‖ = %.3e", residual_history(nls)[end])
    end
    isnan(err) || @printf("  err = %.2e", err)
    if ok && e.tracked
        isnan(e.K) || @printf("  K = %.4e", e.K)
        isnan(e.Π) || @printf("  Π = %.4e", e.Π)
        isnan(e.K) || isnan(e.Π) || @printf("  K+Π = %.4e", e.K + e.Π)
        (m.neumann !== nothing || !isnan(e.K) || !isnan(e.Π)) && @printf("  balance = %.2e", balance(e))
    end
    print("  ")
    if ok
        printstyled("✓"; color=:green)
    else
        printstyled(isnan(err) ? "✗ → bisect" : "✗ → reject"; color=:yellow)
    end
    @printf("  %.2f s\n", time)
end
function run_summary_t(v::Int, m::TransientModel, st::TimeStep, Δtrange, time)
    v >= 1 || return
    nls = m.caches.nls
    e = m.caches.energies
    @printf("t = %.4e reached: %d steps, %d bisections, %d rejections, Δt ∈ [%.3e, %.3e], %.2f s\n",
            st.t, st.nsteps, st.nbisect_total, st.nreject_total, Δtrange[1], Δtrange[2], time)
    if e.tracked && (m.neumann !== nothing || !isnan(e.K) || !isnan(e.Π))
        @printf("energy balance W − ΔK − ΔΠ = %.3e", balance(e))
        isnan(e.K) || @printf("  (K = %.4e", e.K)
        isnan(e.Π) || @printf("%s Π = %.4e", isnan(e.K) ? "  (" : ",", e.Π)
        m.neumann === nothing || @printf("%s W = %.4e", isnan(e.K) && isnan(e.Π) ? "  (" : ",", e.W)
        print(")\n")
    end
    nls isa NewtonRaphson && print_summary(nls.log)
end

#*******************************************************************************
#    					 One attempt of the time loop
#*******************************************************************************

"""
    step!(m::TransientModel, st::TimeStep, stepping; ProjectDirichlet, load, post, presolver, on_bisection, on_reject) -> Bool

One attempt of the time loop: the next target of `st` ([`next_target!`](@ref)) from the last accepted
step; `presolver(t, Δt)` first, `update_time_step!(physics, Δt)`, the Dirichlet values at `load(t)`; then
every stage of the scheme (coefficients, Dirichlet values of the stage, forms, predictor — with the
Dirichlet projection of the static tangent when `ProjectDirichlet` —, solve, `commit_stage!`). A failed
solve bisects the increment ([`bisect!`](@ref), `on_bisection(t, Δt)`); with `AdaptiveStepping` a step
whose error estimate exceeds the tolerance is rejected ([`reject!`](@ref), `on_reject(t, Δt, err)`);
otherwise the step is accepted: `post(t)` (the state is uₙ₊₁ and the Dirichlet values those of t; the
velocity and acceleration of the history are still vₙ, aₙ — the commit comes after), `internal(xh, xh⁻)`, the commit of the scheme (history,
velocities and accelerations, Dirichlet values) and [`accept!`](@ref). Returns whether the step was
accepted. On failure or rejection the state, the time, the Dirichlet values and the history are those of
the last accepted step.
"""
function step!(m::TransientModel, st::TimeStep, stepping::TimeControl;
    ProjectDirichlet::Bool=false, load=identity, post=PostProcessor(), presolver=nothing, on_bisection=nothing, on_reject=nothing)

    U = m.spaces[1]
    h = m.hist
    c = m.caches
    nls, x, y = c.nls, c.x, c.y
    s = m.scheme
    verbose = nls_verbosity(nls)
    adaptive = stepping isa AdaptiveStepping

    t, Δt = next_target!(st, stepping)
    tn = st.t
    _hook(presolver, tn, Δt)
    m.physics === nothing || update_time_step!(m.physics, Δt)
    h.Δt⁻ = st.Δt_prev
    c.op.structured && m.mass_update === :step && _reassemble_mass!(m)
    step_header_t(verbose, st.nsteps + 1, t, Δt)
    t_step = time()
    ok = true
    its = 0
    for i in 1:nstages(s)
        _build_stage!(m, st, i, tn, Δt, load)
        predict!(y, s, h, i, Δt)
        if ProjectDirichlet && i == 1 && implicit(s, i) && projectable(s)
            α = _project_dirichlet!(m, st, tn, Δt, load)
            α > 0 || error("HyperFEM: the Dirichlet projection at t = $tn gives a zero safe load factor (α = $α): the " *
                           "state already violates the injectivity bound of the line search along the Dirichlet increment")
            if α < 1
                # the safe load factor cuts the increment (as the static driver cuts ΔΛ): the stage is rebuilt
                # for Δt α, and the physics and the presolver hear the real increment
                Δt = α * Δt
                t = tn + Δt
                st.t_target, st.Δt_attempt, st.cut = t, Δt, true
                m.physics === nothing || update_time_step!(m.physics, Δt)
                _hook(presolver, tn, Δt)
                _build_stage!(m, st, i, tn, Δt, load)
                predict!(y, s, h, i, Δt)
            end
            # the projection factorised the Newton cache with the static tangent: the linear policies must
            # re-assemble the stage tangent
            c.Δt_assembled = NaN
            y .+= α .* get_free_dof_values(FEFunction(m.spaces[3], c.nls_cache.dx))
        end
        ok = _solve_stage!(m, y, Δt, implicit(s, i))
        its += nls_iterations_of(m, implicit(s, i))
        ok || break
        commit_stage!(s, h, i, y, Δt)
    end
    err = NaN
    if ok && adaptive
        error_estimate!(s, h, y, Δt, h.Δt⁻)
        err = weighted_error(h.e, _field_np1!(x, s, h, y, Δt), h.xn, stepping.rtol, stepping.atol, c.inertia_mask)
        ok = err <= 1.0
    end
    en = c.energies
    if ok
        _field_np1!(x, s, h, y, Δt)
        _dcopy!(_dirichlet_values(U), h.gd)
        en.tracked && _work_increment!(en, m, tn, t)
        post(t)
        m.internal === nothing || m.internal(FEFunction(U, x), h.un)
        commit!(s, h, x, Δt)
        m.history === nothing || commit!(m.history, x, h)
        en.tracked && _energies!(en, m, t)
        step_line_t(verbose, m, st.nsteps + 1, t, Δt, its, true, err, time() - t_step)
        adaptive ? accept!(st, stepping, its, err, error_order(s)) : accept!(st, stepping, its)
        return true
    end
    step_line_t(verbose, m, st.nsteps + 1, t, Δt, its, false, err, time() - t_step)
    # back to the last accepted step
    x .= h.xn
    y === x || fill!(y, zero(eltype(y)))
    _dcopy!(_dirichlet_values(U), h.gdn)
    if !isnan(err)   # converged, rejected by the error estimate
        reject!(st, stepping, err, error_order(s))
        _hook3(on_reject, tn, st.Δt, err)
    else
        @warn "Bisection performed!"
        bisect!(st, stepping)
        _hook(on_bisection, tn, st.Δt)
    end
    return false
end

# Stage `i` of the attempt (tₙ, Δt): the Dirichlet values at the target time into h.gd, the coefficients and β
# vectors of the stage, the Dirichlet values of the trial space during the stage, the inner operator of the
# stage forms (the user's forms of the context, or the static forms at the stage time with the Neumann loads)
function _build_stage!(m::TransientModel, st::TimeStep, i::Int, tn::Float64, Δt::Float64, load)
    U, V = m.spaces[1], m.spaces[2]
    h = m.hist
    c = m.caches
    o = c.op
    s = m.scheme
    t = tn + Δt
    if i == 1
        _dirichlet_ahead!(s, m, U, t, Δt, load)
        TrialFESpace!(U, m.applicators[1], load(t); stage=t)
        _dcopy!(h.gd, _dirichlet_values(U))
    end
    coef = stage_coefficients!(s, h, i, Δt, h.Δt⁻)
    ti = stage_time(s, tn, Δt, i)
    ctx = TimeContext(ti, Δt, st.nsteps + 1, i, coef, h, m.history)
    o.c = coef
    o.use_b0 = h.use_β0
    o.structured ? _stage_dirichlet!(U, s, h, coef) : _dcopy!(_dirichlet_values(U), _unknown_dirichlet(s, h))
    r, j = _stage_forms(m.res, m.jac, o.structured, ctx, m.neumann, ti, load)
    o.inner = build_operator(c.backend, r, j, U, V, c.assem, o.inner)
    if o.structured
        en = m.hooks.energy
        o.energy = en === nothing ? nothing : with_neumann_energy(en(ti), m.neumann, load(ti))
    else
        en = m.hooks.stage_energy
        o.energy = en === nothing ? nothing : en(ctx)
    end
    o.structured && _dirichlet_inertia!(m, coef)
    nothing
end

_hook3(::Nothing, t, Δt, err) = nothing
_hook3(f, t, Δt, err) = (f(t, Δt, err); nothing)

# commit of a user history object: `commit!(obj, x, hist)` when defined, nothing otherwise
commit!(obj, x::AbstractVector, h::StageHistory) = nothing

# the field uₙ₊₁ from the stage unknown(s): y itself, the known field (explicit), uₙ + Δt Σ bᵢ kᵢ (RK)
function _field_np1!(x::AbstractVector, s::TimeScheme, h::StageHistory, y, Δt::Float64)
    x === y || (x .= y)
    x
end
_field_np1!(x::AbstractVector, ::CentralDifference, h::StageHistory, y, Δt::Float64) = (x .= h.b0; x)
function _field_np1!(x::AbstractVector, s::SDIRK, h::StageHistory, y, Δt::Float64)
    x .= h.xn
    for i in eachindex(s.b)
        x .+= (Δt * s.b[i]) .* h.ks[i]
    end
    x
end

# Dirichlet velocity and acceleration at t0 by one-sided differences of the prescribed values at t0, t0 + Δt,
# t0 + 2Δt (the velocity only when no `vh` was given; nothing when the values do not move)
function _initial_dirichlet_kinematics!(m::TransientModel, U, t0::Float64, Δt::Float64, load)
    h = m.hist
    g0 = h.gdn                      # the values at t0 (initialize! has set them)
    TrialFESpace!(U, m.applicators[1], load(t0 + Δt); stage=t0 + Δt)
    g1 = deepcopy(_dirichlet_values(U))
    TrialFESpace!(U, m.applicators[1], load(t0 + 2Δt); stage=t0 + 2Δt)
    g2 = _dirichlet_values(U)
    for i in eachindex(g0)
        h.adn[i] .= (g0[i] .- 2.0 .* g1[i] .+ g2[i]) ./ Δt^2
        m.velocity_given || (h.vdn[i] .= (-3.0 .* g0[i] .+ 4.0 .* g1[i] .- g2[i]) ./ (2Δt))
    end
    TrialFESpace!(U, m.applicators[1], load(t0); stage=t0)
    nothing
end

# the prescribed values one step ahead, g(tₙ₊₁ + Δt), for the central differences of the Dirichlet kinematics
# of the explicit scheme (into h.kd; the trial space is set to tₙ₊₁ right after)
_dirichlet_ahead!(::TimeScheme, m::TransientModel, U, t, Δt, load) = nothing
function _dirichlet_ahead!(::CentralDifference, m::TransientModel, U, t, Δt, load)
    TrialFESpace!(U, m.applicators[1], load(t + Δt); stage=t + Δt)
    _dcopy!(m.hist.kd, _dirichlet_values(U))
    nothing
end

# Dirichlet values of the stage unknown: the prescribed values g_{n+1} (the unknown is uₙ₊₁), the Dirichlet
# derivative kd (SDIRK: the unknown is a stage derivative); for an explicit stage (α₀ = 0) they do not
# enter the stage field
_unknown_dirichlet(::TimeScheme, h::StageHistory) = h.gd
_unknown_dirichlet(::SDIRK, h::StageHistory) = h.kd
_unknown_dirichlet(::CentralDifference, h::StageHistory) = h.b2d
# Dirichlet values of the trial space during a stage of the structured level: the stage field's, α₀ g_y + β₀d
function _stage_dirichlet!(U, s::TimeScheme, h::StageHistory, c::StageCoefficients)
    gy = _unknown_dirichlet(s, h)
    h.use_β0 ? _dcombine!(_dirichlet_values(U), c.α0, gy, 1.0, h.b0d) : _dcombine!(_dirichlet_values(U), c.α0, gy, 0.0, gy)
end

# dofs with a time derivative (rows of the mass with entries), for the error norm and the initial
# acceleration; all on the general level. Rows with a nonzero value: the compiled backend stores the whole
# pattern of the space, with zeros on the fields the mass form does not touch.
function _inertia_mask(o::TransientStepOperator, n::Int)
    o.structured || return trues(n)
    M = o.M
    mask = falses(size(M, 1))
    rv, nz = rowvals(M), nonzeros(M)
    for j in 1:size(M, 2), k in nzrange(M, j)
        nz[k] == 0.0 || (mask[rv[k]] = true)
    end
    mask
end
_inertia_mask(m::TransientModel) = m.caches.inertia_mask

# inertia of the moving Dirichlet values (structured level): the load vector of mass(a_d, v) with the
# algorithmic Dirichlet acceleration (or velocity, first order); skipped when it vanishes
function _dirichlet_inertia!(m::TransientModel, c::StageCoefficients)
    o = m.caches.op
    h = m.hist
    U, V = m.spaces[1], m.spaces[2]
    o.use_bd = false
    second = o.dorder == 2
    βd = second ? h.b2d : h.b1d
    α = second ? c.α2 : c.α1
    # algorithmic time derivative of the Dirichlet values: α × (Dirichlet stage unknown) + β_d
    # skipped when it vanishes up to round-off (a constant Dirichlet value gives α g + β_d = O(ε α g), not 0)
    nz = false
    for i in eachindex(βd)
        for k in eachindex(βd[i])
            gy = _dirichlet_unknown(m.scheme, h, i, k)
            v = α * gy + βd[i][k]
            o.ad[i][k] = v
            abs(v) <= 1e-12 * (abs(α * gy) + abs(βd[i][k])) || (nz = true)
        end
    end
    nz || return
    zh = _fe_function(U, o.zfree, o.ad)
    lf = v -> m.mass(zh, v)
    o.opbd = load_vector!(m.caches.backend, o.bd, lf, U, V, m.caches.assem, o.opbd)
    if second && o.C !== nothing
        # damping of the Dirichlet velocity: α₁ g' + β₁d, added to the same vector
        for i in eachindex(h.b1d)
            for k in eachindex(h.b1d[i])
                o.ad[i][k] = c.α1 * _dirichlet_unknown(m.scheme, h, i, k) + h.b1d[i][k]
            end
        end
        lfC = v -> m.damping(zh, v)
        o.opbdC = load_vector!(m.caches.backend, o.w, lfC, U, V, m.caches.assem, o.opbdC)
        o.bd .+= o.w
    end
    o.use_bd = true
    nothing
end
# the stage unknown's Dirichlet values that the algorithmic derivative multiplies by α: g_{n+1} (the unknown
# is uₙ₊₁), kd (SDIRK: ∂t = kᵢ, with b1d = 0), nothing for CentralDifference (b1d, b2d carry the full values)
_dirichlet_unknown(::TimeScheme, h::StageHistory, i::Int, k::Int) = h.gd[i][k]
_dirichlet_unknown(::CentralDifference, h::StageHistory, i::Int, k::Int) = 0.0
_dirichlet_unknown(::SDIRK, h::StageHistory, i::Int, k::Int) = h.kd[i][k]

function _reassemble_mass!(m::TransientModel)
    c = m.caches
    o = c.op
    U, V = m.spaces[1], m.spaces[2]
    c.opM = linear_system!(c.backend, o.M, o.w, m.mass, nothing, U, V, c.assem, c.opM)
    m.mass_lumping && error("HyperFEM: mass_lumping with mass_update = :step is not supported")
    if o.C !== nothing
        c.opC = linear_system!(c.backend, o.C, o.w, m.damping, nothing, U, V, c.assem, c.opC)
    end
    nothing
end

# the stage solve: Newton (tangent_update = :iteration) or one linear solve from the zero guess
function _solve_stage!(m::TransientModel, y::AbstractVector, Δt::Float64, implicit_stage::Bool)
    c = m.caches
    nls, cache = c.nls, c.nls_cache
    policy = implicit_stage ? m.tangent_update : :Δt
    mop = _model_operator(m.hooks, m.spaces, c.op, c.op.structured)
    if policy === :iteration
        Algebra.solve!(y, nls, mop, cache; r_reference=_residual_reference(c.op, m.hist, y))
        return nls_converged(nls)
    end
    A, b, dx = cache.A, cache.b, cache.dx
    fill!(y, zero(eltype(y)))
    if policy === :step || c.Δt_assembled != Δt || c.coef_assembled != c.op.c
        jacobian!(A, mop, y)
        numerical_setup!(cache, y)
        c.Δt_assembled = Δt
        c.coef_assembled = c.op.c
    end
    residual!(b, mop, y)
    b .*= -1.0
    Algebra.solve!(dx, numerical_setup(cache), b)
    y .= dx
    return linear_converged(numerical_setup(cache))
end
# scale of the stage residual for the relative criterion of the Newton: the inertia term at the initial
# guess (structured level; one SpMV), which a good predictor makes `‖R₀‖` much smaller than
function _residual_reference(o::TransientStepOperator, h::StageHistory, y::AbstractVector)
    o.structured || return 0.0
    c = o.c
    if o.dorder == 2
        o.w .= c.α2 .* y .+ o.b2
        hist = h.an
    elseif o.dorder == 1
        o.w .= c.α1 .* y .+ o.b1
        hist = h.ẋn
    else
        return 0.0
    end
    mul!(o.x_eval, o.M, o.w)        # x_eval is rebuilt by the next residual/jacobian call
    r = norm(o.x_eval)
    # at the Newmark predictor α₂ ũ + β₂ = αm aₙ vanishes (αm = 0): the inertia of the history is the scale then
    mul!(o.x_eval, o.M, hist)
    max(r, norm(o.x_eval))
end
nls_iterations_of(m::TransientModel, implicit_stage::Bool) =
    implicit_stage && m.tangent_update === :iteration ? nls_iterations(m.caches.nls) : 1

# Dirichlet projection of R3 with the stage forms at the state of step n: the increment of the Dirichlet
# values from tₙ to tₙ₊₁ lifted through the tangent, into the increment of the Newton cache; returns the
# safe load factor of the line search along it
function _project_dirichlet!(m::TransientModel, st::TimeStep, tn::Float64, Δt::Float64, load)
    U, V, ∆U = m.spaces
    c = m.caches
    h = m.hist
    o = c.op
    t = tn + Δt
    TrialFESpace!(∆U, m.applicators[2], load(t), load(t) - load(tn); stage=(t, Δt))
    uh = FEFunction(U, h.xn, _dirichlet_values_vec(U, h.gdn))
    if o.structured
        r, j = with_neumann(m.res(tn), m.neumann, load(tn)), m.jac(tn)
    else
        coef = stage_coefficients!(m.scheme, h, 1, Δt, h.Δt⁻)
        ctx = TimeContext(stage_time(m.scheme, tn, Δt, 1), Δt, st.nsteps + 1, 1, coef, h, m.history)
        r, j = m.res(ctx), m.jac(ctx)
    end
    A, b, dx = projection_system(c.nls_cache)
    c.projection = dirichlet_projection!(c.backend, A, b, r, j, uh, ∆U, V, c.assem, c.projection)
    ns = projection_setup!(c.nls_cache, h.xn)
    Algebra.solve!(dx, ns, b)
    duh = FEFunction(∆U, dx)
    dirichlet_step(c.nls.linesearch, m, uh, duh)
end
_dirichlet_values_vec(U::MultiFieldFESpace, d::Vector) = d
_dirichlet_values_vec(U, d::Vector) = d[1]
function Solvers.injectivity_step(m::TransientModel, xh, dxh, β, jmin)
    inj = m.hooks.injectivity
    inj === nothing && error("HyperFEM: the model has no `injectivity = …` measure for the injectivity step")
    injectivity_step(inj[1], _field(xh, inj[2]), _field(dxh, inj[2]), β, jmin)
end

#*******************************************************************************
#    					 The loop
#*******************************************************************************

"""
    solve!(m::TransientModel; stepping, ProjectDirichlet=false, load=identity, post, presolver, on_bisection,
           on_reject) -> (x, flagconv)

Time loop by [`step!`](@ref) from `t0` to `t_end` of `stepping` (a [`TimeStepping`](@ref), an
[`AdaptiveStepping`](@ref) or a `NamedTuple` `(Δt = …, t_end = …)`). The conditions and loads are taken
at `load(t)` (`identity`: seconds). `flagconv` is 1 when `t_end` was reached and 0 when the loop was
exhausted (consecutive bisections or rejections, `Δt < Δtmin`). The initial state is the one of the
model (`xh`, `vh`); the initial acceleration follows `initial_acceleration`.
"""
function solve!(m::TransientModel; stepping, ProjectDirichlet::Bool=false, load=identity,
    post=PostProcessor(), presolver=nothing, on_bisection=nothing, on_reject=nothing)

    stepping = as_time_stepping(stepping)
    reset!(post)
    c = m.caches
    nls, x = c.nls, c.x
    verbose = nls_verbosity(nls)
    nls_reset!(nls)
    t_run = time()
    st = TimeStep(stepping)
    stepping isa AdaptiveStepping && !has_error_estimate(m.scheme) &&
        error("HyperFEM: $(typeof(m.scheme)) has no local error estimate; use TimeStepping")
    initialize!(m, st; load)

    flagconv = 1
    Δtmin, Δtmax = Inf, 0.0
    while !finished(st, stepping)
        if step!(m, st, stepping; ProjectDirichlet, load, post, presolver, on_bisection, on_reject)
            Δtmin, Δtmax = min(Δtmin, st.Δt_prev), max(Δtmax, st.Δt_prev)
        end
        if exhausted(st, stepping)
            @warn "Time loop exhausted (bisections, rejections or Δt below Δtmin)"
            flagconv = 0
            break
        end
    end
    run_summary_t(verbose, m, st, (Δtmin, Δtmax), time() - t_run)
    vtk_save(post)
    return x, flagconv
end

"""
    initialize!(m::TransientModel, st::TimeStep; load=identity)

History of the loop at its start `st.t`: the state of the model and the Dirichlet values at `load(st.t)`
become step 0; the initial acceleration follows `initial_acceleration`. `solve!` calls it; a loop driven
by hand with `step!` must call it first.
"""
function initialize!(m::TransientModel, st::TimeStep; load=identity)
    U = m.spaces[1]
    h = m.hist
    c = m.caches
    TrialFESpace!(U, m.applicators[1], load(st.t); stage=st.t)
    h.xn .= c.x
    _dcopy!(h.gdn, _dirichlet_values(U))
    h.nstored = 0
    h.Δt⁻ = 0.0
    h.Δt⁻⁻ = 0.0
    m.initial_acceleration === :solve && _initial_acceleration!(m, st.t, st.Δt, load)
    c.Δt_assembled = NaN
    _start_energies!(c.energies, m, st.t, nls_verbosity(c.nls))
    nothing
end

# a₀ from M a₀ = −r(t0, u₀) (structured level): the residual of the inner forms at the initial state and the
# lumped or consistent mass
function _initial_acceleration!(m::TransientModel, t0::Float64, Δt::Float64, load)
    c = m.caches
    o = c.op
    U, V = m.spaces[1], m.spaces[2]
    h = m.hist
    r, j = _stage_forms(m.res, m.jac, true, nothing, m.neumann, t0, load)
    o.inner = build_operator(c.backend, r, j, U, V, c.assem, o.inner)
    b = c.nls_cache.b
    residual!(b, o.inner, h.xn)
    b .*= -1.0
    o.C === nothing || mul!(b, o.C, h.vn, -1.0, 1.0)       # − C v₀ (free values)
    # the Dirichlet kinematics at t0 from the prescribed values ahead (one-sided differences with the first
    # increment): the inertia M_fd g̈₀ (and the damping C_fd ġ₀) of the moving Dirichlet values enter a₀ — a
    # start with a₀ = 0 on a boundary that accelerates costs the explicit scheme one order
    _initial_dirichlet_kinematics!(m, U, t0, Δt, load)
    o.use_bd = false
    nz = false
    for i in eachindex(h.adn), k in eachindex(h.adn[i])
        o.ad[i][k] = h.adn[i][k]
        h.adn[i][k] == 0.0 || (nz = true)
    end
    if nz
        zh = _fe_function(U, o.zfree, o.ad)
        o.opbd = load_vector!(c.backend, o.bd, v -> m.mass(zh, v), U, V, c.assem, o.opbd)
        b .-= o.bd
        if o.C !== nothing
            for i in eachindex(h.vdn); o.ad[i] .= h.vdn[i]; end
            o.opbdC = load_vector!(c.backend, o.w, v -> m.damping(zh, v), U, V, c.assem, o.opbdC)
            b .-= o.w
        end
    end
    mask = c.inertia_mask
    if all(mask)
        h.an .= o.M \ b
    else
        # fields without inertia (electric potential, pressure): their acceleration is not defined and
        # their rows of the mass matrix are empty; the solve is on the dofs with inertia
        idx = findall(mask)
        fill!(h.an, zero(eltype(h.an)))
        h.an[idx] .= o.M[idx, idx] \ b[idx]
    end
    nothing
end
