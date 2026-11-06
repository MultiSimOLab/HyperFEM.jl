module HyperFEMKernelsExt

# Compiled backend (HyperFEMKernels) for HyperFEM's drivers: `StaticNonlinearModel(...; backend = :compiled)` and
# `StaticLinearModel(...; backend = :compiled)`.
# The first load step builds a `KernelOperator` (tables, sparsity pattern, colouring); later
# steps only re-capture the weak form for the new load factor with `update`.

using HyperFEM
using HyperFEM.Backends: CompiledBackend
import HyperFEMKernels
using HyperFEMKernels: KernelOperator, update, KernelState, KernelUpdater, MatrixFreeJacobian, MatrixFreeSolver, sparsity_pattern
using HyperFEMKernels: KernelFunctional
using Gridap.FESpaces: FEFunction
using Gridap: ∫, ∇
using Gridap.Algebra: residual!, jacobian!, zero_initial_guess, SymbolicSetup, NumericalSetup
using Gridap.Algebra: allocate_residual, allocate_jacobian

function HyperFEM.ComputationalModels.build_operator(b::CompiledBackend, res, jac, U, V, assem, previous)
    previous isa KernelOperator ? update(previous, res, jac) : KernelOperator(res, jac, U, V; jacobian=b.jacobian)
end

# Dirichlet projection (`solve!(...; ProjectDirichlet=true)`): the linearised problem
# jac(uh, du, v) = -res(uh, v) on ∆U is the operator r(du, v) = jac(uh, du, v) + res(uh, v),
# j(du, ddu, v) = jac(uh, ddu, v), evaluated at du = 0: the residual there is res(uh, v) plus the
# contribution of the Dirichlet increment stored in ∆U, and the Jacobian is the tangent. The
# operator of the previous step is reused (`update`): same spaces, pattern and colouring. The system
# is assembled into the tangent and the residual of the Newton cache, whose pattern (that of the
# model's operator on U, V) is the one of this operator (∆U shares the free dofs of U).
function HyperFEM.ComputationalModels.dirichlet_projection!(bk::CompiledBackend, A, b, res, jac, uh, ∆U, V, assem, previous)
    r(du, v) = jac(uh, du, v) + res(uh, v)
    j(du, ddu, v) = jac(uh, ddu, v)
    op = previous isa KernelOperator ? update(previous, r, j) : KernelOperator(r, j, ∆U, V; jacobian=bk.jacobian)
    _check_pattern(A, op, previous)
    x0 = zero_initial_guess(op)
    residual!(b, op, x0)
    b .*= -1.0
    jacobian!(A, op, x0)
    op
end

# Linear problems (`StaticLinearModel(...; backend = :compiled)`): the affine problem K x = b of
# bf(u, v) = lf(v) is the operator r(u, v) = bf(u, v) - lf(v), j(u, du, v) = bf(du, v) at u = 0, so
# K = j and b = -r(0); r(0) carries the Dirichlet lift (the FE function of u = 0 holds the Dirichlet
# values of U, read by the kernels at every assembly). Re-assembly re-captures the forms with
# `update` (same tables, pattern and colouring), as the nonlinear driver does per load step. A
# linear form returning a number (`l(v) = 0.0`, no load) is accepted as in Gridap.
_linear_res(bf, lf) = (u, v) -> _minus(bf(u, v), lf(v))
_linear_res(bf, ::Nothing) = bf
_linear_jac(bf) = (u, du, v) -> bf(du, v)
_load_res(lf) = (u, v) -> -1.0 * _contribution(lf(v))
_minus(a, b) = a - b
_minus(a, b::Number) = iszero(b) ? a : error("HyperFEM: a linear form must return integrals (got $b)")
_contribution(c) = c
_contribution(c::Number) = iszero(c) ? error("HyperFEM: a linear form without integrals (got $c) cannot be assembled by the kernel backend") :
    error("HyperFEM: a linear form must return integrals (got $c)")

_kernel_operator(op, r, j, U, V) = op isa KernelOperator ? update(op, r, j) : KernelOperator(r, j, U, V)

function _linear_assemble!(K, b, op, x0)
    jacobian!(K, op, x0)
    if b !== nothing
        residual!(b, op, x0)
        b .*= -1.0
    end
end

# `K` was allocated from the pattern of a previous operator (or of the model's operator, for the Dirichlet
# projection): `update` keeps it unless fallback terms (assembled by Gridap) changed the pattern, which the
# setup of the model could not follow.
function _check_pattern(K, op::KernelOperator, previous)
    previous isa KernelOperator && op.pattern === previous.pattern && return
    A0 = sparsity_pattern(op)
    (size(K) == size(A0) && K.colptr == A0.colptr && K.rowval == A0.rowval) ||
        error("HyperFEM: the sparsity pattern of the kernel operator does not match the matrix it assembles into")
end
_check_pattern(::MatrixFreeJacobian, ::KernelOperator, previous) = nothing   # the operator is prepared, not assembled

function HyperFEM.ComputationalModels.linear_system(::CompiledBackend, bf, lf, U, V, assem)
    op = KernelOperator(_linear_res(bf, lf), _linear_jac(bf), U, V)
    x0 = zero_initial_guess(op)
    K, b = allocate_jacobian(op, x0), allocate_residual(op, x0)
    _linear_assemble!(K, lf === nothing ? nothing : b, op, x0)
    K, b, op
end

function HyperFEM.ComputationalModels.linear_system!(::CompiledBackend, K, b, bf, lf, U, V, assem, previous)
    op = _kernel_operator(previous, _linear_res(bf, lf), _linear_jac(bf), U, V)
    _check_pattern(K, op, previous)
    _linear_assemble!(K, lf === nothing ? nothing : b, op, zero_initial_guess(op))
    op
end

function HyperFEM.ComputationalModels.load_vector!(::CompiledBackend, b, lf, U, V, assem, previous)
    previous isa KernelOperator || error("HyperFEM: load_vector! with the kernel backend needs the operator of linear_system")
    op = update(previous, _load_res(lf), previous.jac)
    residual!(b, op, zero_initial_guess(op))
    b .*= -1.0
    op
end

# Matrix-free tangent (`CompiledBackend(jacobian = :matrixfree)`, solver stage S3): the Newton's hooks for the
# setup of HyperFEMKernels' `MatrixFreeSolver`, and the state-aware setup of the solver inside its `LaggedAssembled`
# slot (HyperFEM's `PETScSolver` takes the rigid-body modes from the state).
const MFNS = HyperFEMKernels.MatrixFreeSolverNS
HyperFEM.linear_converged(ns::MFNS) = HyperFEMKernels.linear_converged(ns)
HyperFEM.linear_iterations(ns::MFNS) = HyperFEMKernels.linear_iterations(ns)
HyperFEM.set_tolerance!(ns::MFNS, rtol) = HyperFEMKernels.set_tolerance!(ns, rtol)
HyperFEM.tolerance(ns::MFNS) = HyperFEMKernels.tolerance(ns)
HyperFEM.preconditioner_builds(ns::MFNS) = HyperFEMKernels.preconditioner_builds(ns)
HyperFEM.linear_info(ns::MFNS) = " $(HyperFEMKernels.linear_iterations(ns))"
HyperFEM.Solvers._linear_name(s::MatrixFreeSolver) = "MF-" * uppercase(string(s.krylov))
HyperFEMKernels.numerical_setup_state(ss::SymbolicSetup, A, x) = HyperFEM.numerical_setup_state(ss, A, x)
HyperFEMKernels.numerical_setup_state!(ns::NumericalSetup, A, x) = HyperFEM.numerical_setup_state!(ns, A, x)

# Energy and injectivity step of the drivers' line searches with the kernels (`KernelFunctional`, solver stage S3):
# the functionals live in the step cache (`store`), the energy is re-captured with `update` at every load factor
# (new tables are not built), and the injectivity step reads the increment from an FE function over a buffer of the
# cache, so the pointwise function is compiled once (per β, jmin of the line search).
mutable struct KernelFunctionals
    energy::Any          # KernelFunctional (reduce = :sum) or nothing
    injectivity::Any     # KernelFunctional (reduce = :min) or nothing
    dx::Vector{Float64}  # increment buffer read by `dxh`
    dxh::Any             # FEFunction on V over `dx`
    β::Float64
    jmin::Float64
end
_functionals(store::Base.RefValue{Any}) = store[] === nothing ? (store[] = KernelFunctionals(nothing, nothing, Float64[], nothing, NaN, NaN)) : store[]::KernelFunctionals
_functionals(::Nothing) = KernelFunctionals(nothing, nothing, Float64[], nothing, NaN, NaN)

function HyperFEM.ComputationalModels.energy_function(::CompiledBackend, e, U, store)
    st = _functionals(store)
    fun = st.energy === nothing ? KernelFunctional(e, U) : update(st.energy, e)
    st.energy = fun
    x -> fun(x)
end

function HyperFEM.ComputationalModels.injectivity_function(::CompiledBackend, inj, U, V, store)
    st = _functionals(store)
    dΩ, field = inj
    _field = HyperFEM.ComputationalModels._field
    (x, dx, β, jmin) -> begin
        if st.injectivity === nothing || st.β != β || st.jmin != jmin || length(st.dx) != length(dx)
            st.dx = zeros(length(dx))
            st.dxh = FEFunction(V, st.dx)
            duh = st.dxh
            f(∇u, ∇du) = HyperFEM.Solvers.injectivity_step(∇u, ∇du, β, jmin)
            form(u) = ∫(f ∘ (∇(_field(u, field))', ∇(_field(duh, field))'))dΩ
            st.injectivity = KernelFunctional(form, U; reduce=:min)
            st.β, st.jmin = β, jmin
        end
        copyto!(st.dx, dx)
        min(1.0, st.injectivity(x))
    end
end

# State variables: `CellState(model, dΩ; backend = :compiled)` stores them as dense `KernelState`s, whose
# updates run with the engine automatically; `update_state!(model, ...; backend = :compiled)` runs the
# update with the engine for Gridap's `CellState`s too.
HyperFEM.PhysicalModels._new_state(::CompiledBackend, v, points) = KernelState(v, points)
HyperFEM.PhysicalModels._state_updater(::CompiledBackend, f) = KernelUpdater(f)

end # module
