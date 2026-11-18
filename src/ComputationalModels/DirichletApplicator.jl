#*******************************************************************************
#    					 Dirichlet applicator
#*******************************************************************************
#
# Sets the Dirichlet values of a TrialFESpace from its BoundaryConditions at a load Λ and a stage τ
# (see `Prescription`) without Gridap's cell-by-cell interpolation of every prescribed function over the
# whole space. Built once per (space, conditions) by `dirichlet_applicator(space, bc)` and owned by the
# computational model (one per space it updates): for every condition, the Dirichlet dofs of its tag, the
# coordinates of their nodes and the component of each dof. Then, at every (Λ, τ):
#   - Separable prescriptions (value(x) * load(Λ), the common case): the values at the nodes are
#     computed once at construction and scaled by load(Λ);
#   - constants and functions of x returned by the other prescriptions: evaluated at the nodes of
#     the tag only;
#   - fields (CellField, FEFunction, Interpolable) returned by a Coupling: the nodes of the tag are located
#     in the mesh of the field once (Transfer.jl); afterwards a finite-element function is a gather or a
#     sparse product on its dof values and any other field is evaluated in the located cells, without
#     point searches. A field defined only on the faces of the tag (an FESpace on an interface) is admissible.
# Spaces whose Dirichlet dofs cannot be mapped to Lagrangian nodes (no LagrangianDofBasis, or
# non-Vector Dirichlet values such as distributed ones) get a `DirichletInterpolator`, which uses
# Gridap's interpolation.

"Dirichlet dofs of one condition: their ids, the node of each (into `coords`) and its component."
struct DirichletTagDofs
    dofs::Vector{Int}
    nodes::Vector{Int}
    comps::Vector{Int}
    coords::Vector                    # distinct node coordinates (Points) of the tag
end

"""
    DirichletApplicator

Precomputed data to set the Dirichlet values of one space from its `BoundaryConditions`; see the header
of this file and [`dirichlet_applicator`](@ref).
"""
mutable struct DirichletApplicator{B<:BoundaryConditions}
    bc::B
    tags::Vector{DirichletTagDofs}
    base::Vector{Any}                 # Separable: value(x)[comp] per dof of the tag; nothing otherwise
    cache::Vector{Any}                # per tag: location of its nodes in the mesh of the coupled field
    field::Vector{Any}                # per tag: what the location was built for (space or triangulation)
end

"Fallback for spaces the applicator does not support: Gridap's interpolation of the prescribed functions."
struct DirichletInterpolator{B<:BoundaryConditions}
    bc::B
end

const DirichletUpdater = Union{DirichletApplicator,DirichletInterpolator}

get_conditions(app::DirichletUpdater) = app.bc

_num_components(::Type{<:Integer}) = 1
_num_components(::Type{T}) where {T} = length(T)

"Value of a prescribed constant/VectorValue at component `c`."
_comp(v::Real, c) = v
_comp(v::MultiValue, c) = v[c]

_rounded(x, tol) = ntuple(k -> round(x[k] / tol) * tol, length(x))

"""
    dirichlet_applicator(space, bc)

The object that sets the Dirichlet values of `space` from `bc`: a [`DirichletApplicator`](@ref) when the
Dirichlet dofs of `space` map to Lagrangian nodes, a `DirichletInterpolator` (Gridap's interpolation)
otherwise, `nothing` for `NothingBC`. Multi-field spaces give a tuple, one per field.
"""
function dirichlet_applicator(space::SingleFieldFESpace, bc::BoundaryConditions)
    dv = get_dirichlet_dof_values(space)
    dv isa Vector || return DirichletInterpolator(bc)
    ntags = num_dirichlet_tags(space)
    ntags == length(bc) || error("HyperFEM: the space has $ntags Dirichlet tags but the conditions have $(length(bc))")
    ntags == 0 && return DirichletApplicator(bc, DirichletTagDofs[], Any[], Any[], Any[])
    dofbasis = first(get_data(get_fe_dof_basis(space)))
    dofbasis isa Gridap.ReferenceFEs.LagrangianDofBasis || return DirichletInterpolator(bc)
    nc = _num_components(eltype(dofbasis.node_and_comp_to_dof))
    dim = num_point_dims(get_triangulation(space))
    # coordinates and component of every Dirichlet dof, through Gridap's own interpolation (any nesting, any mask)
    fk(k) = nc == 1 ? (x -> x[k]) : (x -> VectorValue(ntuple(_ -> x[k], nc)))
    xs = [compute_dirichlet_values_for_tags(space, [fk(k) for _ in 1:ntags]) for k in 1:dim]
    cs = nc == 1 ? ones(Int, length(dv)) :
         round.(Int, compute_dirichlet_values_for_tags(space, [x -> VectorValue(ntuple(Float64, nc)) for _ in 1:ntags]))
    tag_of = get_dirichlet_dof_tag(space)
    tol = 1e-10 * max(maximum(maximum(abs, x; init=0.0) for x in xs), 1.0)
    tags = DirichletTagDofs[]
    base = Any[]
    for (i, c) in enumerate(bc.conditions)
        dofs = findall(==(i), tag_of)
        coords = Any[]
        index = Dict{NTuple{dim,Float64},Int}()
        nodes = Vector{Int}(undef, length(dofs))
        for (j, d) in enumerate(dofs)
            x = VectorValue(ntuple(k -> xs[k][d], dim))
            key = _rounded(x, tol)
            n = get(index, key, 0)
            n == 0 && (push!(coords, x); n = length(coords); index[key] = n)
            nodes[j] = n
        end
        t = DirichletTagDofs(dofs, nodes, cs[dofs], coords)
        push!(tags, t)
        push!(base, c.value isa Separable ? _values_at_nodes(c.value.space, t) : nothing)
    end
    DirichletApplicator(bc, tags, base, Any[nothing for _ in tags], Any[nothing for _ in tags])
end

dirichlet_applicator(space::SingleFieldFESpace, ::NothingBC) = nothing

function dirichlet_applicator(space::MultiFieldFESpace, bc::MultiFieldBoundaryConditions)
    @assert length(space) == length(bc)
    Tuple(map(dirichlet_applicator, space.spaces, bc.fields))
end

"Values at the dofs of a tag of a constant or of a function of x."
_values_at_nodes(v::Number, t::DirichletTagDofs) = [_comp(v, c) for c in t.comps]
function _values_at_nodes(f, t::DirichletTagDofs)
    vals = [_to_number(f(x)) for x in t.coords]
    [_comp(vals[n], c) for (n, c) in zip(t.nodes, t.comps)]
end

# Fields returned by a Coupling: see Transfer.jl (`_prescribed!`), where the nodes of the tag are located once.

# Function barriers: the values come out of a dynamic dispatch on the prescription
function _assign!(dv, dofs, vals)
    @inbounds for (j, d) in enumerate(dofs)
        dv[d] = vals[j]
    end
end
function _assign_scaled!(dv, dofs, base, s)
    @inbounds for (j, d) in enumerate(dofs)
        dv[d] = base[j] * s
    end
end
function _assign!(dv, dofs, vals, vals⁻)
    @inbounds for (j, d) in enumerate(dofs)
        dv[d] = vals[j] - vals⁻[j]
    end
end

"""
    apply!(space, app::DirichletApplicator, Λ, τ)

Set the Dirichlet values of `space` from the conditions of `app` at the load `Λ` and the stage `τ`.
"""
function apply!(space::SingleFieldFESpace, app::DirichletApplicator, Λ, τ)
    dv = get_dirichlet_dof_values(space)
    for (i, c) in enumerate(app.bc.conditions)
        t = app.tags[i]
        isempty(t.dofs) && continue
        p = c.value
        if p isa Separable
            s = _at_load(p.load, Λ)
            _assign_scaled!(dv, t.dofs, app.base[i], s)
        else
            _assign!(dv, t.dofs, _prescribed!(app, i, p, Λ, τ, t))
        end
    end
    space
end

"""
    apply!(space, app::DirichletApplicator, Λ, ΔΛ, τ, Δτ)

Set the Dirichlet values of `space` to the increment of the prescriptions between `(Λ - ΔΛ, τ - Δτ)`
and `(Λ, τ)`: `base .* (load(Λ) - load(Λ - ΔΛ))` for `Separable` prescriptions (no evaluation), the
difference of two evaluations at the nodes of the tag for the others (the earlier one first, so that a
field updated in place by its generator ends in the state of `(Λ, τ)`).
"""
function apply!(space::SingleFieldFESpace, app::DirichletApplicator, Λ, ΔΛ, τ, Δτ)
    dv = get_dirichlet_dof_values(space)
    for (i, c) in enumerate(app.bc.conditions)
        t = app.tags[i]
        isempty(t.dofs) && continue
        p = c.value
        if p isa Separable
            s = _at_load(p.load, Λ) - _at_load(p.load, Λ - ΔΛ)
            _assign_scaled!(dv, t.dofs, app.base[i], s)
        else
            vals⁻ = copy(_prescribed!(app, i, p, Λ - ΔΛ, τ - Δτ, t))      # the buffer of the tag is reused
            _assign!(dv, t.dofs, _prescribed!(app, i, p, Λ, τ, t), vals⁻)
        end
    end
    space
end

# Fallback: Gridap's interpolation of all the prescribed functions
apply!(space::SingleFieldFESpace, app::DirichletInterpolator, Λ, τ) = TrialFESpace!(space, get_space_functions(app.bc, Λ, τ))

function apply!(space::SingleFieldFESpace, app::DirichletInterpolator, Λ, ΔΛ, τ, Δτ)
    TrialFESpace!(space, get_space_functions(app.bc, Λ - ΔΛ, τ - Δτ))
    values⁻ = copy(get_dirichlet_dof_values(space))
    TrialFESpace!(space, get_space_functions(app.bc, Λ, τ))
    get_dirichlet_dof_values(space) .-= values⁻
    space
end
