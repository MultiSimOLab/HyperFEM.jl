#*******************************************************************************
#    					 Dirichlet applicator
#*******************************************************************************
#
# Sets the Dirichlet values of a TrialFESpace from its BoundaryConditions at any load parameter Λ
# without Gridap's cell-by-cell interpolation of every prescribed function over the whole space.
# Built once per (space, conditions): for every condition, the Dirichlet dofs of its tag, the
# coordinates of their nodes and the component of each dof. Then, at every Λ:
#   - Separable prescriptions (value(x) * load(Λ), the common case): the values at the nodes are
#     computed once at construction and scaled by load(Λ);
#   - constants and functions of x returned by the other prescriptions: evaluated at the nodes of
#     the tag only;
#   - fields (CellField, FEFunction, Interpolable): evaluated at the nodes of the tag only, with a
#     cached point search, so a field defined only on the faces of the tag (e.g. an FESpace on an
#     interface) is admissible and the update costs one point search per node.
# Spaces whose Dirichlet dofs cannot be mapped to Lagrangian nodes (no LagrangianDofBasis, or
# non-Vector Dirichlet values such as distributed ones) fall back to Gridap's interpolation.

"Dirichlet dofs of one condition: their ids, the node of each (into `coords`) and its component."
struct DirichletTagDofs
    dofs::Vector{Int}
    nodes::Vector{Int}
    comps::Vector{Int}
    coords::Vector                    # distinct node coordinates (Points) of the tag
end

"""
    DirichletApplicator(space, bc)

Precomputed data to set the Dirichlet values of `space` from `bc` (see the header of this file).
`nothing` when the space is not supported (Gridap's interpolation is used instead).
"""
mutable struct DirichletApplicator
    tags::Vector{DirichletTagDofs}
    base::Vector{Any}                 # Separable: value(x)[comp] per dof of the tag; nothing otherwise
    cache::Vector{Any}                # per tag: Gridap evaluation cache of the last field
    field::Vector{Any}                # per tag: the field the cache was built for
end

_num_components(::Type{<:Integer}) = 1
_num_components(::Type{T}) where {T} = length(T)

"Value of a prescribed constant/VectorValue at component `c`."
_comp(v::Real, c) = v
_comp(v::MultiValue, c) = v[c]

_rounded(x, tol) = ntuple(k -> round(x[k] / tol) * tol, length(x))

function DirichletApplicator(space::SingleFieldFESpace, bc::BoundaryConditions)
    dv = get_dirichlet_dof_values(space)
    dv isa Vector || return nothing
    ntags = num_dirichlet_tags(space)
    ntags == length(bc) || error("HyperFEM: the space has $ntags Dirichlet tags but the conditions have $(length(bc))")
    ntags == 0 && return DirichletApplicator(DirichletTagDofs[], Any[], Any[], Any[])
    dofbasis = first(get_data(get_fe_dof_basis(space)))
    dofbasis isa Gridap.ReferenceFEs.LagrangianDofBasis || return nothing
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
    DirichletApplicator(tags, base, Any[nothing for _ in tags], Any[nothing for _ in tags])
end

"Values at the dofs of a tag of a constant or of a function of x."
_values_at_nodes(v::Number, t::DirichletTagDofs) = [_comp(v, c) for c in t.comps]
function _values_at_nodes(f, t::DirichletTagDofs)
    vals = [_to_number(f(x)) for x in t.coords]
    [_comp(vals[n], c) for (n, c) in zip(t.nodes, t.comps)]
end

_is_field(v) = v isa CellField || v isa Gridap.CellData.Interpolable
_field(f) = f
_field(f::Gridap.CellData.Interpolable) = f.uh

"Values at the dofs of a tag of a field, with the evaluation cache of the applicator."
function _values_at_nodes!(app::DirichletApplicator, i::Int, v, t::DirichletTagDofs)
    fld = _field(v)
    if app.cache[i] === nothing || app.field[i] !== fld
        app.cache[i] = Gridap.Arrays.return_cache(fld, t.coords[1])
        app.field[i] = fld
    end
    cache = app.cache[i]
    vals = [Gridap.Arrays.evaluate!(cache, fld, x) for x in t.coords]
    [_comp(vals[n], c) for (n, c) in zip(t.nodes, t.comps)]
end

"""
    apply!(space, app::DirichletApplicator, bc, Λ)

Set the Dirichlet values of `space` from `bc` at `Λ`. A prescription returning `nothing` (a legacy
coupling not set yet) leaves the values of its tag untouched.
"""
function apply!(space::SingleFieldFESpace, app::DirichletApplicator, bc::BoundaryConditions, Λ)
    dv = get_dirichlet_dof_values(space)
    for (i, c) in enumerate(bc.conditions)
        t = app.tags[i]
        isempty(t.dofs) && continue
        p = c.value
        if p isa Separable
            s = _at_load(p.load, Λ)
            b = app.base[i]
            @inbounds for (j, d) in enumerate(t.dofs)
                dv[d] = b[j] * s
            end
        else
            v = p(Λ)
            v === nothing && continue
            vals = _is_field(v) ? _values_at_nodes!(app, i, v, t) : _values_at_nodes(v, t)
            @inbounds for (j, d) in enumerate(t.dofs)
                dv[d] = vals[j]
            end
        end
    end
    space
end

"The applicator of `space` for `bc` (built at the first use), or `nothing` if not supported."
function _applicator(space::SingleFieldFESpace, bc::BoundaryConditions)
    key = get_dirichlet_dof_values(space)
    get!(bc.applicators, key) do
        DirichletApplicator(space, bc)
    end
end
