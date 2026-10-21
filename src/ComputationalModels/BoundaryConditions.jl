
abstract type TimedependentCondition end
abstract type DirichletCoupling end


struct MultiFieldTC{A} <: TimedependentCondition
    vh::A # could be a multifield or single field
    BlockID::Int64
    function MultiFieldTC(vel::Function, V::MultiFieldFESpace; BlockID::Int64=1)
        v = zero_free_values(V)
        vh = FEFunction(V, v)
        vuh = interpolate_everywhere(vel, V[BlockID])

        view_vh = get_free_dof_values(vh[1])
        view_vuh = get_free_dof_values(vuh)
        view_vh .= view_vuh
        new{typeof(vh)}(vh, BlockID)
    end
end

function (obj::MultiFieldTC)()
    obj.vh[obj.BlockID]
end

struct SingleFieldTC{A} <: TimedependentCondition
    vh::A
    function SingleFieldTC(vel::Function, V::SingleFieldFESpace)
        vh = interpolate_everywhere(vel, V)
        new{typeof(vh)}(vh)
    end
end

function (obj::SingleFieldTC)()
    obj.vh
end


#*******************************************************************************
#    					 Prescriptions
#*******************************************************************************

"""
    Prescription

What is prescribed on a boundary, as a function of the load parameter `Λ`.
`p(Λ)` returns an object accepted by Gridap for a single tag:
a `Number`/`VectorValue`, a function of the coordinates `x -> value`, or a `CellField`.
"""
abstract type Prescription end

_to_number(v::Number) = v
_to_number(v::AbstractVector{<:Real}) = VectorValue(v...)

_to_space(v::Union{Number,AbstractVector{<:Real}}) = _to_number(v)
_to_space(f) = x -> _to_number(f(x))  # Any callable accepting the coordinates `x`

"""
    Separable(space, load=nothing)

Prescribed value `space(x) * load(Λ)`. `space` is either a constant (`Number`, `VectorValue`,
`AbstractVector`) or a callable `x -> value`. `load === nothing` stands for `Λ -> 1`.
See [`Prescription`](@ref).
"""
struct Separable{S,L} <: Prescription
    space::S
    load::L
    Separable(space, load=nothing) = (s = _to_space(space); new{typeof(s),typeof(load)}(s, load))
end

_at_load(::Nothing, Λ) = 1.0
_at_load(f, Λ) = f(Λ)

(p::Separable{<:Number})(Λ) = p.space * _at_load(p.load, Λ)

(p::Separable)(Λ) = x -> p.space(x) * _at_load(p.load, Λ)

"""
    NonSeparable(f)

Prescribed value `f(x, Λ)`, for prescriptions such as a rigid rotation.
See [`Prescription`](@ref)
"""
struct NonSeparable{F} <: Prescription
    f::F
end

(p::NonSeparable)(Λ) = x -> _to_number(p.f(x, Λ))

"""
    Parametric(g)

Prescribed object `g(Λ)`, which can be anything accepted by Gridap for a single tag
(e.g. a `CellField` or a function of `x`). See [`Prescription`](@ref)
"""
struct Parametric{G} <: Prescription
    g::G
end

(p::Parametric)(Λ) = p.g(Λ)

"""
    Coupled(g)

Prescribed field `g(Λ)` (an `FEFunction`, `CellField` or `Interpolable`) that may be defined only on
the faces of its tag, e.g. a field of an `FESpace` built on an interface. Gridap receives a placeholder
for the tag; the field is evaluated solely at the Dirichlet nodes of the tag and written by mask after
Gridap's interpolation (see `apply_coupling!`). The evaluation cache (point search included) is kept
while `g(Λ)` returns the same field object, so a field updated in place (e.g. `interpolate_everywhere!`
into the same `FEFunction`) costs one point search per node; a new field object rebuilds it.
`g(Λ) === nothing` leaves the values of the tag untouched (coupling not set yet). See [`Prescription`](@ref).
"""
mutable struct Coupled{G} <: Prescription
    g::G
    mask::Union{Nothing,Vector{Bool}}   # Dirichlet dofs of the tag (set at first use for the space)
    coords::Vector                      # coordinates of the nodes of those dofs
    ndofs::Int                          # dofs per node (components)
    cache::Any                          # Gridap evaluation cache (point search included)
    field::Any                          # the field the cache was built for
end
Coupled(g) = Coupled(g, nothing, [], 0, nothing, nothing)

(p::Coupled)(Λ) = x -> x                # placeholder for Gridap, overwritten by the mask

"""
    central_difference(f, Λ; h=1e-6)

Derivative of `f` at `Λ` by central differences. Default backend of [`get_time_derivative`](@ref).
"""
central_difference(f, Λ; h=1e-6) = (f(Λ + h) - f(Λ - h)) / (2h)

# Derivative of a prescription with respect to Λ, using the backend `d(f, Λ)`.
function _derivative(p::Separable, Λ, d)
    dg = d(λ -> _at_load(p.load, λ), Λ)
    if p.space isa Number
        p.space * dg
    else
        x -> p.space(x) * dg
    end
end

function _derivative(p::NonSeparable, Λ, d)
    x -> d(λ -> p(λ)(x), Λ)
end

function _derivative(p::Parametric, Λ, d)
    if p(Λ) isa Function
        x -> d(λ -> p(λ)(x), Λ)
    else
        d(p.g, Λ)  # Number and CellField support subtraction
    end
end

_derivative(p::Coupled, Λ, d) = d(p.g, Λ)   # as Parametric: CellFields support subtraction


#*******************************************************************************
#    					 Boundary conditions
#*******************************************************************************

"Activity window of a boundary condition. Only `Always` is implemented."
abstract type Activity end
struct Always <: Activity end

is_active(::Always, Λ) = true

"""
    BoundaryCondition(tag, value, load=nothing; mask=nothing)

Single boundary condition on the boundary `tag`. Whether it is applied as a Dirichlet
or a Neumann condition is decided by the user (or driver) consuming it.

- `value`: a `Prescription`, or the `space` argument of a `Separable` prescription.
- `load`: the `load` argument of a `Separable` prescription (only if `value` is not a `Prescription`).
- `mask`: components to be constrained, e.g. `(true, false, true)`. Default: all components.
"""
struct BoundaryCondition{P<:Prescription,M,A<:Activity}
    tag::String
    value::P
    mask::M
    activity::A
    function BoundaryCondition(tag::String, value::Prescription; mask=nothing, activity::Activity=Always())
        m = isnothing(mask) ? nothing : Tuple(Bool.(mask))
        new{typeof(value),typeof(m),typeof(activity)}(tag, value, m, activity)
    end
end

BoundaryCondition(tag::String, value, load=nothing; kwargs...) = BoundaryCondition(tag, Separable(value, load); kwargs...)

get_tags(bc::BoundaryCondition) = bc.tag
get_masks(bc::BoundaryCondition) = bc.mask
get_space_functions(bc::BoundaryCondition, Λ::Real) = bc.value(Λ)
get_time_derivative(bc::BoundaryCondition, Λ::Real; backend=central_difference) = _derivative(bc.value, Λ, backend)
is_active(bc::BoundaryCondition, Λ) = is_active(bc.activity, Λ)


"""
    BoundaryConditions(conditions...)

Collection of boundary conditions applied to a single field.
"""
struct BoundaryConditions{C<:Tuple}
    conditions::C
    BoundaryConditions(conditions::Tuple{Vararg{BoundaryCondition}}) = new{typeof(conditions)}(conditions)
end

BoundaryConditions(conditions::BoundaryCondition...) = BoundaryConditions(conditions)
BoundaryConditions(conditions::AbstractVector{<:BoundaryCondition}) = BoundaryConditions(Tuple(conditions))

"Empty collection of boundary conditions."
const NothingBC = BoundaryConditions{Tuple{}}
NothingBC() = BoundaryConditions(())

Base.length(bc::BoundaryConditions) = length(bc.conditions)
Base.isempty(bc::BoundaryConditions) = isempty(bc.conditions)
Base.iterate(bc::BoundaryConditions, args...) = iterate(bc.conditions, args...)
Base.getindex(bc::BoundaryConditions, i) = bc.conditions[i]

"""
    get_tags(bc::BoundaryConditions)

Get the strings identifying the regions where each boundary condition is applied.
"""
get_tags(bc::BoundaryConditions) = String[get_tags(c) for c in bc.conditions]

"""
    get_masks(bc::BoundaryConditions)

Masks in the format of the `dirichlet_masks` keyword of Gridap: `nothing` when no
condition is masked, otherwise one `Vector{Bool}` per tag.
"""
function get_masks(bc::BoundaryConditions)
    masks = map(get_masks, bc.conditions)
    i = findfirst(!isnothing, masks)
    isnothing(i) && return nothing
    ncomp = length(masks[i])  # At least, one BC has been prescribed by the user
    [isnothing(m) ? fill(true, ncomp) : collect(m) for m in masks]
end

"""
    get_space_functions(bc::BoundaryConditions, Λ::Real)

Values prescribed at `Λ`, one per condition, in the format accepted by Gridap:
a constant (`Real` or `VectorValue`, both `Number`s), a function of the
coordinates `x -> value`, or a Gridap object such as a `CellField` or an `Interpolable`.
"""
get_space_functions(bc::BoundaryConditions, Λ::Real) = map(c -> get_space_functions(c, Λ), bc.conditions)

_coupled_field(f) = f
_coupled_field(f::Gridap.CellData.Interpolable) = f.uh

"Mask and node coordinates of the Dirichlet dofs of `tag` (index in the space's tags) in `space`."
function _coupling_nodes!(p::Coupled, space::TrialFESpace, tag::Int)
    mask = get_dirichlet_dof_tag(space) .== tag
    # Dirichlet values of the identity = node coordinates (nested TrialFESpaces, e.g. ∆U of the drivers, included)
    coords_all = compute_dirichlet_values_for_tags(space, [x -> x for _ in 1:num_dirichlet_tags(space)])
    dim = num_point_dims(get_triangulation(space))
    vals = coords_all[mask]
    p.ndofs = dim                                            # vector-valued fields: dim components per node
    p.coords = [VectorValue(vals[i:i+dim-1]...) for i in 1:dim:length(vals)]
    p.mask = mask
    p
end

"""
    apply_coupling!(space, p::Coupled, tag, Λ)

Write `p.g(Λ)` at the Dirichlet nodes of `tag` (its index among the space's Dirichlet tags) into
`space`, after Gridap's `TrialFESpace!`. The field is evaluated only at those nodes, with a cached
point search, so it may live only on the faces of the tag.
"""
function apply_coupling!(space::TrialFESpace, p::Coupled, tag::Int, Λ)
    f = p.g(Λ)
    f === nothing && return space
    p.mask === nothing && _coupling_nodes!(p, space, tag)
    fld = _coupled_field(f)
    if p.cache === nothing || p.field !== fld
        p.cache = Gridap.Arrays.return_cache(fld, p.coords[1])
        p.field = fld
    end
    dv = get_dirichlet_dof_values(space)
    idx = findall(p.mask)
    k = 0
    for x in p.coords
        v = Gridap.Arrays.evaluate!(p.cache, fld, x)
        for c in 1:p.ndofs
            dv[idx[k+c]] = v[c]
        end
        k += p.ndofs
    end
    space
end

"Apply every `Coupled` condition of `bc` to `space` (after Gridap's `TrialFESpace!`)."
function apply_couplings!(space::TrialFESpace, bc::BoundaryConditions, Λ)
    for (i, c) in enumerate(bc.conditions)
        c.value isa Coupled && apply_coupling!(space, c.value, i, Λ)
    end
    space
end

"""
    get_time_derivative(bc, Λ; backend=central_difference)

Derivative of the prescribed values with respect to `Λ`, in the same format as `get_space_functions`.
`backend(f, Λ)` differentiates a function of `Λ`. `CellField`s returned by a `Parametric` prescription
require a backend supporting them, such as the default `central_difference`.
"""
get_time_derivative(bc::BoundaryConditions, Λ::Real; kwargs...) = map(c -> get_time_derivative(c, Λ; kwargs...), bc.conditions)


"""
    MultiFieldBoundaryConditions(fields...)

One `BoundaryConditions` per field of a multi-field problem.
"""
struct MultiFieldBoundaryConditions{B<:Tuple}
    fields::B
    MultiFieldBoundaryConditions(fields::Tuple{Vararg{BoundaryConditions}}) = new{typeof(fields)}(fields)
end

MultiFieldBoundaryConditions(fields::BoundaryConditions...) = MultiFieldBoundaryConditions(fields)
MultiFieldBoundaryConditions(fields::AbstractVector) = MultiFieldBoundaryConditions(Tuple(fields))

const MultiFieldBC = MultiFieldBoundaryConditions

Base.length(bc::MultiFieldBoundaryConditions) = length(bc.fields)
Base.iterate(bc::MultiFieldBoundaryConditions, args...) = iterate(bc.fields, args...)
Base.getindex(bc::MultiFieldBoundaryConditions, i) = bc.fields[i]


#*******************************************************************************
#    					 Legacy constructors
#*******************************************************************************

"Load function of a legacy condition, replaceable in place by `updateBC!`."
mutable struct LegacyLoad
    f::Any
end

(l::LegacyLoad)(Λ) = l.f(Λ)

_legacy_condition(tag, value, timestep) = BoundaryCondition(tag, _legacy_prescription(value, timestep))

# `timestep === nothing` means `value` is a generator `Λ -> object`. Couplings ignore `timestep`.
function _legacy_prescription(value, timestep)
    value === DirichletCoupling && return Coupled(LegacyCoupling(nothing))
    isnothing(timestep) && return Parametric(value)
    Separable(value, LegacyLoad(timestep))
end

function _legacy_conditions(tags::Vector{String}, values, timesteps)
    @assert length(tags) == length(values) == length(timesteps)
    BoundaryConditions(map(_legacy_condition, tags, values, timesteps))
end

"""
    DirichletBC(tags, values, timesteps)

Legacy constructor, returning `BoundaryConditions`. `values[i]` is multiplied by `timesteps[i](Λ)`,
unless `timesteps[i] === nothing`; then `values[i]` is a generator `Λ -> object`.
"""
DirichletBC(tags::Vector{String}, values, timesteps) = _legacy_conditions(tags, values, timesteps)
DirichletBC(tags::Vector{String}, values) = _legacy_conditions(tags, values, [Λ -> 1.0 for _ in tags])

"Legacy constructor, returning `BoundaryConditions`. See `DirichletBC`."
NeumannBC(tags::Vector{String}, values, timesteps) = _legacy_conditions(tags, values, timesteps)

"""
    updateBC!(bc, values, timesteps)

Legacy interface for `StaggeredModel`: replace the load function of each condition built by
`DirichletBC` or `NeumannBC` with `timesteps[i]`. `values` is only checked for its length.
"""
function updateBC!(bc::BoundaryConditions, values, timesteps)
    @assert length(bc) == length(values) == length(timesteps)
    for (c, timestep) in zip(bc.conditions, timesteps)
        c.value isa Separable{<:Any,LegacyLoad} && (c.value.load.f = timestep)
    end
    bc
end

# Legacy field access by `StaggeredModel`: `bc.caches` is only used for its length.
Base.getproperty(bc::BoundaryConditions, s::Symbol) = s === :caches ? getfield(bc, :conditions) : getfield(bc, s)


#*******************************************************************************
#    					 Neumann conditions
#*******************************************************************************

"""
    residual_Neumann(...)::Function

Return the Neumann residual as a FUNCTION.
"""
function residual_Neumann(bc::BoundaryConditions, dΓ::Vector, Λ::Float64)
    v -> mapreduce((fi, dΓi) -> ∫(v ⋅ fi)dΓi, +, get_space_functions(bc, Λ), dΓ; init=DomainContribution())
end

function residual_Neumann(bc::BoundaryConditions, v, dΓ, Λ)
    mapreduce((fi, dΓi) -> ∫(-1.0 * (v ⋅ fi))dΓi, +, get_space_functions(bc, Λ), dΓ; init=DomainContribution())
end

function residual_Neumann(bc::BoundaryConditions, v, dΓ, Λ⁺, Λ⁻)
    f⁺ = get_space_functions(bc, Λ⁺)
    f⁻ = get_space_functions(bc, Λ⁻)
    mapreduce((fi⁺, fi⁻, dΓi) -> ∫(-0.5 * (v ⋅ fi⁺))dΓi + ∫(-0.5 * (v ⋅ fi⁻))dΓi, +, f⁺, f⁻, dΓ; init=DomainContribution())
end

"""
    get_Neumann_dΓ(...)::Vector{Gridap.CellData.GenericMeasure}

Return a collection of boundary triangulations at the specified Neumann boundaries.
"""
function get_Neumann_dΓ(model, bc::BoundaryConditions, degree::Int)
    all_Γ = map(tag -> BoundaryTriangulation(model, tags=tag), get_tags(bc))
    all_dΓ = map(Γi -> Measure(Γi, degree), all_Γ)
    all_dΓ
end

function get_Neumann_dΓ(model, bc::MultiFieldBoundaryConditions, degree::Int)
    [get_Neumann_dΓ(model, bc_i, degree) for bc_i in bc]
end


#*******************************************************************************
#    					 Dirichlet coupling
#*******************************************************************************

# Legacy couplings. New code can prescribe `Coupled(Λ -> Interpolable(uh))` instead.

"""
Placeholder of a legacy `DirichletCoupling`, filled in place by `InterpolableBC!` with a generator
`Λ -> field` (an `FEFunction`, `CellField` or `Interpolable`). Until then it yields `nothing` and the
Dirichlet values of its tag are left untouched.
"""
mutable struct LegacyCoupling
    Interpolable::Any
end

(c::LegacyCoupling)(Λ) = isnothing(c.Interpolable) ? nothing : c.Interpolable(Λ)

"""
    InterpolableBC!(U, bc, interface_tag, Interpolable)

Legacy interface: set the generator `Λ -> field` of the `DirichletCoupling` declared on `interface_tag`
in `bc`. Returns the `Coupled` prescription. The values of `U` are updated at the next `TrialFESpace!`.
"""
function InterpolableBC!(U::TrialFESpace, bc::BoundaryConditions, interface_tags::String, Interpolable)
    coupling = bc[findfirst(==(interface_tags), get_tags(bc))].value
    @assert coupling isa Coupled{LegacyCoupling} "Tag $interface_tags was not declared as a DirichletCoupling"
    coupling.g.Interpolable = Interpolable
    return coupling
end

"Legacy name of [`Coupled`](@ref) (kept for `InterpolableBC(U, bc, tag, Interpolable)` call sites)."
InterpolableBC(U::TrialFESpace, bc::BoundaryConditions, interface_tags::String, Interpolable) =
    InterpolableBC!(U, bc, interface_tags, Interpolable)
