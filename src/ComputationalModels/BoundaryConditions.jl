
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

_to_value(v::Number) = v
_to_value(v::AbstractVector{<:Real}) = VectorValue(v...)
_to_value(f) = f  # Any callable accepting the coordinates `x`

"""
    Separable(space, load=nothing)

Prescribed value `space(x) * load(Λ)`. `space` is either a constant (`Number`, `VectorValue`,
`AbstractVector`) or a callable `x -> value`. `load === nothing` stands for `Λ -> 1`.
See [`Prescription`](@ref).
"""
struct Separable{S,L} <: Prescription
    space::S
    load::L
    Separable(space, load=nothing) = new(_to_value(space), load)
end

_at_load(::Nothing, Λ) = 1.0
_at_load(f, Λ) = f(Λ)

(p::Separable{<:Number})(Λ) = p.space * _at_load(p.load, Λ)

function (p::Separable)(Λ)
    g = _at_load(p.load, Λ)
    x -> _to_value(p.space(x)) * g
end

"""
    NonSeparable(f)

Prescribed value `f(x, Λ)`, for prescriptions such as a rigid rotation.
See ['Prescription'](@ref)
"""
struct NonSeparable{F} <: Prescription
    f::F
end

(p::NonSeparable)(Λ) = x -> _to_value(p.f(x, Λ))

"""
    Parametric(g)

Prescribed object `g(Λ)`, which can be anything accepted by Gridap for a single tag
(e.g. a `CellField` or a function of `x`). See [`Prescription`](@ref)
"""
struct Parametric{G} <: Prescription
    g::G
end

(p::Parametric)(Λ) = p.g(Λ)


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
    ncomp = length(masks[i])
    [isnothing(m) ? fill(true, ncomp) : collect(m) for m in masks]
end

get_space_functions(bc::BoundaryConditions, Λ::Real) = map(c -> get_space_functions(c, Λ), bc.conditions)


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

# `timestep === nothing` means `value` is a generator `Λ -> object`.
_legacy_condition(tag, value, ::Nothing) = BoundaryCondition(tag, Parametric(value))
_legacy_condition(tag, value, timestep) = BoundaryCondition(tag, Separable(value, timestep))

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
DirichletBC(tags::Vector{String}, values) = BoundaryConditions(map(BoundaryCondition, tags, values))

"Legacy constructor, returning `BoundaryConditions`. See `DirichletBC`."
NeumannBC(tags::Vector{String}, values, timesteps) = _legacy_conditions(tags, values, timesteps)


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

# TODO: Port to a `Parametric` prescription. Requires `DirichletBC` caches, which no longer exist.
struct InterpolableBC{A,B,C} <: DirichletCoupling
    coords::A
    Interpolable::B
    caches::C
    function InterpolableBC(U::TrialFESpace, bc::BoundaryConditions, interface_tags::String, Interpolable::B) where {B}
        dcmask = findall(x -> x == interface_tags, get_tags(bc))
        dim = length(U.space.fe_dof_basis.trian.model.grid.node_coordinates[1])
        mask = U.space.dirichlet_dof_tag .== dcmask[1]
        vals = U.dirichlet_values[mask]
        Interface_coords_ = reshape(vals, dim, :)'
        coords = VectorValue.(eachrow(Interface_coords_))
        v = evaluate(Interpolable(1.0), coords)
        bc_values = reduce(vcat, map(x -> get_array(x), v))
        caches = (bc_values, mask)
        new{typeof(coords),B,typeof(caches)}(coords, Interpolable, caches)
    end
end

function (obj::InterpolableBC)(Λ::Float64=1.0)
    bc_values = obj.caches[1]
    bc_values .= reduce(vcat, map(x -> get_array(x), evaluate(obj.Interpolable(Λ), obj.coords)))
end

function InterpolableBC!(U::TrialFESpace, bc::BoundaryConditions, interface_tags::String, Interpolable)
    error("InterpolableBC! is not supported yet by BoundaryConditions. Use a `Parametric` prescription instead.")
end
