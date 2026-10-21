
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
    applicators::IdDict{Any,Any}     # DirichletApplicator per space (keyed by its Dirichlet values vector)
    loadmap::Base.RefValue{Any}      # Λ -> Λ' applied before the prescriptions (see `set_load_map!`)
    BoundaryConditions(conditions::Tuple{Vararg{BoundaryCondition}}) =
        new{typeof(conditions)}(conditions, IdDict{Any,Any}(), Ref{Any}(identity))
end

"""
    set_load_map!(bc, f)

Make the `load` of the `Separable` prescriptions of `bc` see `f(Λ)` instead of `Λ` (in `TrialFESpace!`,
`get_space_functions`, `residual_Neumann`). Used by `StaggeredModel` to map the load parameter of each
inner solve to the global one, composing with the loads instead of replacing them (as the former
`updateBC!` did). The other prescriptions (`NonSeparable`, `Parametric`, e.g. a coupling interpolating
between two staggered stages) keep receiving the parameter of the inner solve. `identity` resets.
"""
set_load_map!(bc::BoundaryConditions, f) = (bc.loadmap[] = f; bc)
load_map(bc::BoundaryConditions, Λ) = bc.loadmap[](Λ)
"Load parameter seen by a condition: mapped for `Separable` prescriptions, raw otherwise."
_load_of(c::BoundaryCondition, bc::BoundaryConditions, Λ) = c.value isa Separable ? load_map(bc, Λ) : Λ

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
get_space_functions(bc::BoundaryConditions, Λ::Real) = map(c -> get_space_functions(c, _load_of(c, bc, Λ)), bc.conditions)

"""
    get_time_derivative(bc, Λ; backend=central_difference)

Derivative of the prescribed values with respect to `Λ`, in the same format as `get_space_functions`.
`backend(f, Λ)` differentiates a function of `Λ`. `CellField`s returned by a `Parametric` prescription
require a backend supporting them, such as the default `central_difference`.
"""
get_time_derivative(bc::BoundaryConditions, Λ::Real; kwargs...) = map(c -> get_time_derivative(c, _load_of(c, bc, Λ); kwargs...), bc.conditions)   # w.r.t. the (mapped) parameter


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

set_load_map!(bc::MultiFieldBoundaryConditions, f) = (foreach(b -> set_load_map!(b, f), bc.fields); bc)

Base.length(bc::MultiFieldBoundaryConditions) = length(bc.fields)
Base.iterate(bc::MultiFieldBoundaryConditions, args...) = iterate(bc.fields, args...)
Base.getindex(bc::MultiFieldBoundaryConditions, i) = bc.fields[i]


#*******************************************************************************
#    					 Legacy constructors
#*******************************************************************************

_legacy_condition(tag, value, timestep) = BoundaryCondition(tag, _legacy_prescription(value, timestep))

# `timestep === nothing` means `value` is a generator `Λ -> object`. Couplings ignore `timestep`.
function _legacy_prescription(value, timestep)
    value === DirichletCoupling && return Parametric(LegacyCoupling(nothing))
    isnothing(timestep) && return Parametric(value)
    Separable(value, timestep)
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

# Legacy couplings. New code prescribes the field directly: `Parametric(Λ -> uh)` (or an `Interpolable`).

"""
Placeholder of a legacy `DirichletCoupling`, filled in place by `InterpolableBC!` with a generator
`Λ -> field` (an `FEFunction`, `CellField` or `Interpolable`). Until then it yields `nothing` and the
Dirichlet values of its tag are left untouched. Deprecated: prescribe the field directly,
`BoundaryCondition(tag, Parametric(Λ -> field))`.
"""
mutable struct LegacyCoupling
    Interpolable::Any
end

(c::LegacyCoupling)(Λ) = isnothing(c.Interpolable) ? nothing : c.Interpolable(Λ)

"""
    InterpolableBC!(U, bc, interface_tag, Interpolable)

Deprecated: set the generator `Λ -> field` of the `DirichletCoupling` declared on `interface_tag` in
`bc`. Prescribe the field directly instead: `BoundaryCondition(interface_tag, Parametric(Λ -> field))`.
"""
function InterpolableBC!(U::TrialFESpace, bc::BoundaryConditions, interface_tags::String, Interpolable)
    Base.depwarn("`DirichletCoupling` + `InterpolableBC!` are deprecated: prescribe the field directly with " *
                 "`BoundaryCondition(\"$interface_tags\", Parametric(Λ -> field))`", :InterpolableBC!)
    coupling = bc[findfirst(==(interface_tags), get_tags(bc))].value
    @assert coupling isa Parametric{LegacyCoupling} "Tag $interface_tags was not declared as a DirichletCoupling"
    coupling.g.Interpolable = Interpolable
    return coupling
end
