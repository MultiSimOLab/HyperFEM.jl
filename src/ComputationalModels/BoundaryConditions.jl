
#*******************************************************************************
#    					 Prescriptions
#*******************************************************************************

"""
    Prescription

What is prescribed on a boundary. `p(Λ, τ)` returns an object accepted by Gridap for a single tag:
a `Number`/`VectorValue`, a function of the coordinates `x -> value`, or a `CellField`.

Two parameters are passed by the drivers:
- `Λ`, the **load**: the parameter of the loading programme of the whole simulation. `Separable` and
  `NonSeparable` prescribe values as functions of it.
- `τ`, the **stage**: the parameter of the solver, the one `res(τ)` and `jac(τ)` receive. A `Coupling`
  prescribes the state of another problem as a function of it.

They coincide in a static or monolithic solve; in a staggered loop `Λ` runs over the ramp of the step
while `τ` runs from 0 to 1 inside it (see `StaggeredModel`). `p(Λ)` is `p(Λ, Λ)`.
"""
abstract type Prescription end

(p::Prescription)(Λ) = p(Λ, Λ)

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

(p::Separable{<:Number})(Λ, τ) = p.space * _at_load(p.load, Λ)

(p::Separable)(Λ, τ) = x -> p.space(x) * _at_load(p.load, Λ)

"""
    NonSeparable(f)

Prescribed value `f(x, Λ)`, for prescriptions such as a rigid rotation.
See [`Prescription`](@ref)
"""
struct NonSeparable{F} <: Prescription
    f::F
end

(p::NonSeparable)(Λ, τ) = x -> _to_number(p.f(x, Λ))

"""
    Coupling(g)

The state of another problem prescribed on a boundary: `g(τ)` returns a field (`FEFunction`, `CellField`,
`Interpolable`) or a function of `x`, evaluated at the stage parameter `τ` of the solver, the same one
`res(τ)` receives. In a staggered loop `g` typically interpolates between the previous and the current
stage of the other field, `τ -> uh⁻ + (uh⁺ - uh⁻) * τ`, and the field may live only on the faces of the
tag (an `FESpace` on an interface). See [`Prescription`](@ref).
"""
struct Coupling{G} <: Prescription
    g::G
end

(p::Coupling)(Λ, τ) = p.g(τ)

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

# A coupling depends on the stage parameter only
function _derivative(p::Coupling, τ, d)
    if p.g(τ) isa Function
        x -> d(t -> p.g(t)(x), τ)
    else
        d(p.g, τ)  # Number and CellField support subtraction
    end
end

_parameter(::Prescription, Λ, τ) = Λ
_parameter(::Coupling, Λ, τ) = τ



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
get_space_functions(bc::BoundaryCondition, Λ::Real, τ::Real=Λ) = bc.value(Λ, τ)
get_time_derivative(bc::BoundaryCondition, Λ::Real, τ::Real=Λ; backend=central_difference) =
    _derivative(bc.value, _parameter(bc.value, Λ, τ), backend)
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
    get_space_functions(bc::BoundaryConditions, Λ::Real, τ::Real=Λ)

Values prescribed at the load `Λ` and the stage `τ` (see [`Prescription`](@ref)), one per condition, in
the format accepted by Gridap: a constant (`Real` or `VectorValue`, both `Number`s), a function of the
coordinates `x -> value`, or a Gridap object such as a `CellField` or an `Interpolable`.
"""
get_space_functions(bc::BoundaryConditions, Λ::Real, τ::Real=Λ) = map(c -> get_space_functions(c, Λ, τ), bc.conditions)

"""
    get_time_derivative(bc, Λ, τ=Λ; backend=central_difference)

Derivative of the prescribed values with respect to their parameter (`Λ` for `Separable` and
`NonSeparable`, `τ` for `Coupling`), in the same format as `get_space_functions`. `backend(f, Λ)`
differentiates a function of one parameter. `CellField`s returned by a `Coupling` require a backend
supporting them, such as the default `central_difference`.
"""
get_time_derivative(bc::BoundaryConditions, Λ::Real, τ::Real=Λ; kwargs...) =
    map(c -> get_time_derivative(c, Λ, τ; kwargs...), bc.conditions)


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
#    					 Neumann conditions
#*******************************************************************************

"""
    residual_Neumann(bc, v, dΓ, Λ)

Neumann contribution `-∑ ∫(v ⋅ fᵢ) dΓᵢ` to the residual at the load `Λ`, one term per condition of `bc`
(`dΓ` from [`get_Neumann_dΓ`](@ref)). Inside a staggered loop the load is not the parameter of `res(τ)`:
give the conditions to the model (`neumann = NeumannConditions(model, bc, degree)`) and the driver adds
this term at the right load.
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

"""
    NeumannConditions(model, bc, degree)

Neumann conditions of a field with their boundary measures, to be given to a computational model
(`neumann=`), which adds `residual_Neumann` to its residual at the load of every step.
"""
struct NeumannConditions{B<:BoundaryConditions,D}
    bc::B
    dΓ::D
end

NeumannConditions(model, bc::BoundaryConditions, degree::Int) = NeumannConditions(bc, get_Neumann_dΓ(model, bc, degree))

residual_Neumann(n::NeumannConditions, v, Λ) = residual_Neumann(n.bc, v, n.dΓ, Λ)

"Residual function `res` with the Neumann term of `n` at the load `Λ` added (`n === nothing`: `res`)."
with_neumann(res, ::Nothing, Λ) = res
with_neumann(res, n::NeumannConditions, Λ) = (u, v) -> res(u, v) + residual_Neumann(n, v, Λ)

"Energy function `e` with the work of the Neumann loads of `n` at the load `Λ` added, `-∑ ∫(u ⋅ fᵢ) dΓᵢ`."
with_neumann_energy(e, ::Nothing, Λ) = e
with_neumann_energy(e, n::NeumannConditions, Λ) = u -> e(u) + residual_Neumann(n, u, Λ)
