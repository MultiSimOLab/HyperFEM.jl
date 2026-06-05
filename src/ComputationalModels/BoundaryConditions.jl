

abstract type BoundaryCondition end
abstract type TimedependentCondition end
abstract type DirichletCoupling end

struct NothingBC <: BoundaryCondition end

struct MultiFieldBC <: BoundaryCondition
    BoundaryCondition::Vector{BoundaryCondition}
end

getindex(bc::MultiFieldBC, i) = bc.BoundaryCondition[i]


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

function ϝ(v::Float64)
    (x) -> v
end

function ϝ(v::Vector{Float64})
    (x) -> VectorValue(v)
end

function ϝ(v::Function)
    (x) -> v(x)
end

function _get_bc_func(tags_::Vector{String}, values_, bc_timesteps)
    bc_func_ = Vector{Function}(undef, length(tags_))
    @assert(length(tags_) == length(values_))

    @inbounds for i in eachindex(tags_)

        if values_[i] === DirichletCoupling || typeof(values_[i]) <: DirichletCoupling
            bc_func_[i] = (Λ) -> (x) -> x
        else
            if isnothing(bc_timesteps[i])
                bc_func_[i] = values_[i]
            else
                # get funcion generators for boundary conditions
                u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * bc_timesteps[i](Λ)
                bc_func_[i] = u_bc
            end
        end

    end
    return (bc_tags=tags_, bc_func=bc_func_,)
end




struct DirichletBC{A} <: BoundaryCondition
    tags::Vector{String}         # tags for boundary conditions
    values::Vector{Union{Function,DirichletCoupling}}     # f(x)
    timesteps::Vector{Union{Function,Nothing}}  # f(Λ)
    caches::A

    function DirichletBC(bc_tags::Vector{String}, bc_values, bc_timesteps)
        @assert(length(bc_tags) == length(bc_values) == length(bc_timesteps))
        tags_, funcs_ = _get_bc_func(bc_tags, bc_values, bc_timesteps)
        caches = (bc_values)
        new{typeof(caches)}(tags_, funcs_, bc_timesteps, caches)
    end


    function DirichletBC(bc_tags::Vector{String}, bc_values)
        @assert(length(bc_tags) == length(bc_values))
        bc_timesteps = [Λ -> 1.0 for _ in 1:length(bc_tags)]
        tags_, funcs_ = _get_bc_func(bc_tags, bc_values, bc_timesteps)
        caches = (bc_values)
        new{typeof(caches)}(tags_, funcs_, bc_timesteps, caches)
    end

end

function updateBC!(m::DirichletBC, bc_values, bc_timesteps)
    @assert(length(m.tags) == length(bc_values) == length(bc_timesteps))
    _, newfuncs = _get_bc_func(m.tags, bc_values, bc_timesteps)
    m.values .= newfuncs
    m.timesteps .= bc_timesteps
end

function updateBC!(m::DirichletBC, bc_values)
    @assert(length(m.tags) == length(bc_values))
    _, newfuncs = _get_bc_func(m.tags, bc_values, m.timesteps)
    m.values .= newfuncs
end

function updateBC!(m::DirichletBC, bc_timesteps::Function)
    _, newfuncs = _get_bc_func(m.tags, m.caches, [bc_timesteps for _ in 1:length(m.caches)])
    m.values .= newfuncs
end

function updateBC!(bc::MultiFieldBC, bc_timesteps::Function)
    for (i, bc_i) in enumerate(bc.BoundaryCondition)
        bc_i isa NothingBC ? nothing : updateBC!(bc_i, bc_i.caches, [bc_timesteps for _ in 1:length(bc_i.caches)])
    end
end


# function updateBC!(m::DirichletBC; interface_tags::String ,  interface_values::InterpolableBC)
#     mask = findall(x -> x == interface_tags, m.tags)
#     m.caches[mask[1]] = interface_values
# end



struct NeumannBC <: BoundaryCondition
    tags::Vector{String}         # tags for boundary conditions
    values::Vector{Function}     # f(x)
    timesteps::Vector{Function}  # f(Λ)

    function NeumannBC(bc_tags::Vector{String}, bc_values, bc_timesteps)
        @assert(length(bc_tags) == length(bc_values) == length(bc_timesteps))
        tags_, funcs_ = _get_bc_func(bc_tags, bc_values, bc_timesteps)
        new(tags_, funcs_, bc_timesteps)
    end
end

function residual_Neumann(::NothingBC, kwargs...) end

"""
    residual_Neumann(...)::Function

Return the Neumann residual as a FUNCTION.
"""
function residual_Neumann(bc::NeumannBC, dΓ::Vector, Λ::Float64)
    v -> mapreduce((fi, dΓi) -> ∫(v ⋅ fi(Λ))dΓi, +, bc.values, dΓ)
end

function residual_Neumann(bc::NeumannBC, v, dΓ, Λ)
    bc_func_ = Vector{Function}(undef, length(bc.tags))
    for (i, f) in enumerate(bc.values)
        bc_func_[i] = (v) -> ∫(-1.0 * (v ⋅ f(Λ)))dΓ[i]
    end
    return mapreduce(f -> f(v), +, bc_func_)
end

function residual_Neumann(bc::NeumannBC, v, dΓ, Λ⁺, Λ⁻)
    bc_func_ = Vector{Function}(undef, length(bc.tags))
    for (i, f) in enumerate(bc.values)
        bc_func_[i] = (v) -> (∫(-0.5 * (v ⋅ f(Λ⁺)))dΓ[i] + ∫(-0.5 * (v ⋅ f(Λ⁻)))dΓ[i])
    end
    return mapreduce(f -> f(v), +, bc_func_)
end

"""
    get_Neumann_dΓ(...)::Vector{Gridap.CellData.GenericMeasure}

Return a collection of boundary triangulations at the specified Neumann boundaries.
"""
function get_Neumann_dΓ(model, ::NothingBC, degree::Int)
    Vector{Gridap.CellData.GenericMeasure}(undef, 1)
end

function get_Neumann_dΓ(model, bc::NeumannBC, degree::Int)
    all_Γ = map(tag -> BoundaryTriangulation(model, tags=tag), bc.tags)
    all_dΓ = map(Γi -> Measure(Γi, degree), all_Γ)
    all_dΓ
end

function get_Neumann_dΓ(model, bc::MultiFieldBC, degree::Int)
    dΓ = Vector{Vector{Gridap.CellData.GenericMeasure}}(undef, length(bc.BoundaryCondition))
    for (i, bc_i) in enumerate(bc.BoundaryCondition)
        dΓ[i] = get_Neumann_dΓ(model, bc_i, degree)
    end
    return dΓ
end





struct InterpolableBC{A,B,C} <: DirichletCoupling
    coords::A
    Interpolable::B
    caches::C
    function InterpolableBC(U::TrialFESpace, bc::DirichletBC, interface_tags::String, Interpolable::B) where {B}
        dcmask = findall(x -> x == interface_tags, bc.tags)
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


function InterpolableBC!(U::TrialFESpace, bc::DirichletBC, interface_tags::String, Interpolable)
    obj = InterpolableBC(U, bc, interface_tags, Interpolable)
    dcmask = findall(x -> x == interface_tags, bc.tags)
    bc.caches[dcmask[1]] = obj
    return obj
end

