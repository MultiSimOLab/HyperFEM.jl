abstract type AbstractPostProcessor end
get_pvd(::AbstractPostProcessor) = @abstractmethod
vtk_save(::AbstractPostProcessor) = @abstractmethod

include("PostMetrics.jl")


mutable struct PostProcessor{A,B,C} <:AbstractPostProcessor
    comp_model::A
    cache::B
    cachevtk::C
    iter::Int64
    Λ ::Vector{Float64}

    function PostProcessor() 
        cachevtk = (false, nothing, nothing)
        cache = ((x...)->())
        Λ = Vector{Float64}()
        A, B, C = typeof(nothing), typeof(cache), typeof(cachevtk)
        new{A,B,C}(nothing, cache, cachevtk, 0, Λ)
    end

    function PostProcessor(comp_model,driver;
        is_vtk=true,
        filepath=datadir("sims", "Temp"),
        kwargs...)

        pvd = paraview_collection(filepath * "/Results", append=false)
        cache = (driver, kwargs)
        cachevtk = (is_vtk, filepath, pvd)
        Λ = Vector{Float64}()

        A, B, C = typeof(comp_model), typeof(cache), typeof(cachevtk)
        new{A,B,C}(comp_model, cache, cachevtk, 0,Λ)
    end


end
get_pvd(p::PostProcessor{<:Any,<:Any,<:Any}) = p.cachevtk[3]
function vtk_save(p::PostProcessor{<:Any,<:Any,<:Any}) 
    if p.cachevtk[1]
        WriteVTK.vtk_save(get_pvd(p))
    end
end

function reset!(obj::PostProcessor)  
    obj.iter=0 
    obj.Λ = Vector{Float64}()
    is_vtk,filepath,pvd= obj.cachevtk
    isnothing(pvd) ? pvd =nothing : pvd = paraview_collection(filepath * "/Results", append=false)
    obj.cachevtk = (is_vtk, filepath, pvd)
end

function (obj::PostProcessor{<:Nothing,<:Any,<:Any})(Λ) end

function (obj::PostProcessor{<:StaticNonlinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

function (obj::PostProcessor{<:StaticLinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

function (obj::PostProcessor{<:DynamicNonlinearModel,<:Any,<:Any})(Λ)
    obj.iter +=1
    push!(obj.Λ, Λ)
    obj.cache[1](obj, obj.cache[2]...)
end

function Jacobian(uh,km)
  @warn "The function Jacobian is deprecated and it will be removed after release 0.0.7." maxlog=1
  F, _, J = get_Kinematics(km)
  J ∘ F ∘ ∇(uh)
end

function Piola(physmodel::ThermoElectroMechano, kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
  @warn "The function Piola is deprecated and it will be removed after release 0.0.7." maxlog=1
  DΨ = physmodel()
  F, _, _ = get_Kinematics(kine[1])
  E = get_Kinematics(kine[2])
  ∂Ψu = DΨ[2]
  σh = ∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Cauchy(args...)
  @warn "The function Cauchy is deprecated and it will be removed after release 0.0.7." maxlog=1
  Piola(args...)
end


function Piola(model::Elasto, km::KinematicModel, uh, unh, state_vars, Ω, dΩ, t=0)
  @warn "The function Piola is deprecated and it will be removed after release 0.0.7." maxlog=1
  σh = Piola(model,km,uh)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Piola(model::ViscoElastic, km::KinematicModel, uh, unh, state_vars, Ω, dΩ, t=0)
  @warn "The function Piola is deprecated and it will be removed after release 0.0.7." maxlog=1
  σh = Piola(model, km, uh, unh, state_vars)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Piola(model::Elasto, km::KinematicModel, uh, vars...)
  @warn "The function Piola is deprecated and it will be removed after release 0.0.7." maxlog=1
  _, ∂Ψu, _ = model()
  F, _, _ = get_Kinematics(km)
  ∂Ψu ∘ (F∘∇(uh))
end


function Piola(model::ViscoElastic,  km::KinematicModel, uh, unh, states)
  @warn "The function Piola is deprecated and it will be removed after release 0.0.7." maxlog=1
  _, ∂Ψu, _ = model()
  F, _, _ = get_Kinematics(km)
  ∂Ψu ∘ (F∘∇(uh), F∘∇(unh), states...)
end


function Entropy(physmodel::ThermoElectroMechano,  kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
  @warn "The function Entropy is deprecated and it will be removed after release 0.0.7." maxlog=1
  DΨ = physmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  η = -DΨ[4]
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  ref = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  V = FESpace(Ω, ref, conformity=:H1)
  ηh = interpolate_everywhere(L2_projection((η ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), VL2, dΩ), V)
  return ηh
end


function D0(physmodel::ThermoElectroMechano,  kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
  @warn "The function D0 is deprecated and it will be removed after release 0.0.7." maxlog=1
  DΨ = physmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  ∂ΨE = DΨ[3]
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  ref = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  V = FESpace(Ω, ref, conformity=:H1)
  n1 = VectorValue(-1.0, 0.0, 0.0)
  n2 = VectorValue(0.0, -1.0, 0.0)
  n3 = VectorValue(0.0, 0.0, -1.0)
  D0_1h = interpolate_everywhere(L2_projection(n1 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), VL2, dΩ), V)
  D0_2h = interpolate_everywhere(L2_projection(n2 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), VL2, dΩ), V)
  D0_3h = interpolate_everywhere(L2_projection(n3 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), VL2, dΩ), V)
  return (D0_1h,D0_2h,D0_3h)
end


function interpolate_L2_tensor(A, Ω, dΩ, Γ=Ω)
  @warn "The function interpolate_L2_tensor is deprecated and it will be removed after release 0.0.7. Use interpolate_L2_field." maxlog=1
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  n1 = VectorValue(1.0, 0.0, 0.0)
  n2 = VectorValue(0.0, 1.0, 0.0)
  n3 = VectorValue(0.0, 0.0, 1.0)
  A11 = interpolate_everywhere(L2_projection(n1 ⋅ A ⋅ n1, VL2, dΩ), VH1)
  A12 = interpolate_everywhere(L2_projection(n1 ⋅ A ⋅ n2, VL2, dΩ), VH1)
  A13 = interpolate_everywhere(L2_projection(n1 ⋅ A ⋅ n3, VL2, dΩ), VH1)
  A22 = interpolate_everywhere(L2_projection(n2 ⋅ A ⋅ n2, VL2, dΩ), VH1)
  A23 = interpolate_everywhere(L2_projection(n2 ⋅ A ⋅ n3, VL2, dΩ), VH1)
  A33 = interpolate_everywhere(L2_projection(n3 ⋅ A ⋅ n3, VL2, dΩ), VH1)
  trA = interpolate_everywhere(L2_projection(tr ∘ A,     VL2, dΩ), VH1)
  (A11, A12, A13, A22, A23, A33, trA)
end


function interpolate_L2_vector(b, Ω, dΩ, Γ=Ω)
  @warn "The function interpolate_L2_vector is deprecated and it will be removed after release 0.0.7. Use interpolate_L2_field." maxlog=1
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  n1 = VectorValue(1.0, 0.0, 0.0)
  n2 = VectorValue(0.0, 1.0, 0.0)
  n3 = VectorValue(0.0, 0.0, 1.0)
  b1 = interpolate_everywhere(L2_projection(n1 ⋅ b, VL2, dΩ), VH1)
  b2 = interpolate_everywhere(L2_projection(n2 ⋅ b, VL2, dΩ), VH1)
  b3 = interpolate_everywhere(L2_projection(n3 ⋅ b, VL2, dΩ), VH1)
  (b1, b2, b3)
end


function interpolate_L2_scalar(x, Ω, dΩ, Γ=Ω)
  @warn "The function interpolate_L2_scalar is deprecated and it will be removed after release 0.0.7. Use interpolate_L2_field." maxlog=1
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  interpolate_everywhere(L2_projection(x, VL2, dΩ), VH1)
end


function get_cell_field_type(f::CellField, dΩ::Measure)
  pts = get_cell_points(dΩ)
  f_x = f(pts)
  T = eltype(Gridap.Arrays.testitem(f_x))
  return T
end


"""
    interpolate_L2_field(x, dΩ, [Ω₂], [T])

Interpolate an L2 field into an H1 field.
The field `x` is first projected onto a piecewise-constant space defined over the
triangulation of the measure `dΩ`, and then interpolated into a continuous space.
The type of the field is inferred from the cell field `x` and the measure `dΩ`.
Optionally, a destination triangulation `Ω₂` can be specified, being different
from the triangulation of `dΩ`.
"""
function interpolate_L2_field(x, dΩ::Measure, Ω₂::Triangulation=get_triangulation(Gridap.CellData.get_cell_quadrature(dΩ)), T::Type=get_cell_field_type(x, dΩ))
  Ω₁ = get_triangulation(Gridap.CellData.get_cell_quadrature(dΩ))
  refL2 = ReferenceFE(lagrangian, T, 0)
  reffe = ReferenceFE(lagrangian, T, 1)
  VL2 = FESpace(Ω₁, refL2, conformity=:L2)
  VH1 = FESpace(Ω₂, reffe, conformity=:H1)
  interpolate_everywhere(L2_projection(x, VL2, dΩ), VH1)
end

function interpolate_L2_field(x, Ω, dΩ::Measure, T::Type=get_cell_field_type(x, dΩ))
  @warn "This signature is kept for direct compatibility with `interpolate_L2_scalar/vector/tensor` and it will be removed after release 0.0.7. The domain `Ω` can be removed from the list of arguments." maxlog=1
  refL2 = ReferenceFE(lagrangian, T, 0)
  reffe = ReferenceFE(lagrangian, T, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Ω, reffe, conformity=:H1)
  interpolate_everywhere(L2_projection(x, VL2, dΩ), VH1)
end


"""
    L2_projection(u, V, dΩ)

Perform an L2 projection of a field `u` onto a finite element
space `V` defined over the domain `Ω` with measure `dΩ`.
"""
function L2_projection(u, V, dΩ::Measure)
  a(w, v) = ∫(w ⊙ v) * dΩ
  l(v)    = ∫(v ⊙ u) * dΩ
  op      = AffineFEOperator(a, l, V, V)
  solve(op)
end

function L2_projection(u, dΩ::Measure, V)
  @warn "The signature L2_projection(u, dΩ, V) is deprecated and it will be removed after release 0.0.7. Use L2_projection(u, V, dΩ)." maxlog=1
  L2_projection(u, V, dΩ)
end

function L2_Projection(u, dΩ::Measure, V)
  @warn "The function L2_Projection is deprecated and it will be removed after release 0.0.7. Use L2_projection(u, V, dΩ) (lowercase, idiomatic Julia)." maxlog=1
  L2_projection(u, V, dΩ)
end


"""
    component_LInf(u, direction, [Ω])::Float64

Calculate the maximum absolute value of a component of a vector-valued field `u`.
The `direction` can be a symbol (`:x`, `:y` or `:z`), a component index or a `VectorValue`.
The field is evaluated at the nodes of a first order discontinuous space over `Ω`,
so for higher order fields this is an approximation of the L-inf norm.
By default, Ω is the triangulation where the function is defined.
It could be useful to find the maximum displacement.

# Examples
    x_max = component_LInf(uh, :x)
    y_max = component_LInf(uh, 2)
    d_max = component_LInf(uh, VectorValue(1.0, 1.0, 0.0) / √2)
"""
function component_LInf(u, n::VectorValue, Ω::Triangulation=get_triangulation(u))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = FESpace(Ω, reffe, conformity=:L2)
  un = interpolate_everywhere(u ⋅ n, V)
  norm(get_free_dof_values(un), Inf)
end

function component_LInf(u, i::Integer, Ω::Triangulation=get_triangulation(u))
  D = num_point_dims(Ω)
  1 ≤ i ≤ D || throw(ArgumentError("Component must be in 1:$D. Got $i"))
  n = VectorValue(ntuple(j -> Float64(j == i), D))
  component_LInf(u, n, Ω)
end

function component_LInf(u, direction::Symbol, Ω::Triangulation=get_triangulation(u))
  i = findfirst(==(direction), (:x, :y, :z))
  isnothing(i) && throw(ArgumentError("Direction must be either :x, :y or :z. Got $direction"))
  component_LInf(u, i, Ω)
end
