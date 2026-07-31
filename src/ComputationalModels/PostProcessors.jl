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
  F, _, J = get_Kinematics(km)
  J ∘ F ∘ ∇(uh)
end

function Piola(physmodel::ThermoElectroMechano, kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
  DΨ = physmodel()
  F, _, _ = get_Kinematics(kine[1])
  E = get_Kinematics(kine[2])
  ∂Ψu = DΨ[2]
  σh = ∂Ψu ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Cauchy(args...)
  @warn "The function Cauchy is deprecated. Shortly it'll be J^-1*P*F^-T."
  Piola(args...)
end


function Piola(model::Elasto, km::KinematicModel, uh, unh, state_vars, Ω, dΩ, t=0)
  σh = Piola(model,km,uh)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Piola(model::ViscoElastic, km::KinematicModel, uh, unh, state_vars, Ω, dΩ, t=0)
  σh = Piola(model, km, uh, unh, state_vars)
  interpolate_L2_tensor(σh, Ω, dΩ)
end


function Piola(model::Elasto, km::KinematicModel, uh, vars...)
  _, ∂Ψu, _ = model()
  F, _, _ = get_Kinematics(km)
  ∂Ψu ∘ (F∘∇(uh))
end


function Piola(model::ViscoElastic,  km::KinematicModel, uh, unh, states)
  _, ∂Ψu, _ = model()
  F, _, _ = get_Kinematics(km)
  ∂Ψu ∘ (F∘∇(uh), F∘∇(unh), states...)
end


function Entropy(physmodel::ThermoElectroMechano,  kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
  DΨ = physmodel()
  F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
  E     = get_Kinematics(kine[2]; Λ=Λ)
  η = -DΨ[4]
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  ref = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  V = FESpace(Ω, ref, conformity=:H1)
  ηh = interpolate_everywhere(L2_Projection((η ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
  return ηh
end


function D0(physmodel::ThermoElectroMechano,  kine::NTuple{3,KinematicModel}, uh, φh, θh, Ω, dΩ, Λ=1.0)
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
  D0_1h = interpolate_everywhere(L2_Projection(n1 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
  D0_2h = interpolate_everywhere(L2_Projection(n2 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
  D0_3h = interpolate_everywhere(L2_Projection(n3 ⋅(∂ΨE ∘ (F∘(∇(uh)'), E∘(∇(φh)), θh)), dΩ, VL2), V)
  return (D0_1h,D0_2h,D0_3h)
end


function interpolate_L2_tensor(A, Ω, dΩ, Γ=Ω)
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  n1 = VectorValue(1.0, 0.0, 0.0)
  n2 = VectorValue(0.0, 1.0, 0.0)
  n3 = VectorValue(0.0, 0.0, 1.0)
  A11 = interpolate_everywhere(L2_Projection(n1 ⋅ A ⋅ n1, dΩ, VL2), VH1)
  A12 = interpolate_everywhere(L2_Projection(n1 ⋅ A ⋅ n2, dΩ, VL2), VH1)
  A13 = interpolate_everywhere(L2_Projection(n1 ⋅ A ⋅ n3, dΩ, VL2), VH1)
  A22 = interpolate_everywhere(L2_Projection(n2 ⋅ A ⋅ n2, dΩ, VL2), VH1)
  A23 = interpolate_everywhere(L2_Projection(n2 ⋅ A ⋅ n3, dΩ, VL2), VH1)
  A33 = interpolate_everywhere(L2_Projection(n3 ⋅ A ⋅ n3, dΩ, VL2), VH1)
  trA = interpolate_everywhere(L2_Projection(tr ∘ A,     dΩ, VL2), VH1)
  (A11, A12, A13, A22, A23, A33, trA)
end


function interpolate_L2_vector(b, Ω, dΩ, Γ=Ω)
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  n1 = VectorValue(1.0, 0.0, 0.0)
  n2 = VectorValue(0.0, 1.0, 0.0)
  n3 = VectorValue(0.0, 0.0, 1.0)
  b1 = interpolate_everywhere(L2_Projection(n1 ⋅ b, dΩ, VL2), VH1)
  b1 = interpolate_everywhere(L2_Projection(n2 ⋅ b, dΩ, VL2), VH1)
  b1 = interpolate_everywhere(L2_Projection(n3 ⋅ b, dΩ, VL2), VH1)
  (b1, b2, b3)
end


function interpolate_L2_scalar(x, Ω, dΩ, Γ=Ω)
  refL2 = ReferenceFE(lagrangian, Float64, 0)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  VL2 = FESpace(Ω, refL2, conformity=:L2)
  VH1 = FESpace(Γ, reffe, conformity=:H1)
  interpolate_everywhere(L2_Projection(x, dΩ, VL2), VH1)
end


function L2_Projection(u, dΩ, V)
  a(w, v) = ∫(w ⊙ v) * dΩ
  l(v)    = ∫(v ⊙ u) * dΩ
  op      = AffineFEOperator(a, l, V, V)
  solve(op)
end
