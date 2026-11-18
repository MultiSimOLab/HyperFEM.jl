
# =================
# Hyper-elasticity
# =================

# -----------------
# Stagered residual
# -----------------

function residual(physicalmodel::ElectroMechano, ::Type{Mechano}, kine::NTuple{2,KinematicModel}, (u, φ), v, dΩ, Λ=1.0, vars...; kwargs...)
    DΨ    = physicalmodel(; kwargs...)
    F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
    E     = get_Kinematics(kine[2]; Λ=Λ)
    ∂Ψu   = DΨ[2]
    ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', E∘∇(φ), vars...)))dΩ
end

function residual(physicalmodel::ElectroMechano, ::Type{Electro}, kine::NTuple{2,KinematicModel}, (u, φ), vφ, dΩ, Λ=1.0, vars...; kwargs...)
    DΨ    = physicalmodel(; kwargs...)
    F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
    E     = get_Kinematics(kine[2]; Λ=Λ)
    ∂Ψφ   = DΨ[3]
    -1.0*∫(∇(vφ) ⋅ (∂Ψφ ∘ (F∘∇(u)', E∘∇(φ), vars...)))dΩ
end

# -----------------
# Stagered jacobian
# -----------------

function jacobian(physicalmodel::ElectroMechano, ::Type{Mechano}, kine::NTuple{2,KinematicModel}, (u, φ), du, v, dΩ, Λ=1.0, vars...; kwargs...)
    DΨ    = physicalmodel(; kwargs...)
    F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
    E     = get_Kinematics(kine[2]; Λ=Λ)
    ∂Ψuu  = DΨ[4]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)', E∘∇(φ), vars...)) ⊙ ∇(du)'))dΩ
end

function jacobian(physicalmodel::ElectroMechano, ::Type{Electro}, kine::NTuple{2,KinematicModel}, (u, φ), dφ, vφ, dΩ, Λ=1.0, vars...; kwargs...)
    DΨ    = physicalmodel(; kwargs...)
    F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
    E     = get_Kinematics(kine[2]; Λ=Λ)
    ∂Ψφφ  = DΨ[6]
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘∇(u)', E∘∇(φ), vars...)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ElectroMechano, ::Type{ElectroMechano}, kine::NTuple{2,KinematicModel}, (u, φ), (du, dφ), (v, vφ), dΩ, Λ=1.0, vars...; kwargs...)
    DΨ    = physicalmodel(; kwargs...)
    F,_,_ = get_Kinematics(kine[1]; Λ=Λ)
    E     = get_Kinematics(kine[2]; Λ=Λ)
    ∂Ψφu  = DΨ[5]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ), vars...)) ⊙ ∇(v)'))dΩ -
         ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ), vars...)) ⊙ ∇(du)'))dΩ 
end

# -------------------
# Monolithic strategy
# -------------------

function residual(physicalmodel::ElectroMechano,  kine::NTuple{2,KinematicModel}, (u, φ), (v, vφ), dΩ, Λ=1.0)
    residual(physicalmodel, Mechano, kine, (u, φ), v, dΩ, Λ) +
    residual(physicalmodel, Electro, kine, (u, φ), vφ, dΩ, Λ)
end


function jacobian(physicalmodel::ElectroMechano,  kine::NTuple{2,KinematicModel}, (u, φ), (du, dφ), (v, vφ), dΩ, Λ=1.0)
    jacobian(physicalmodel, Mechano, kine, (u, φ), du, v, dΩ, Λ)+
    jacobian(physicalmodel, Electro, kine, (u, φ), dφ, vφ, dΩ, Λ)+
    jacobian(physicalmodel, ElectroMechano, kine, (u, φ), (du, dφ), (v, vφ), dΩ, Λ)
end

# -------------------
# Monolithic strategy
# -------------------

function residual(physicalmodel::ElectroMechModel{<:Electro,<:ViscoElastic}, kine::NTuple{2,KinematicModel},  (u, φ), (v, vφ), dΩ, Λ, un, A)
    residual(physicalmodel, Mechano, kine, (u, φ), v, dΩ, Λ, un, A) +
    residual(physicalmodel, Electro, kine, (u, φ), vφ, dΩ, Λ, un, A)
end

function jacobian(physicalmodel::ElectroMechModel{<:Electro,<:ViscoElastic}, kine::NTuple{2,KinematicModel}, (u, φ), (du, dφ), (v, vφ), dΩ, Λ, un, A)
    jacobian(physicalmodel, Mechano, kine,(u, φ), du, v, dΩ, Λ, un, A) +
    jacobian(physicalmodel, Electro, kine,(u, φ), dφ, vφ, dΩ, Λ, un, A) +
    jacobian(physicalmodel, ElectroMechano, kine,(u, φ), (du, dφ), (v, vφ), dΩ, Λ, un, A)
end
