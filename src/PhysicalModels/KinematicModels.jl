
abstract type KinematicModel end
abstract type KinematicDescription end

abstract type Solid <: KinematicDescription end
abstract type SolidShell <: KinematicDescription end



get_Kinematics(::KinematicModel; Λ::Float64) = @abstractmethod


struct Kinematics{T1,T2} <: KinematicModel
    metrics

    function Kinematics(::Type{Mechano}, ::Type{Solid}; F::Function=(∇u) -> one(∇u) + ∇u)
        J, H = det, cof
        metrics = (F, H, J)
        new{Mechano,Solid}(metrics)
    end

    # function Kinematics(::Type{Mechano}, ::Type{SolidShell})
    #     function F(∇u, N)
    #         F̂ = ∇u + one(∇u)
    #         Ĵ = det(F̂)
    #         Ĥ = (det(F̂) * inv(F̂)')
    #         HN = (Ĥ * N) / norm(Ĥ * N)
    #         F̂ + (1 / Ĵ - 1.0) * (HN ⊗ N)
    #     end
    #     J(F) = det(F)
    #     H(F) = det(F) * inv(F)'
    #     metrics = (F, H, J)
    #     new{Mechano,SolidShell}(metrics)
    # end

    function Kinematics(::Type{Electro}, ::Type{Solid}; E::Function=(∇φ) -> -∇φ)
        metrics = E
        new{Electro,Solid}(metrics)
    end

    function Kinematics(::Type{Magneto}, ::Type{Solid}; H::Function=(∇φ) -> -∇φ)
        metrics = H
        new{Magneto,Solid}(metrics)
    end
    function Kinematics(::Type{Thermo}, ::Type{Solid})
        new{Thermo,Solid}(nothing)
    end
end

get_Kinematics(obj::Kinematics; Λ=1.0) = obj.metrics


function getIsoInvariants(obj::Kinematics{Mechano,<:KinematicDescription})
    F, H, J = obj.metrics
    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)
    return (I1, I2, I3)
end


function getIsoInvariants(obj_m::Kinematics{Mechano,<:KinematicDescription}, obj_e::Kinematics{Electro,<:KinematicDescription})
    F, H, J = obj_m.metrics
    E = obj_e.metrics
    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)
    HE(F, E) = H(F) * E
    I4(F, E) = HE(F, E) ⋅ HE(F, E)
    I5(E) = E ⋅ E
    return (I1, I2, I3, I4, I5)
end

struct EvolutiveKinematics{T} <: KinematicModel
    metrics

    function EvolutiveKinematics(::Type{Mechano}; F::Function=(t) -> ((∇u) -> one(∇u) + ∇u))
        # F_(∇u) = one(∇u) + ∇u
        # F(t) = (∇u)->(Fmapping(t) ∘ F_)(∇u) 
        J, H = det, cof
        metrics = (F, H, J)
        new{Mechano}(metrics)
    end

    function EvolutiveKinematics(::Type{Mechano}, δ::Float64; F::Function=(t) -> ((∇u) -> one(∇u) + ∇u))
        # F_(∇u) = one(∇u) + ∇u
        # F(t) = (∇u)->(Fmapping(t) ∘ F_)(∇u) 
        J(F) = 0.5 * (det(F) + sqrt(det(F)^2 + δ^2))
        H(F) = J(F) * inv(F)'
        metrics = (F, H, J)
        new{Mechano}(metrics)
    end

    function EvolutiveKinematics(::Type{Electro}; E::Function=(t) -> ((∇φ) -> -∇φ))
        metrics = E
        new{Electro}(metrics)
    end

    function EvolutiveKinematics(::Type{Magneto}; H::Function=(t) -> ((∇φ) -> -∇φ))
        metrics = H
        new{Magneto}(metrics)
    end
end

get_Kinematics(obj::EvolutiveKinematics{Mechano}; Λ=1.0) = (obj.metrics[1](Λ), obj.metrics[2], obj.metrics[3])
get_Kinematics(obj::EvolutiveKinematics{Electro}; Λ=1.0) = obj.metrics(Λ)
get_Kinematics(obj::EvolutiveKinematics{Magneto}; Λ=1.0) = obj.metrics(Λ)
