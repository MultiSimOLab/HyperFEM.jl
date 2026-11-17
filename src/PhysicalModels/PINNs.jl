

function SoftPlus(x::Vector)
  n = size(x, 1)
  onesV = SVector{n}(ones(n))
  return log.(onesV + exp.(x))
end

function SoftPlus(x::Float64)
  return log.(1.0 + exp.(x))
end


function Identity(x)
  return x
end

function LayerComputation(W, b, Input, TheFunction)
  if size(b, 1) == 1
    return TheFunction(W ⋅ Input + b)
  else
    return TheFunction(W * Input + b)
  end
end

function InvariantScaling(Inv, ϵ, β)
  return (Inv .* ϵ) .+ β
end



struct ThermoElectroMech_PINNs{A,B,C,D} <: ThermoElectroMechano{Thermo,Electro,Mechano}
  W::A
  b::B
  ϵ::C
  β::D
  nLayer::Int64
  κ::Float64

  function ThermoElectroMech_PINNs(; W, b, ϵ, β, nLayer::Int64, κ::Float64=1e-4)
    A, B, C, D = typeof(W), typeof(b), typeof(ϵ), typeof(β)
    new{A,B,C,D}(W, b, ϵ, β, nLayer, κ)
  end

  function (obj::ThermoElectroMech_PINNs)(Λ::Float64=1.0)

    function Ψ(F, E, δθ)
      Kinematic_mec = Kinematics(Mechano,Solid)
      Kinematic_elec = Kinematics(Electro,Solid)
      I1, I2, I3, I4, I5 = getIsoInvariants(Kinematic_mec, Kinematic_elec)
      I_ = InvariantScaling([I1(F), I2(F), I3(F), I4(F, E), I5(E), δθ], obj.ϵ, obj.β)
      h = LayerComputation(obj.W[1], obj.b[1], [I_[1], I_[2], I_[3], I_[4], I_[5], I_[6]], SoftPlus)
      @inbounds for iLayer in 2:obj.nLayer-1
        h = LayerComputation(obj.W[iLayer], obj.b[iLayer], h, SoftPlus)
      end
      return LayerComputation(obj.W[end], obj.b[end], h, Identity)
    end

    ∂Ψ_∂F(F, E, δθ) = ForwardDiff.gradient(F -> Ψ(F, get_array(E), δθ), get_array(F))
    ∂Ψ_∂E(F, E, δθ) = ForwardDiff.gradient(E -> Ψ(get_array(F), E, δθ), get_array(E))
    ∂Ψ_∂θ(F, E, δθ) = ForwardDiff.derivative(δθ -> Ψ(get_array(F), get_array(E), δθ), δθ)


    ∂2Ψ_∂2E(F, E, θ)   = ForwardDiff.hessian(E -> Ψ(get_array(F), E, θ), get_array(E))
    ∂2Ψ_∂2F(F, E, θ)   = ForwardDiff.jacobian(F -> ∂Ψ_∂F(F, get_array(E), θ), get_array(F))
    ∂2Ψ_∂2EF(F, E, θ)  = ForwardDiff.jacobian(F -> ∂Ψ_∂E(F, get_array(E), θ), get_array(F))
    ∂2Ψ_∂2Fθ(F, E, θ)  = ForwardDiff.derivative(θ -> ∂Ψ_∂F(get_array(F), get_array(E), θ), θ)
    ∂2Ψ_∂2Eθ(F, E, θ)  = ForwardDiff.derivative(θ -> ∂Ψ_∂E(get_array(F), get_array(E), θ), θ)
    ∂2Ψ_∂2∇θθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂θ(get_array(F), get_array(E), θ), θ)


    ∂ΨF(F, E, θ) = TensorValue(∂Ψ_∂F(F, E, θ))
    ∂ΨE(F, E, θ) = VectorValue(∂Ψ_∂E(F, E, θ))
    ∂Ψθ(F, E, θ) = ∂Ψ_∂θ(F, E, θ)

    ∂ΨFF(F, E, θ) = FourthOrderTensorValue(Tuple(∂2Ψ_∂2F(F, E, θ)))
    ∂ΨEE(F, E, θ) = TensorValue(∂2Ψ_∂2E(F, E, θ))
    ∂ΨEF(F, E, θ) = ThirdOrderTensorValue(Tuple(∂2Ψ_∂2EF(F, E, θ)))
    ∂ΨFθ(F, E, θ) = TensorValue(∂2Ψ_∂2Fθ(F, E, θ))
    ∂ΨEθ(F, E, θ) = VectorValue(∂2Ψ_∂2Eθ(F, E, θ))
    ∂Ψθθ(F, E, θ) = ∂2Ψ_∂2∇θθ(F, E, θ)

    η(F, E, θ) = -∂Ψθ(F, E, θ)

    return (Ψ, ∂ΨF, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η)
  end
end
