
"The evolution functions have been designed to apply variable boundary conditions."
module EvolutionFunctions

export ramp
export triangular
export heaviside
export smoothstep
export constant

import Base: +, -, *

struct EvolutionLaw{F} <: Function
  f::F
end

(law::EvolutionLaw)(t) = law.f(t)

*(c::Number, law::EvolutionLaw) = EvolutionLaw(t -> c * law(t))
*(law::EvolutionLaw, c::Number) = EvolutionLaw(t -> law(t) * c)

+(law::EvolutionLaw, c::Number) = EvolutionLaw(t -> law(t) + c)
+(c::Number, law::EvolutionLaw) = EvolutionLaw(t -> c + law(t))

-(law::EvolutionLaw, c::Number) = EvolutionLaw(t -> law(t) - c)
-(c::Number, law::EvolutionLaw) = EvolutionLaw(t -> c - law(t))

+(law1::EvolutionLaw, law2::EvolutionLaw) = EvolutionLaw(t -> law1(t) + law2(t))
-(law1::EvolutionLaw, law2::EvolutionLaw) = EvolutionLaw(t -> law1(t) - law2(t))

"""Return a bounded ramp function from 0 to 1. By default, the slope is the identity. Otherwise, the scaling factor is 1/T.
```math
f(t) = \\begin{cases}
  0   &, t < 0 \\\\
  t/T &, 0 \\leq t < T \\\\
  1   &, t \\geq T
\\end{cases}
```
"""
function ramp(T::Real=1.0)
  EvolutionLaw(t::Real -> max(min(t/T, 1.0), 0.0))
end

"""Return a triangular evolution function ranging from 0 to 1, centered at `Tmax`, having edges at `T0` and `2Tmax-T0`.
By default, `T0=0` and `Tmax=T`.
```math
f(t) = \\begin{cases}
  0 &, t < T_0 \\\\
  \\frac{t-T_0}{T_{max}-T_0} &, T_0 \\leq t < T_{max} \\\\
  1-\\frac{t-T_{max}}{T_{max}-T_0} &, T_{max} \\leq t < 2T_{max}-T_0 \\\\
  0 &, t \\geq 2T_{max}-T_0
\\end{cases}
```
"""
function triangular(T::Real)
  triangular(0.0, T)
end

function triangular(T0::Real, Tmax::Real)
  EvolutionLaw(t::Real -> begin
    Δ = Tmax - T0
    u = (t - T0) / Δ
    v = (t - Tmax) / Δ
    max(min(u, 1.0-v), 0.0)
  end)
end

"""Return the Heaviside function.
```math
f(t) = H(t,T)
```
"""
function heaviside(T::Real)
  EvolutionLaw(t::Real -> t > T ? 1.0 : 0.0)
end

"""Return a sigmoid-like function centered at T and edges at ±ϵ.
by default, `T=ϵ`.
```math
u(t) = \\frac{t - T + \\epsilon}{2\\epsilon} \\\\[10pt]
f(t) = \\begin{cases}
0 &, u < 0 \\\\
3u^2 - 2u^3 &, 0 \\leq u < 1 \\\\
1 &, u \\geq 1
\\end{cases}
```
"""
function smoothstep(T::Real, ϵ::Real)
  EvolutionLaw(t::Real -> begin
    u::Real = (t - T + ϵ) / (2 * ϵ)
    if u < 0.0 return 0.0
    elseif u < 1.0 return 3*u^2 - 2*u^3
    else return 1.0
    end
  end)
end

function smoothstep(ϵ::Real)
  smoothstep(ϵ, ϵ)    
end

"""Return a constant function which is always evaluated to 1.
```math
f(x) = 1
```
"""
function constant()
  EvolutionLaw(t::Real -> 1.0)
end

end
