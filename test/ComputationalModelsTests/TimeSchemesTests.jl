# The stage contract of the time schemes on scalar problems: the stage algebra against hand-written updates
# and the asymptotic exactness of the local error estimates (ratio estimate / true local error → 1 as Δt → 0,
# constant and variable steps), on ẋ = λx (first order) and ẍ = −ω²x (second order).
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.ComputationalModels: StageHistory, stage_coefficients!, predict!, commit_stage!, commit!, error_estimate!,
  order, nstages, _dirichlet_values
using LinearAlgebra
using Test

# a trial space with one free dof (P1 on a 2 × 2 grid, Dirichlet on the boundary) to build a StageHistory
function one_dof_space()
  model = CartesianDiscreteModel((0, 1, 0, 1), (2, 2))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  bc = BoundaryConditions(BoundaryCondition("boundary", 0.0, EvolutionFunctions.constant()))
  V = TestFESpace(model, reffe, bc; conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  @assert num_free_dofs(U) == 1
  U
end

# history of a first-order scalar problem filled with the exact solution x(t) at tₙ = 0 and the two previous
# steps (Δt⁻, Δt⁻⁻)
function first_order_history(U, x, Δt⁻, Δt⁻⁻; nstored=3)
  h = StageHistory(U, [x(0.0)], 2)
  h.xn .= x(0.0); h.xnm1 .= x(-Δt⁻); h.xnm2 .= x(-Δt⁻ - Δt⁻⁻)
  h.Δt⁻ = Δt⁻; h.Δt⁻⁻ = Δt⁻⁻; h.nstored = nstored
  h
end

@testset "First-order schemes: stage algebra and error estimates (ẋ = λx)" begin
  U = one_dof_space()
  λ = -1.7
  x(t) = exp(λ * t)
  for (ω, ω′) in ((1.0, 1.0), (1.5, 0.7), (0.6, 1.3))
    for scheme in (BDF(1), BDF(2), Theta(1.0), Theta(0.5), Theta(0.7), SDIRK(2))
      ratios = Float64[]
      for Δt in (0.1, 0.05, 0.025, 0.0125)
        Δt⁻ = Δt / ω; Δt⁻⁻ = Δt⁻ / ω′
        h = first_order_history(U, x, Δt⁻, Δt⁻⁻)
        y = [0.0]
        for i in 1:nstages(scheme)
          c = stage_coefficients!(scheme, h, i, Δt, Δt⁻)
          predict!(y, scheme, h, i, Δt)
          # the scalar stage: ∂t = λ field, i.e. c.α1 y + β1 = λ (c.α0 y + β0)
          β1 = h.b1[1]
          β0 = h.use_β0 ? h.b0[1] : 0.0
          y[1] = (λ * β0 - β1) / (c.α1 - λ * c.α0)
          commit_stage!(scheme, h, i, y, Δt)
        end
        error_estimate!(scheme, h, y, Δt, Δt⁻)
        if scheme isa SDIRK
          # embedded pair: the estimate is the error of the lower-order solution x̂ (up to the sign)
          x̂ = h.xn[1] + Δt * sum(scheme.b̂ .* [k[1] for k in h.ks])
          push!(ratios, -h.e[1] / (x̂ - x(Δt)))
        else
          push!(ratios, h.e[1] / (y[1] - x(Δt)))
        end
      end
      # the ratio tends to 1 with the order of the next term; at Δt = 0.0125 it is within a few percent
      @test abs(ratios[end] - 1) < 0.05
      @test abs(ratios[end] - 1) < abs(ratios[1] - 1)
    end
  end
  # the hand-written BDF2 and the midpoint rule: the stage algebra reproduces the formulas
  Δt = 0.05
  h = first_order_history(U, x, Δt, Δt)
  c = stage_coefficients!(BDF(2), h, 1, Δt, Δt)
  @test c.α1 ≈ 1.5 / Δt && h.b1[1] ≈ -(2 * h.xn[1] - 0.5 * h.xnm1[1]) / Δt
  c = stage_coefficients!(Theta(0.5), h, 1, Δt, Δt)
  @test c.α0 == 0.5 && h.b0[1] ≈ 0.5 * h.xn[1] && c.α1 ≈ 1 / Δt && h.b1[1] ≈ -h.xn[1] / Δt && c.s_K == 0.5
end

@testset "Second-order schemes: stage algebra and error estimates (ẍ = −ω²x)" begin
  U = one_dof_space()
  ω0 = 2.3
  x(t) = cos(ω0 * t) + 0.3 * sin(ω0 * t)
  v(t) = -ω0 * sin(ω0 * t) + 0.3 * ω0 * cos(ω0 * t)
  a(t) = -ω0^2 * x(t)
  for scheme in (Newmark(), GeneralizedAlpha(0.8), HHT(0.1), Midpoint(), Newmark(β=0.3, γ=0.6))
    ratios = Float64[]
    for Δt in (0.1, 0.05, 0.025, 0.0125)
      h = StageHistory(U, [x(0.0)], 1)
      h.xn .= x(0.0); h.vn .= v(0.0); h.an .= a(0.0); h.nstored = 1; h.Δt⁻ = Δt
      c = stage_coefficients!(scheme, h, 1, Δt, Δt)
      y = [0.0]
      predict!(y, scheme, h, 1, Δt)
      # the scalar stage: α2 y + β2 = −ω0² (α0 y + β0)
      β2 = h.b2[1]; β0 = h.use_β0 ? h.b0[1] : 0.0
      y[1] = (-ω0^2 * β0 - β2) / (c.α2 + ω0^2 * c.α0)
      # Newmark hand update (αm = αf = 0): check the algebra of the plain Newmark member
      if scheme.αm == 0 && scheme.αf == 0
        β, γ = scheme.β, scheme.γ
        ũ = x(0) + Δt * v(0) + Δt^2 * (0.5 - β) * a(0)
        yh = (ũ / (β * Δt^2)) / (1 / (β * Δt^2) + ω0^2)
        @test y[1] ≈ yh
      end
      error_estimate!(scheme, h, y, Δt, Δt)
      commit!(scheme, h, y, Δt)
      @test h.xn[1] == y[1]
      # Zienkiewicz–Xie estimates the local displacement error of the Newmark family
      push!(ratios, h.e[1] / (y[1] - x(Δt)))
    end
    @test abs(ratios[end] - 1) < 0.1
    @test abs(ratios[end] - 1) < abs(ratios[1] - 1)
  end
  # the velocity and acceleration of the commit (trapezoidal rule: exact momentum relations)
  Δt = 0.01
  h = StageHistory(U, [x(0.0)], 1)
  h.xn .= x(0.0); h.vn .= v(0.0); h.an .= a(0.0); h.nstored = 1
  c = stage_coefficients!(Newmark(), h, 1, Δt, Δt)
  y = [(-c.α2 * 0.0 - h.b2[1]) / (c.α2 + ω0^2)]
  commit!(Newmark(), h, y, Δt)
  @test h.an[1] ≈ -ω0^2 * y[1]
  @test h.vn[1] ≈ v(0.0) + 0.5Δt * (a(0.0) + h.an[1])
  # central differences: the explicit field and the commit
  h = StageHistory(U, [x(0.0)], 1)
  h.xn .= x(0.0); h.vn .= v(0.0); h.an .= a(0.0); h.nstored = 1
  c = stage_coefficients!(CentralDifference(), h, 1, Δt, Δt)
  @test c.α0 == 0.0 && c.α2 == 1.0 && c.s_K == 0.0
  @test h.b0[1] ≈ x(0.0) + Δt * v(0.0) + 0.5Δt^2 * a(0.0)
  y = [-ω0^2 * h.b0[1]]
  commit_stage!(CentralDifference(), h, 1, y, Δt)
  commit!(CentralDifference(), h, h.b0, Δt)
  @test h.an[1] == y[1] && h.vn[1] ≈ v(0.0) + 0.5Δt * (a(0.0) + y[1]) && h.xn[1] ≈ x(0.0) + Δt * v(0.0) + 0.5Δt^2 * a(0.0)
end
