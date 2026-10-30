# Flow around a cylinder, DFG benchmark 2D-1 (Schäfer & Turek 1996): steady laminar flow at Re = 20, reached by
# time stepping with the same discretization as examples/aorta_cfd (equal-order P1-P1 with SUPG + PSPG + LSIC,
# BDF2 with backward Euler in the first step, Oseen linearization: one linear system per step) written on
# HyperFEM's TransientModel (general level of the stage contract, `BDF(2)`, `tangent_update = :step`) with a
# `backend` keyword. Mesh test/models/turek2d.msh (7 402 triangles, generated with
# `benchmark/meshes/generate.jl turek 0.02` of HyperFEMKernels.jl).
# Reference values of the benchmark (fine meshes, P2-P1): C_D = 5.5795, C_L = 0.0106, Δp = 0.1175.
using Gridap, GridapGmsh, Gridap.FESpaces, Gridap.CellData, GridapSolvers
using GridapSolvers.LinearSolvers
using HyperFEM
using LinearAlgebra

function turek_navier_stokes_simulation(; backend=:lazy, nsteps=12, dt=0.5, verbose=false)
  geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "models", "turek2d.msh"))
  ρ, μ = 1.0, 1e-3                      # kg/m3, Pa s (ν = 1e-3 m2/s)
  Um, H, D = 0.3, 0.41, 0.1             # inlet peak velocity, channel height, cylinder diameter: Re = 2/3 Um D / ν = 20
  Ū = 2Um / 3

  # Dirichlet velocity: parabolic inlet, no slip on the walls and the cylinder; pressure free (do-nothing outlet)
  Du = BoundaryConditions(
    BoundaryCondition("inlet", x -> VectorValue(4Um * x[2] * (H - x[2]) / H^2, 0.0)),
    BoundaryCondition("wall", [0.0, 0.0]),
    BoundaryCondition("cylinder", [0.0, 0.0]))
  D_bc = MultiFieldBC([Du, NothingBC()])

  order = 1
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  reffep = ReferenceFE(lagrangian, Float64, order)
  Vu = TestFESpace(geomodel, reffeu, D_bc[1], conformity=:H1)
  Vp = TestFESpace(geomodel, reffep, D_bc[2], conformity=:H1)
  Uu = TrialFESpace(Vu, D_bc[1], 1.0)
  Up = TrialFESpace(Vp, D_bc[2], 1.0)
  V = MultiFieldFESpace([Vu, Vp])
  U = MultiFieldFESpace([Uu, Up])

  Ω = Triangulation(geomodel)
  dΩ = Measure(Ω, 2order + 1)
  he = CellField(map(a -> sqrt(2a), collect(get_cell_measure(Ω))), Ω)   # element size (triangles)

  # Stabilization parameters (semi-discrete), evaluated with uⁿ; with P1 velocity the viscous term has no
  # element Laplacian, so the strong residual is ρ ∂ₜu + ρ ∇u·w + ∇p - ρ f
  τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
  τpspg(u, h) = τsupg(u, h) / ρ
  τlsic(u, h) = ρ * norm(u) * h * 0.5
  ε(∇u) = 0.5 * (∇u + ∇u')

  # The time derivative of the velocity comes from the context of the velocity field (BDF2, backward Euler
  # in the first step: `∂t(cu, u) = c0 u - (c1 uⁿ - c2 uⁿ⁻¹)`), also inside the SUPG and PSPG residuals; the
  # Oseen advecting velocity is extrapolated from the history, 2uⁿ - uⁿ⁻¹ (uⁿ in the first step, which has
  # no uⁿ⁻¹: uⁿ carries the inlet profile in its Dirichlet values from the start). The expression keeps
  # its type across the steps (the operator of the model is rebuilt with the same type every step).
  function advecting(cu)
    a, b = cu.step == 1 ? (1.0, 0.0) : (2.0, 1.0)
    a * cu.u⁻ - b * cu.u⁻⁻
  end
  function res(ctx)
    cu = ctx[1]
    un = cu.u⁻; w = advecting(cu)
    τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
    ((u, p), (v, q)) -> begin
      ut = ∂t(cu, u)
      Ru = ρ * ut + ρ * (∇(u)' ⋅ w)
      ∫(ρ * (ut ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(u))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
      ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
      ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
      ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
      ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
    end
  end
  function jac(ctx)
    cu = ctx[1]
    un = cu.u⁻; w = advecting(cu)
    τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
    ((u, p), (du, dp), (v, q)) -> begin
      dut = δ∂t(cu, du)
      dRu = ρ * dut + ρ * (∇(du)' ⋅ w)
      ∫(ρ * (dut ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(du))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(du)' ⋅ w) ⋅ v))dΩ -
      ∫(dp * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ du))dΩ +
      ∫(τs * ((∇(v)' ⋅ w) ⋅ dRu))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(dp)))dΩ -
      ∫(τp * (∇(q) ⋅ dRu))dΩ - ∫(τp * (∇(q) ⋅ ∇(dp)))dΩ +
      ∫(τl * (∇ ⋅ v) * (∇ ⋅ du))dΩ
    end
  end
  # linear in the unknown with a tangent that depends on the history: one solve per step from the zero guess
  nls = NewtonRaphson(LUSolver(); verbose=0)
  compmodel = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), tangent_update=:step, nls, backend)

  # Forces on the cylinder: F = -∫ σ·n dΓ with n the outward normal of the fluid (σ = -p I + 2μ ε(u))
  Γc = BoundaryTriangulation(geomodel, tags="cylinder"); dΓc = Measure(Γc, 2order + 1); nc = get_normal_vector(Γc)
  I2 = TensorValue(1.0, 0.0, 0.0, 1.0)
  function coefficients(xh)
    uh, ph = xh
    σ = 2μ * (ε ∘ (∇(uh))) - ph * I2
    F = -sum(∫(σ ⋅ nc)dΓc)
    cD, cL = 2 * F[1] / (ρ * Ū^2 * D), 2 * F[2] / (ρ * Ū^2 * D)
    Δp = ph(Point(0.15, 0.2)) - ph(Point(0.25, 0.2))
    (; cD, cL, Δp)
  end

  cD_history = Float64[]
  function record(post)
    push!(cD_history, coefficients(get_state(compmodel)).cD)
    verbose && println("step $(post.iter)  t = $(post.Λ[end])  C_D = $(cD_history[end])")
  end
  post = PostProcessor(compmodel, record; is_vtk=false, filepath=mktempdir())
  x, flag = solve!(compmodel; stepping=TimeStepping(; Δt=dt, t_end=nsteps * dt), post)
  flag == 1 || error("Turek: the time loop did not complete")
  xh = get_state(compmodel)
  (; x, cD_history, coefficients(xh)...)
end
