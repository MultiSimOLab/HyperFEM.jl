# Flow around a cylinder, DFG benchmark 2D-1 (Schäfer & Turek 1996): steady laminar flow at Re = 20, reached by
# time stepping with the same discretization as examples/aorta_cfd (equal-order P1-P1 with SUPG + PSPG + LSIC,
# BDF2 with backward Euler in the first step, Oseen linearization: one linear system per step) written with
# HyperFEM's StaticLinearModel (re-assembled at every step) and a `backend` keyword. Mesh test/models/turek2d.msh
# (7 402 triangles, generated with `benchmark/meshes/generate.jl turek 0.02` of HyperFEMKernels.jl).
# Reference values of the benchmark (fine meshes, P2-P1): C_D = 5.5795, C_L = 0.0106, Δp = 0.1175.
using Gridap, GridapGmsh, Gridap.FESpaces, Gridap.CellData, GridapSolvers
using GridapSolvers.LinearSolvers
using HyperFEM
using LinearAlgebra

function turek_navier_stokes_simulation(; backend=:gridap, nsteps=12, dt=0.5, verbose=false)
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

  # Previous states (free values overwritten in place at every step; the Dirichlet values are those of U)
  xh⁻ = FEFunction(U, zero_free_values(U)); xh⁻⁻ = FEFunction(U, zero_free_values(U))
  un = xh⁻[1]; unm1 = xh⁻⁻[1]

  # BDF2 (first step backward Euler): ∂ₜu ≈ c0 u - (c1 uⁿ - c2 uⁿ⁻¹); Oseen advecting velocity 2uⁿ - uⁿ⁻¹
  bdf2 = Ref(false)
  c0() = bdf2[] ? 1.5 / dt : 1.0 / dt
  c1() = bdf2[] ? 2.0 / dt : 1.0 / dt
  c2() = bdf2[] ? 0.5 / dt : 0.0
  hist() = c1() * un - c2() * unm1
  wadv() = bdf2[] ? 2.0 * un - 1.0 * unm1 : un

  # Stabilization parameters (semi-discrete), evaluated with uⁿ; with P1 velocity the viscous term has no
  # element Laplacian, so the strong residual is ρ (c u + ∇u·w) + ∇p - ρ f
  τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
  τpspg(u, h) = τsupg(u, h) / ρ
  τlsic(u, h) = ρ * norm(u) * h * 0.5
  Rvel(c) = (u, ∇u, w) -> ρ * c * u + ρ * (∇u' ⋅ w)
  ε(∇u) = 0.5 * (∇u + ∇u')

  function a((u, p), (v, q))
    c = c0(); w = wadv()
    τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
    Ru = Rvel(c) ∘ (u, ∇(u), w)
    ∫(ρ * c * (u ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(u))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
    ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
    ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
    ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
    ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
  end
  function l((v, q))
    w = wadv(); f = hist()
    τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he)
    ∫(ρ * (f ⋅ v))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ (ρ * f)))dΩ - ∫(τp * (∇(q) ⋅ (ρ * f)))dΩ
  end
  compmodel = StaticLinearModel(l, a, U, V, D_bc; backend)

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
  x = zero_free_values(U)
  for n in 1:nsteps
    x = solve!(compmodel; Assembly=true)             # new uⁿ, uⁿ⁻¹ and BDF2 numbers in the forms
    xh = FEFunction(U, x)
    get_free_dof_values(xh⁻⁻) .= get_free_dof_values(xh⁻)
    get_free_dof_values(xh⁻) .= x
    bdf2[] = true
    push!(cD_history, coefficients(xh).cD)
    verbose && println("step $n  t = $(n * dt)  C_D = $(cD_history[end])")
  end
  xh = FEFunction(U, x)
  (; x, cD_history, coefficients(xh)...)
end
