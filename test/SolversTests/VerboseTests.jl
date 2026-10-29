# Verbose output of the drivers and the Newton: one line per step (1), rows per iteration (2), line-search
# detail (3), silence (0); the run summary.
using Gridap, Gridap.Algebra, Gridap.FESpaces
using HyperFEM
using Test

function verbose_test_problem()
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (4, 2))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  km = Kinematics(Mechano, Solid)
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [0.4, 0.1], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> residual(physmodel, km, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, km, u, du, v, dΩ)
  (; res, jac, U, V, bc, dΩ)
end

function capture_verbose(verbose; linesearch=FullStepLS(), injectivity=nothing)
  p = verbose_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=verbose, linesearch=linesearch)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls, injectivity=injectivity === nothing ? nothing : p.dΩ)
  out = capture_stdout(() -> solve!(m; stepping=(nsteps=2, maxbisec=0)))
  out, nls
end

@testset "Verbose output" begin
  out, _ = capture_verbose(0)
  @test isempty(out)

  out, nls = capture_verbose(1)
  lines = split(strip(out), '\n')
  @test count(l -> startswith(l, "step"), lines) == 2
  @test occursin("Λ = 0.5000", lines[1]) && occursin("ΔΛ = 0.5000", lines[1]) && occursin("✓", lines[1])
  @test occursin("Λ = 1.0000 reached: 2 steps, 0 bisections", out)
  @test occursin("$(nls.log.niters) iterations", out) && occursin("0 % projected, 0 switches", out)
  @test occursin("assembly", out) && occursin("line search", out)
  @test !occursin("‖R‖/‖R₀‖", out)          # no iteration rows at level 1

  out, nls = capture_verbose(2)
  @test occursin("━━ Step 1 ━━ Λ = 0.5000  ΔΛ = 0.5000 ━━", out)
  @test occursin("━━ Step 2 ━━ Λ = 1.0000", out)
  @test occursin("‖R‖/‖R₀‖", out) && occursin("consistent", out) && occursin("LU", out)
  @test count(l -> occursin("✓ Newton-Raphson: converged", l), split(out, '\n')) == 2
  @test occursin("reached: 2 steps", out)
  @test !occursin("line search:", out)      # detail only at level 3
  nrows = count(l -> occursin(r"^\s+\d+\s+\d\.\d{3}e[+-]\d{2}\s+\d\.\d{3}e", l), split(out, '\n'))
  @test nrows == nls.log.niters + 2         # iteration 0 of every step plus the iterations

  out, nls = capture_verbose(3; linesearch=InjectivityLS(), injectivity=true)
  @test occursin("line search: injectivity step", out)
  @test occursin("†", out) || !any(<(1), step_history(nls))   # steps at the injectivity limit are marked
  @test occursin("reductions", out)

  # colours only on a terminal: nothing escaped in the captured text
  @test !occursin("\e[", out)
  # the solver's own summary
  s = capture_stdout(() -> print_summary(nls.log))
  @test occursin("solves", s) && occursin("time", s)
  @test reason_text(:maxiter) == "maximum number of iterations"
end
