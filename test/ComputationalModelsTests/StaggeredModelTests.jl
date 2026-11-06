using Test
using Gridap, Gridap.FESpaces, Gridap.MultiField
using HyperFEM
using HyperFEM.ComputationalModels: StaggeredStepping, StaggeredLog, Aitken, get_log, num_steps, num_substeps, mean_substeps,
  NoRelaxation, FixedRelaxation, as_relaxation
using LinearAlgebra

# Electro-mechanics of a block (the coarse EM box of the solver tests, `s1_em_problem`): the staggered
# solve (electrics with the lagged deformation, mechanics with the new potential) converged to the coupling
# tolerance must reach the monolithic solution, whatever the number of steps; the fixed loop with one pass
# per step must not.
function em_block(; n=(8, 2, 4), φtop=0.05)
  model = CartesianDiscreteModel((0.0, 1.0, 0.0, 0.2, 0.0, 0.2), n)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
  # electrodes on the mid plane and the top face of the half of the block next to the clamped face: the fringe
  # field bends the cantilever and the bending changes the field (two-way coupling)
  add_tag_from_vertex_filter!(labels, "topsuf", model, x -> abs(x[3] - 0.2) < 1e-9 && x[1] < 0.5 + 1e-9)
  add_tag_from_vertex_filter!(labels, "midsuf", model, x -> abs(x[3] - 0.1) < 1e-9 && x[1] < 0.5 + 1e-9)
  physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
  k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
  dΩ = Measure(Triangulation(model), 2)
  Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
  Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", φtop, Λ -> Λ))
  Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
  (; physmodel, k, dΩ, Du, Dφ, Vu, Vφ)
end

newton(; maxiter=30) = NewtonRaphson(LUSolver(); maxiter, atol=1e-12, rtol=1e-10, verbose=0)

"Monolithic reference: free values of u and φ (numbered as the single-field spaces)."
function em_monolithic(p; nsteps=4)
  D = MultiFieldBC([p.Du, p.Dφ])
  V = MultiFieldFESpace([p.Vu, p.Vφ])
  U = MultiFieldFESpace([TrialFESpace(p.Vu, p.Du, 1.0), TrialFESpace(p.Vφ, p.Dφ, 1.0)])
  res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(p.physmodel, p.k, (u, φ), (v, vφ), p.dΩ)
  jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(p.physmodel, p.k, (u, φ), (du, dφ), (v, vφ), p.dΩ)
  m = StaticNonlinearModel(res, jac, U, V, D; nls=newton())
  x, flag = solve!(m; stepping=(nsteps=nsteps, maxbisec=4))
  @test flag == 1
  xh = FEFunction(U, x)
  (; u=copy(get_free_dof_values(xh[1])), φ=copy(get_free_dof_values(xh[2])))
end

"Staggered model (electrics, mechanics) of the block; the states and their lagged copies start at zero."
function em_staggered(p; backend=:lazy, injectivity=false, maxiter=30)
  Uu = TrialFESpace(p.Vu, p.Du, 1.0)
  uh⁺ = FEFunction(Uu, zero_free_values(Uu))
  uh⁻ = FEFunction(TrialFESpace(p.Vu, p.Du, 1.0), zero_free_values(Uu))
  Uφ = TrialFESpace(p.Vφ, p.Dφ, 1.0)
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
  φh⁻ = FEFunction(TrialFESpace(p.Vφ, p.Dφ, 1.0), zero_free_values(Uφ))
  mech(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ
  elec(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
  res_elec(Λ) = (φ, vφ) -> HyperFEM.residual(p.physmodel, Electro, p.k, (mech(Λ), φ), vφ, p.dΩ)
  jac_elec(Λ) = (φ, dφ, vφ) -> HyperFEM.jacobian(p.physmodel, Electro, p.k, (mech(Λ), φ), dφ, vφ, p.dΩ)
  res_mec(Λ) = (u, v) -> HyperFEM.residual(p.physmodel, Mechano, p.k, (u, elec(Λ)), v, p.dΩ)
  jac_mec(Λ) = (u, du, v) -> HyperFEM.jacobian(p.physmodel, Mechano, p.k, (u, elec(Λ)), du, v, p.dΩ)
  m_elec = StaticNonlinearModel(res_elec, jac_elec, Uφ, p.Vφ, p.Dφ; nls=newton(; maxiter), xh=φh⁺, backend)
  nls_mec = injectivity ? NewtonRaphson(LUSolver(); maxiter, atol=1e-12, rtol=1e-10, verbose=0, linesearch=InjectivityLS()) : newton(; maxiter)
  m_mec = StaticNonlinearModel(res_mec, jac_mec, Uu, p.Vu, p.Du; nls=nls_mec, xh=uh⁺, backend,
                               injectivity=injectivity ? p.dΩ : nothing)
  sm = StaggeredModel((m_elec, m_mec), (φh⁺, uh⁺), (φh⁻, uh⁻))
  args = ((stepping=(nsteps=1, maxbisec=2),), (stepping=(nsteps=1, maxbisec=2),))
  (; sm, uh⁺, φh⁺, Uu, Uφ, args, m_elec, m_mec)
end

relerr(a, b) = norm(a - b) / norm(b)
errors(s, ref) = (relerr(get_free_dof_values(s.uh⁺), ref.u), relerr(get_free_dof_values(s.φh⁺), ref.φ))

@testset "StaggeredStepping and relaxations" begin
  s = StaggeredStepping(; nsteps=4, nsubsteps=3, maxbisec=2)                 # the names of the NamedTuple of earlier versions
  @test s.max_substeps == 3 && s.coupling_tol === nothing
  @test HyperFEM.ComputationalModels.staggered_stepping((nsteps=4, nsubsteps=2, maxbisec=1)).max_substeps == 2
  s = StaggeredStepping(; nsteps=4, max_substeps=10, coupling_tol=1e-6, relaxation=(:none, 0.5), grow=2.0)
  @test s.coupling_tol == 1e-6 && s.grow == 2.0
  @test as_relaxation(:none) isa NoRelaxation && as_relaxation(0.5) isa FixedRelaxation && as_relaxation(:aitken) isa Aitken
  @test_throws ErrorException StaggeredStepping(; relaxation=:foo)
  @test_throws ErrorException StaggeredStepping(; coupling_tol=0.0)
  @test_throws ErrorException StaggeredStepping(; norm=:energy)
  @test_throws ErrorException Aitken(; ωmin=0.5, ωmax=0.1)
  # Aitken formula on vectors: r⁻ = (1, 0), r = (0.5, 0) → ω = -ω⁻ (r⁻ ⋅ (r - r⁻)) / ‖r - r⁻‖² = 2 ω⁻, clamped
  f = HyperFEM.ComputationalModels.relaxation_factor
  a = Aitken(; ω0=1.0, ωmin=0.01, ωmax=1.0)
  @test f(a, 1, 0.3, [0.5, 0.0], [1.0, 0.0]) == 1.0
  @test f(a, 2, 0.3, [0.5, 0.0], [1.0, 0.0]) ≈ 0.6
  @test f(a, 2, 0.6, [0.5, 0.0], [1.0, 0.0]) == 1.0                         # clamped at ωmax
  @test f(a, 2, 0.3, [1.0, 0.0], [1.0, 0.0]) == 0.3                         # r = r⁻: the previous ω
  @test f(FixedRelaxation(0.7), 5, 0.3, [1.0], [2.0]) == 0.7
  @test f(NoRelaxation(), 5, 0.3, [1.0], [2.0]) == 1.0
end

@testset "StaggeredModel: coupling tolerance reaches the monolithic solution" begin
  p = em_block()
  ref = em_monolithic(p)

  # the fixed loop with one pass per step (the NamedTuple of earlier versions) does not: its result depends on the steps
  s5 = em_staggered(p)
  _, flag = solve!(s5.sm; stepping=(nsteps=5, nsubsteps=1, maxbisec=1), kargsolve=s5.args, verbose=false)
  @test flag == 1
  eu5, eφ5 = errors(s5, ref)
  @info "staggered EM block: fixed loop of 5 × 1: relative error of u, φ vs the monolithic solution: $eu5, $eφ5"
  @test eu5 > 1e-5 && eφ5 > 1e-6
  log = get_log(s5.sm)
  @test num_steps(log) == 5 && all(==(1), log.nsub) && log.nbisect_total == 0
  @test log.τ ≈ (1:5) ./ 5 && all(==(0.2), log.Δτ)
  @test all(t -> all(>(0), t), log.times) && all(e -> all(isfinite, e), log.errors)
  s1 = em_staggered(p)
  solve!(s1.sm; stepping=(nsteps=1, nsubsteps=1, maxbisec=1), kargsolve=s1.args, verbose=false)
  @test errors(s1, ref)[1] > eu5                                            # fewer steps, larger error

  # with the tolerance, 1, 5 and 20 steps give the monolithic solution
  passes = Int[]
  for nsteps in (1, 5, 20)
    s = em_staggered(p)
    _, flag = solve!(s.sm; stepping=StaggeredStepping(; nsteps, max_substeps=50, coupling_tol=1e-8),
                     kargsolve=s.args, verbose=false)
    @test flag == 1
    eu, eφ = errors(s, ref)
    @test eu ≤ 1e-6 && eφ ≤ 1e-6
    log = get_log(s.sm)
    @test num_steps(log) == nsteps && log.τ[end] ≈ 1.0 && log.nbisect_total == 0
    @test all(e -> all(≤(1e-8), e), log.errors)
    push!(passes, num_substeps(log))
  end
  @info "staggered EM block: passes to 1e-8 with 1, 5, 20 steps: $passes"
  @test passes[1] ≥ 3                                                       # strongly coupled: more than one pass

  # the residual norm converges to the same solution
  s = em_staggered(p)
  _, flag = solve!(s.sm; stepping=StaggeredStepping(; nsteps=2, max_substeps=50, coupling_tol=1e-8, norm=:residual),
                   kargsolve=s.args, verbose=false)
  @test flag == 1
  eu, eφ = errors(s, ref)
  @test eu ≤ 1e-6 && eφ ≤ 1e-6
  @test all(l -> l ≥ 2, get_log(s.sm).nsub)                                # the residual needs two passes at least
end

@testset "StaggeredModel: Aitken relaxation" begin
  p = em_block()
  ref = em_monolithic(p)
  run(relaxation; injectivity=false) = begin
    s = em_staggered(p; injectivity)
    _, flag = solve!(s.sm; stepping=StaggeredStepping(; nsteps=2, max_substeps=60, coupling_tol=1e-8, relaxation),
                     kargsolve=s.args, verbose=false)
    @test flag == 1
    eu, eφ = errors(s, ref)
    @test eu ≤ 1e-6 && eφ ≤ 1e-6
    get_log(s.sm)
  end
  l0 = run(:none)
  la = run((:none, Aitken()))                                               # the displacement relaxed
  @info "staggered EM block: passes to 1e-8 without / with Aitken: $(num_substeps(l0)) / $(num_substeps(la))"
  @test num_substeps(la) < num_substeps(l0)
  @test all(ω -> ω[1] == 1.0 && 0.01 ≤ ω[2] ≤ 1.0, la.ω)
  @test all(ω -> ω == [1.0, 1.0], l0.ω)
  l2 = run((:none, Aitken(; ωmax=2.0)))                                     # over-relaxation allowed: same solution, no more passes
  @test num_substeps(l2) ≤ num_substeps(l0) && all(ω -> 0.01 ≤ ω[2] ≤ 2.0, l2.ω)
  lf = run((:none, 0.8))                                                    # a fixed factor: same solution
  @test all(ω -> ω[2] == 0.8, lf.ω)
  # with the injectivity measure and the injectivity line search the factor is cut to the injectivity step
  # (never active in this problem: the relaxed states are far from inverting elements), same solution
  li = run((:none, Aitken()); injectivity=true)
  @test num_substeps(li) == num_substeps(la)
  li2 = run((:none, Aitken(; ωmax=2.0)); injectivity=true)
  @test num_substeps(li2) == num_substeps(l2)
end

@testset "StaggeredModel: adaptive increment, rejection and rollback" begin
  p = em_block()
  ref = em_monolithic(p)
  # the passes a single step (Δτ0 = 1) needs: one fewer forces its rejection, the bisected steps go through
  s = em_staggered(p)
  solve!(s.sm; stepping=StaggeredStepping(; nsteps=1, max_substeps=60, coupling_tol=1e-8), kargsolve=s.args, verbose=false)
  n1 = get_log(s.sm).nsub[1]
  @test n1 ≥ 4
  K = n1 - 1
  s = em_staggered(p)
  stages = Tuple{Float64,Float64}[]
  steps = Tuple{Float64,Float64,Int}[]
  _, flag = @test_logs (:warn, r"bisection performed") match_mode = :any begin
    solve!(s.sm; stepping=StaggeredStepping(; nsteps=1, max_substeps=K, coupling_tol=1e-8, maxbisec=4, grow=2.0, easy=K),
           kargsolve=s.args, verbose=false,
           presolver=(τ, Δτ) -> push!(stages, (τ, Δτ)), on_step=(τ, Δτ, nsub) -> push!(steps, (τ, Δτ, nsub)))
  end
  @test flag == 1
  eu, eφ = errors(s, ref)
  @test eu ≤ 1e-6 && eφ ≤ 1e-6
  log = get_log(s.sm)
  @info "staggered EM block: adaptive from Δτ0 = 1 with $K passes at most: steps $(log.τ), passes $(log.nsub), bisections $(log.nbisect)"
  @test log.nbisect_total ≥ 1 && log.τ[end] ≈ 1.0 && num_steps(log) ≥ 2
  @test sum(log.nbisect) == log.nbisect_total && all(≤(K), log.nsub)
  # the presolver saw the rejected attempt (0, 1), then the halved increment from the same stage; every attempt
  # starts at the last accepted stage and the last one ends at τ = 1
  @test stages[1] == (0.0, 1.0) && stages[2] == (0.0, 0.5)
  @test issorted(first.(stages)) && stages[end][1] + stages[end][2] ≈ 1.0
  @test all(st -> st[1] ∈ first.(stages) || st[1] ≈ 1.0, steps)
  @test length(steps) == num_steps(log) && steps[end][1] ≈ 1.0 && all(st -> st[3] ≤ K, steps)
  # the increment grows after an accepted step (grow = 2, easy = K), never beyond Δτ0 = 1
  @test all(i -> steps[i][2] ≈ min(2 * log.Δτ[i], 1.0) || steps[i][1] ≈ 1.0, 1:length(steps))

  # rollback: with no bisection allowed the first attempt (one pass, far from the tolerance) is rejected and
  # the states, the lagged states and the Dirichlet values return to the start
  s = em_staggered(p)
  _, flag = @test_logs (:warn, r"bisection performed") (:warn, r"Maximum number of bisections") match_mode = :any begin
    solve!(s.sm; stepping=StaggeredStepping(; nsteps=1, max_substeps=1, coupling_tol=1e-12, maxbisec=0),
           kargsolve=s.args, verbose=false)
  end
  @test flag == 0
  @test all(iszero, get_free_dof_values(s.uh⁺)) && all(iszero, get_free_dof_values(s.φh⁺))
  @test all(iszero, s.sm.caches.x⁻[1]) && all(iszero, s.sm.caches.x⁻[2])
  @test maximum(abs, get_dirichlet_dof_values(s.Uφ)) == 0.0                 # back to the load of the last accepted step (0)
  @test maximum(abs, get_dirichlet_dof_values(get_fe_space(s.sm.state⁻[1]))) == 0.0
  @test num_steps(get_log(s.sm)) == 0 && get_log(s.sm).nattempts == 1
  # and the model can go on from there
  _, flag = solve!(s.sm; stepping=StaggeredStepping(; nsteps=2, max_substeps=60, coupling_tol=1e-8), kargsolve=s.args, verbose=false)
  @test flag == 1 && errors(s, ref)[1] ≤ 1e-6
end

@testset "StaggeredModel: presolver stages and hooks" begin
  p = em_block(; φtop=0.02)
  # the stage ramp with a StaggeredStepping: τ⁻ ∈ [0, 1] and the increment, consistent with the accepted steps
  s = em_staggered(p)
  stages = Tuple{Float64,Float64}[]
  subs = Tuple{Int,Tuple{Float64,Float64}}[]
  solve!(s.sm; stepping=StaggeredStepping(; nsteps=4, max_substeps=20, coupling_tol=1e-6), kargsolve=s.args, verbose=false,
         presolver=(τ, Δτ) -> push!(stages, (τ, Δτ)), on_substep=(k, e) -> push!(subs, (k, e)))
  @test stages == [(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)]
  @test length(subs) == num_substeps(get_log(s.sm)) && all(e -> all(≥(0), e[2]), subs)
  @test subs[end][2] == Tuple(get_log(s.sm).errors[end])
  # the fixed loop of a StaggeredStepping also passes the stage
  s = em_staggered(p)
  empty!(stages)
  solve!(s.sm; stepping=StaggeredStepping(; nsteps=4), kargsolve=s.args, verbose=false, presolver=(τ, Δτ) -> push!(stages, (τ, Δτ)))
  @test stages == [(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)]
  # the NamedTuple of earlier versions passes the index of the step (deprecated)
  s = em_staggered(p)
  idx = Any[]
  @test_deprecated solve!(s.sm; stepping=(nsteps=4, nsubsteps=1, maxbisec=1), kargsolve=s.args, verbose=false,
                          presolver=(τ, Δτ) -> push!(idx, (τ, Δτ)))
  Base.JLOptions().depwarn == 2 || @test idx == [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]   # (with --depwarn=error the call throws)
  # errors of the keywords
  @test_throws ErrorException solve!(s.sm; stepping=StaggeredStepping(; relaxation=(:none, :none, :none)), kargsolve=s.args, verbose=false)
  @test_throws ErrorException solve!(s.sm; stepping=(nsteps=1, nsubsteps=1, maxbisec=1), kargsolve=(s.args[1],), verbose=false)
end
