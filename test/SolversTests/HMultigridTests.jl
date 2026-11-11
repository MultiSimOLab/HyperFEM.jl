# S4-B step 2 (design/solvers-s4b-plan.md §3.3–3.4, option J): `HMultigridSolver`, the Julia V-cycle on a refined mesh
# hierarchy — Galerkin coarse operators, Chebyshev smoothing on the inverted nodal blocks, an injected coarse solver —
# as the preconditioner of CG/GMRES inside HyperFEM's Newton. Same solution as LU; Krylov iterations nearly independent
# of the number of levels (h-independence); the cycle is symmetric (CG valid); Newton trajectory equal to LU on
# hyperelasticity with the Dirichlet projection and lagged rebuilds; non-finite or indefinite tangents fail the solve.
include("krylov_common.jl")
import AlgebraicMultigrid
using GridapGmsh
using HyperFEM.Solvers: nodal_blocks, level_spectra

const HMG_MODELS = joinpath(@__DIR__, "..", "models")

"Vector P1/Q1 test space with Dirichlet on `tags`."
hmg_space(m, tags; p=1) = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{num_cell_dims(m),Float64}, p); dirichlet_tags=tags, conformity=:H1)

"Linear elasticity on the fine level of `h` (and the hMG solver on it): matrix, right-hand side, solver."
function hmg_elasticity(h, tags; krylov=:cg, coarse=LUSolver(), kwargs...)
  V = hmg_space(fine_model(h), tags)
  U = TrialFESpace(V, x -> zero(VectorValue{3,Float64}))
  dΩ = Measure(Triangulation(fine_model(h)), 2)
  λ, μ = 2.0, 1.0
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩ
  l(v) = ∫(v ⋅ VectorValue(0.0, 0.0, -1.0))dΩ
  op = AffineFEOperator(a, l, U, V)
  ls = HMultigridSolver(h, m -> hmg_space(m, tags); krylov, coarse, kwargs...)
  (; A=get_matrix(op), b=get_vector(op), ls, V)
end

"Nonlinear mechanics (neo-Hookean bar, clamped and sheared) on the fine level of a Cartesian hierarchy."
function hmg_mech_problem(nlevels)
  coarse = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5, 1, 1))
  labels = get_face_labeling(coarse)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  h = MeshHierarchy(coarse, nlevels)
  model = fine_model(h)
  physmodel = NeoHookean3D(λ=10.0, μ=1.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  D = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]), BoundaryCondition("top", [0.05, 0.0, 0.0], Λ -> Λ))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D, conformity=:H1)
  U = TrialFESpace(V, D, 0.0)
  k = Kinematics(Mechano, Solid)
  res(Λ) = (u, v) -> HyperFEM.residual(physmodel, k, u, v, dΩ)
  jac(Λ) = (u, du, v) -> HyperFEM.jacobian(physmodel, k, u, du, v, dΩ)
  space(m) = TestFESpace(Triangulation(m), ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D, conformity=:H1)
  (; res, jac, U, V, D, h, space)
end

@testset "h-multigrid solver (S4-B step 2)" begin

  @testset "nodal blocks" begin
    coarse = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
    labels = get_face_labeling(coarse)
    add_tag_from_tags!(labels, "left", CartesianTags.face0YZ⁺)
    V = TestFESpace(coarse, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags="left",
                    dirichlet_masks=[(true, false, false)])
    bl = nodal_blocks(V)
    @test sum(length, bl) == num_free_dofs(V)
    @test sort(reduce(vcat, bl)) == 1:num_free_dofs(V)
    @test count(b -> length(b) == 3, bl) == 27 - 9                   # interior nodes: 3 components
    @test count(b -> length(b) == 2, bl) == 9                        # x = 0 nodes: only x is Dirichlet
    Vs = TestFESpace(coarse, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags="left")
    @test all(b -> length(b) == 1, nodal_blocks(Vs))
    X = MultiFieldFESpace([V, Vs])
    bx = nodal_blocks(X)
    @test sum(length, bx) == num_free_dofs(X) && maximum(maximum, bx) == num_free_dofs(X)
  end

  @testset "same solution as LU, h-independent iterations (tets and hexes, CG and GMRES)" begin
    for (name, coarse) in (("tets", GmshDiscreteModel(joinpath(HMG_MODELS, "s4b_tet.msh"))),
                           ("hexes", CartesianDiscreteModel((0, 1, 0, 0.5, 0, 0.3), (4, 2, 2))))
      name == "hexes" && add_tag_from_tags!(get_face_labeling(coarse), "left", CartesianTags.face0YZ⁺)
      tags = ["left"]
      its = Int[]
      for nlevels in (2, 3, 4)
        h = MeshHierarchy(coarse, nlevels)
        t = hmg_elasticity(h, tags; rtol=1e-10, maxiter=100)
        x, ns = s1_solve(t.ls, t.A, t.b)
        @test linear_converged(ns)
        @test norm(x - t.A \ t.b) ≤ 1e-8 * norm(x)
        @test all(((lo, hi),) -> lo > 0 && hi > lo, level_spectra(ns))
        push!(its, linear_iterations(ns))
        if nlevels == 3
          tg = hmg_elasticity(h, tags; krylov=:gmres, rtol=1e-10, maxiter=100)
          xg, nsg = s1_solve(tg.ls, tg.A, tg.b)
          @test linear_converged(nsg) && norm(xg - x) ≤ 1e-7 * norm(x)
        end
      end
      @info "hMG $name: CG iterations with 2, 3, 4 levels" its
      # ±30 % between the multilevel hierarchies (DS4B-6); the two-level one (coarse solved exactly) is the lower bound
      @test its[3] ≤ 1.3 * its[2]
      @test maximum(its) ≤ 1.5 * minimum(its)
      @test maximum(its) ≤ 30
    end
  end

  @testset "the V-cycle is symmetric positive definite on an SPD tangent" begin
    coarse = CartesianDiscreteModel((0, 1, 0, 0.5, 0, 0.5), (2, 1, 1))
    add_tag_from_tags!(get_face_labeling(coarse), "left", CartesianTags.face0YZ⁺)
    h = MeshHierarchy(coarse, 3)
    t = hmg_elasticity(h, ["left"])
    ns = numerical_setup(symbolic_setup(t.ls, t.A), t.A)
    n = size(t.A, 1)
    M = zeros(n, n)
    P = HyperFEM.Solvers.HMultigridPrec(ns)
    for j in 1:n
      e = zeros(n); e[j] = 1.0
      ldiv!(view(M, :, j), P, e)
    end
    @test norm(M - M') ≤ 1e-10 * norm(M)
    @test minimum(eigvals(Symmetric((M + M') / 2))) > 0
  end

  @testset "coarse AMG, lagged rebuilds, forcing-term hooks" begin
    coarse = GmshDiscreteModel(joinpath(HMG_MODELS, "s4b_tet.msh"))
    h = MeshHierarchy(coarse, 3)
    Vc = hmg_space(level_model(h, 1), ["left"])
    N1 = near_nullspace(TrialFESpace(Vc))
    t = hmg_elasticity(h, ["left"]; coarse=AMGSolver(:cg; rtol=1e-10, nullspace=N1), reuse=3, rtol=1e-8)
    x, ns = s1_solve(t.ls, t.A, t.b)
    @test linear_converged(ns) && norm(x - t.A \ t.b) ≤ 1e-6 * norm(x)
    @test preconditioner_builds(ns) == 1
    numerical_setup!(ns, t.A); numerical_setup!(ns, t.A)
    @test preconditioner_builds(ns) == 1                             # setups 2 and 3 reuse the hierarchy
    numerical_setup!(ns, t.A)
    @test preconditioner_builds(ns) == 2                             # setup 4 rebuilds
    numerical_setup!(ns, copy(t.A))
    @test preconditioner_builds(ns) == 3                             # a new matrix object always rebuilds
    # the Galerkin operators are refilled in the pattern fixed by the first product: the refill of a rescaled and perturbed
    # tangent equals the sparse product itself
    A3 = copy(t.A)
    A3.nzval .*= 2.0 .+ 0.3 .* sin.(1:nnz(A3))                       # same pattern, every value changed (symmetry not needed)
    numerical_setup!(ns, A3)
    @test preconditioner_builds(ns) == 4
    Ac = ns.levels[1].A
    @test Ac ≈ transpose(t.ls.Ps[2]) * A3 * t.ls.Ps[2] rtol = 1e-13
    @test ns.Acoarse ≈ transpose(t.ls.Ps[1]) * Ac * t.ls.Ps[1] rtol = 1e-13
    numerical_setup!(ns, t.A)                                        # back on the SPD tangent: the symmetric refill (exact
    Ac = ns.levels[1].A                                              # cancellations in the accumulators) equals the product too
    @test Ac ≈ transpose(t.ls.Ps[2]) * t.A * t.ls.Ps[2] rtol = 1e-13
    @test ns.Acoarse ≈ transpose(t.ls.Ps[1]) * Ac * t.ls.Ps[1] rtol = 1e-13
    # with reuse the coarse levels lag but the fine smoother follows the current tangent (its interval is re-estimated):
    # CG on 3A with the coarse operators of A converges, where a stale fine interval would make Chebyshev diverge
    A2 = copy(t.A)
    ns2 = numerical_setup(symbolic_setup(t.ls, A2), A2)
    λ1 = level_spectra(ns2)[end]
    A2.nzval .*= 3.0
    numerical_setup!(ns2, A2)                                        # setup 2 of 3: no rebuild of the coarse levels
    @test preconditioner_builds(ns2) == 1
    @test level_spectra(ns2)[end][2] ≈ λ1[2] rtol = 1e-8             # D⁻¹A is scale-invariant: re-estimated on 3A, same value
    solve!(x, ns2, t.b)
    @test linear_converged(ns2) && norm(x - A2 \ t.b) ≤ 1e-6 * norm(x)
    @test tolerance(ns) == 1e-8
    set_tolerance!(ns, 1e-2); solve!(x, ns, t.b)
    loose = linear_iterations(ns)
    set_tolerance!(ns, nothing); solve!(x, ns, t.b)
    @test tolerance(ns) == 1e-8 && linear_iterations(ns) > loose
  end

  @testset "non-finite and indefinite tangents fail the solve at once" begin
    coarse = CartesianDiscreteModel((0, 1, 0, 0.5, 0, 0.5), (2, 1, 1))
    add_tag_from_tags!(get_face_labeling(coarse), "left", CartesianTags.face0YZ⁺)
    h = MeshHierarchy(coarse, 3)
    t = hmg_elasticity(h, ["left"])
    A = copy(t.A)
    ns = numerical_setup(symbolic_setup(t.ls, A), A)
    x = zeros(size(A, 1))
    A.nzval[1] = NaN
    numerical_setup!(ns, A)
    solve!(x, ns, t.b)
    @test !linear_converged(ns) && linear_iterations(ns) == 0
    @test !linear_indefinite(ns)                                     # non-finite is not the instability signal (S5)
    copyto!(A.nzval, t.A.nzval)
    numerical_setup!(ns, A)
    solve!(x, ns, t.b)
    @test linear_converged(ns)                                       # a finite tangent recovers
    # indefinite: elasticity − s · mass has negative low modes
    dΩ = Measure(Triangulation(fine_model(h)), 2)
    U = TrialFESpace(t.V, x -> zero(VectorValue{3,Float64}))
    Mm = assemble_matrix((u, v) -> ∫(300.0 * (v ⋅ u))dΩ, U, t.V)
    Bi = t.A - Mm
    @test minimum(eigvals(Symmetric(Matrix(Bi)))) < 0
    numerical_setup!(ns, Bi)
    @test any(((lo, hi),) -> lo < 0 || isnan(lo), level_spectra(ns))   # a negative Ritz value, or non-SPD nodal blocks
    @test occursin(r"indefinite|blocks", HyperFEM.Solvers.linear_info(ns))
    solve!(x, ns, t.b)
    @test !linear_converged(ns) && linear_iterations(ns) == 0
    @test linear_indefinite(ns)                                      # the S5 signal: HybridTangent switches on it
    numerical_setup!(ns, t.A)
    solve!(x, ns, t.b)
    @test linear_converged(ns) && !linear_indefinite(ns)
  end

  @testset "Newton trajectory of LU inside the drivers (Dirichlet projection, load steps, reuse)" begin
    p = hmg_mech_problem(3)
    for backend in S1_BACKENDS
      ref = s1_solve_nl(LUSolver(), p; backend)
      @test ref.flag
      Vc = p.space(level_model(p.h, 1))
      N1 = near_nullspace(TrialFESpace(Vc))
      for (name, ls) in (("CG, coarse LU", HMultigridSolver(p.h, p.space; krylov=:cg, rtol=1e-10, maxiter=200)),
                         ("CG, coarse LU, reuse 3", HMultigridSolver(p.h, p.space; krylov=:cg, rtol=1e-10, maxiter=200, reuse=3)),
                         ("GMRES, coarse AMG", HMultigridSolver(p.h, p.space; krylov=:gmres, rtol=1e-10, maxiter=200,
                                                                coarse=AMGSolver(:cg; rtol=1e-10, nullspace=N1))))
        @testset "$name ($backend)" begin
          r = s1_solve_nl(ls, p; backend)
          @test r.flag
          @test r.its == ref.its
          @test maximum(abs.(log10.(r.hist) .- log10.(ref.hist))) < 0.5
          @test norm(r.x - ref.x) ≤ 1e-7 * norm(ref.x)
          @test all(≤(40), r.lin)
        end
      end
    end
  end
end
