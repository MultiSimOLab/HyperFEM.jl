# Shared battery of the block preconditioners of stage S2 (design/solvers-s2-plan.md): run for `AMGSolver` in
# FieldSplitTests.jl (CI) and for `PETScSolver` in PETScFieldSplitTests.jl (local, needs GridapPETSc).
# `make(U; split, kwargs...)` builds the block solver under test from the multi-field trial space (the constructors
# `AMGSolver(U; …)` / `PETScSolver(U; …)`), with GMRES outside.
# Uses the problems and helpers of krylov_common.jl (included first).

"Linear electro-mechanics-like saddle point: elasticity coupled to a Poisson field, K_φφ negative definite."
function s2_saddle_problem(n)
  model = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5n, n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags=["left"])
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); dirichlet_tags=["left", "top"])
  U = MultiFieldFESpace([TrialFESpace(Vu, x -> zero(VectorValue{3,Float64})), TrialFESpace(Vφ, [x -> 0.0, x -> 1.0])])
  V = MultiFieldFESpace([Vu, Vφ])
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  λ, μ, εr, κ = 1.0, 1.0, 0.5, 0.3                     # εr: permittivity; κ: strength of the (symmetric) coupling
  σ(e) = λ * tr(e) * one(e) + 2μ * e
  a((u, φ), (v, ψ)) = ∫(ε(v) ⊙ (σ ∘ ε(u)) - εr * (∇(φ) ⋅ ∇(ψ)) + κ * (divergence(v) * φ + divergence(u) * ψ))dΩ
  l((v, ψ)) = ∫(v ⋅ VectorValue(0.0, 0.0, -1.0))dΩ
  op = AffineFEOperator(a, l, U, V)
  U, V, get_matrix(op), get_vector(op)
end

"""
    s2_battery(name, make)

The S2 tests for the block solver built by `make(U; split, rtol, maxiter)`: the saddle point is solved to the LU
solution with the three splits (where the monolithic AMG fails or crawls), the block near-null spaces reach the
blocks, bounded iterations under refinement (additive and Schur), the lagged preconditioner across a new matrix
object, and the Newton trajectory of LU on the nonlinear EM problem with the three splits and both backends —
the test S1 had to move to mechanics.
"""
function s2_battery(name, make; growth=1.3)
  @testset "$name" begin
    U, V, A, b = s2_saddle_problem(2)
    xlu = A \ b

    @testset "field_ranges" begin
      r = field_ranges(U)
      @test length(r) == 2
      @test first(r[1]) == 1 && last(r[2]) == num_free_dofs(U)
      @test length(r[1]) == num_free_dofs(U[1]) && length(r[2]) == num_free_dofs(U[2])
      N = near_nullspace(U)
      Ns = split_nullspace(N, r)
      @test size(Ns[1]) == (length(r[1]), 6) && size(Ns[2]) == (length(r[2]), 1)
      o = block_order(U[1])                              # the clamped face constrains whole nodes: blocks of 3
      @test o isa Vector{Int} && length(o) == length(r[1]) && isperm(o)
      @test block_order(U[2]) === nothing                # scalar field
      model = get_background_model(get_triangulation(U[1]))
      Vm = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1); dirichlet_tags=["left"], dirichlet_masks=[(true, false, false)])
      @test block_order(TrialFESpace(Vm, x -> zero(VectorValue{3,Float64}))) === nothing   # partially constrained nodes
    end

    @testset "same solution as LU with every split" begin
      b0 = copy(b)
      for split in (:additive, :multiplicative, :schur)
        x, ns = s1_solve(make(U; split, rtol=1e-10), A, b)
        @test linear_converged(ns)
        @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
        @test b == b0                                    # the negated equations are restored
        @info "$name, $split: $(linear_iterations(ns)) iterations"
      end
    end

    @testset "bounded iterations under refinement" begin
      for split in (:additive, :schur)
        its = Int[]
        for n in (2, 4, 6)
          Un, Vn, An, bn = s2_saddle_problem(n)
          _, ns = s1_solve(make(Un; split, rtol=1e-8), An, bn)
          @test linear_converged(ns)
          push!(its, linear_iterations(ns))
        end
        @info "$name: $split iterations under refinement $its"
        @test maximum(its) ≤ growth * minimum(its) + 4           # the coarsest mesh can be unusually cheap (few GAMG levels)
      end
    end

    @testset "lagged preconditioner, new matrix object" begin
      # reuse = 3: the second setup keeps the hierarchies; a new matrix object (a new pattern for PETSc) rebuilds
      # them whatever the lag says — the path that held a use-after-free in S1
      solver = make(U; split=:additive, rtol=1e-10, reuse=3)
      ns = numerical_setup(symbolic_setup(solver, A), A)
      x = zeros(length(b))
      solve!(x, ns, b)
      @test norm(x - xlu) ≤ 1e-7 * norm(xlu)
      A2 = 2 * A                                       # same pattern, scaled values, same object type
      numerical_setup!(ns, A2)                         # new object → rebuild (the lag is not applied)
      @test preconditioner_builds(ns) == 2
      fill!(x, 0.0); solve!(x, ns, b)
      @test linear_converged(ns)
      @test norm(x - xlu / 2) ≤ 1e-7 * norm(xlu)
      numerical_setup!(ns, A2)                         # same object, lagged: no rebuild
      @test preconditioner_builds(ns) == 2
      fill!(x, 0.0); solve!(x, ns, b)
      @test norm(x - xlu / 2) ≤ 1e-7 * norm(xlu)
    end

    @testset "Newton trajectory of LU (nonlinear EM)" begin
      p = s1_em_problem()
      for backend in S1_BACKENDS
        @testset "$backend" begin
          ref = s1_solve_nl(LUSolver(), p; backend)
          @test ref.flag
          for split in (:additive, :multiplicative, :schur)
            r = s1_solve_nl(make(p.U; split, rtol=1e-8), p; backend)
            @test r.flag
            @test r.its == ref.its
            @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
          end
        end
      end
    end
  end
end
