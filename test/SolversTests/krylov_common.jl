# Shared battery of the Krylov + AMG solvers of stage S1 (design/solvers.md): run for `AMGSolver` in
# KrylovSolversTests.jl (CI) and for `PETScSolver` in PETScSolverTests.jl (local, needs GridapPETSc).
# `make(krylov; rtol, maxiter, reuse, nullspace)` builds the solver under test with the Krylov method
# `krylov` (:cg or :gmres) and the common options.
using Gridap, Gridap.Algebra, Gridap.FESpaces
using HyperFEM
using LinearAlgebra, SparseArrays
using Test

const S1_BACKENDS = Tuple(Symbol.(strip.(split(get(ENV, "HYPERFEM_TEST_BACKENDS", "lazy,compiled"), ","))))
:compiled in S1_BACKENDS && @eval using HyperFEMKernels

# --- linear elasticity on a bar, n cells across ----------------------------------------------------------------
function s1_elasticity(n; dirichlet=true, order=1)
  model = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5n, n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "left", CartesianTags.face0YZ⁺)
  reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  V = dirichlet ? TestFESpace(model, reffe; dirichlet_tags=["left"]) : TestFESpace(model, reffe)
  U = dirichlet ? TrialFESpace(V, x -> zero(VectorValue{3,Float64})) : TrialFESpace(V)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2order)
  λ, μ = 1.0, 1.0
  σ(ε) = λ * tr(ε) * one(ε) + 2μ * ε
  a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩ
  l(v) = ∫(v ⋅ VectorValue(0.0, 0.0, -1.0))dΩ
  op = AffineFEOperator(a, l, U, V)
  U, V, get_matrix(op), get_vector(op)
end

"Solves A x = b with `ls`; returns x and the numerical setup."
function s1_solve(ls, A, b)
  ns = numerical_setup(symbolic_setup(ls, A), A)
  x = zeros(length(b))
  solve!(x, ns, b)
  x, ns
end

# --- nonlinear mechanics: a neo-Hookean bar clamped at one end and sheared on its top face (SPD tangents along the
# --- path, like the mechanical cases of the solver benchmarks; the stretched bar of the Dirichlet simulation test
# --- has an indefinite tangent after the projection of the first step, which CG rightly refuses) -------------------
function s1_mech_problem()
  model = CartesianDiscreteModel((0, 1, 0, 0.1, 0, 0.1), (10, 3, 2))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  physmodel = NeoHookean3D(λ=10.0, μ=1.0)
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  D = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]), BoundaryCondition("top", [0.05, 0.0, 0.0], Λ -> Λ))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D, conformity=:H1)
  U = TrialFESpace(V, D, 0.0)
  k = Kinematics(Mechano, Solid)
  res(Λ) = (u, v) -> HyperFEM.residual(physmodel, k, u, v, dΩ)
  jac(Λ) = (u, du, v) -> HyperFEM.jacobian(physmodel, k, u, du, v, dΩ)
  (; res, jac, U, V, D)
end

# --- nonlinear electro-mechanics (the EM box of the solver benchmarks, coarse): a symmetric indefinite tangent ----------
function s1_em_problem()
  model = CartesianDiscreteModel((0.0, 0.1, 0.0, 0.01, 0.0, 0.002), (8, 2, 2))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixedup", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "topsuf", CartesianTags.faceXY1⁺)
  add_tag_from_vertex_filter!(labels, "midsuf", model, x -> abs(x[3] - 0.001) < 1e-9)
  physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
  k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
  Ω = Triangulation(model); dΩ = Measure(Ω, 2)
  Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
  Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", 5e-5, Λ -> Λ))
  D = MultiFieldBC([Du, Dφ])
  Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
  Vφ = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
  V = MultiFieldFESpace([Vu, Vφ])
  U = MultiFieldFESpace([TrialFESpace(Vu, Du, 1.0), TrialFESpace(Vφ, Dφ, 1.0)])
  res(Λ) = ((u, φ), (v, vφ)) -> HyperFEM.residual(physmodel, k, (u, φ), (v, vφ), dΩ)
  jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> HyperFEM.jacobian(physmodel, k, (u, φ), (du, dφ), (v, vφ), dΩ)
  (; res, jac, U, V, D)
end

"Solves `p` (a problem above) with the Newton on `ls`; returns the solution, the flag and the Newton counters."
function s1_solve_nl(ls, p; backend, nsteps=2)
  nls = NewtonRaphson(ls; maxiter=20, atol=1e-14, rtol=1e-6)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend)
  x, flag = solve!(m; stepping=(nsteps=nsteps, maxbisec=0), ProjectDirichlet=true)   # the projection uses the same linear solver
  (; x=copy(x), flag=flag == 1, its=num_iterations(nls), reason=stop_reason(nls), U=p.U)
end

"""
    s1_battery(name, make; growth=1.3)

The S1 tests for the solver built by `make(krylov; rtol, maxiter, reuse, nullspace)`: same solution as LU, rigid
modes reduce the iterations, bounded iterations under refinement, non-convergence reported, lagged
preconditioner, and the Newton trajectory of LU on nonlinear mechanics with both backends. `growth` is
the admitted ratio of CG iterations between the finest and the coarsest mesh (the plan's ±30 % for PETSc).
"""
function s1_battery(name, make; growth=1.3)
  @testset "$name" begin
    U, V, A, b = s1_elasticity(2)
    N = near_nullspace(U)
    xlu = A \ b

    @testset "same solution as LU" begin
      for ls in (make(:cg; rtol=1e-12, nullspace=N), make(:gmres; rtol=1e-12, nullspace=N), make(:cg; rtol=1e-12))
        x, ns = s1_solve(ls, A, b)
        @test linear_converged(ns)
        @test norm(x - xlu) ≤ 1e-8 * norm(xlu)
        @test linear_iterations(ns) > 0
        @test preconditioner_builds(ns) == 1
      end
    end

    @testset "rigid modes help" begin
      _, ns0 = s1_solve(make(:cg; rtol=1e-10), A, b)
      _, ns1 = s1_solve(make(:cg; rtol=1e-10, nullspace=N), A, b)
      @test linear_iterations(ns1) < linear_iterations(ns0)
    end

    @testset "bounded iterations under refinement" begin
      its = Int[]
      for n in (2, 4, 8)
        Un, Vn, An, bn = s1_elasticity(n)
        _, ns = s1_solve(make(:cg; rtol=1e-8, nullspace=near_nullspace(Un)), An, bn)
        @test linear_converged(ns)
        push!(its, linear_iterations(ns))
      end
      @info "$name: CG iterations under refinement $its"
      @test maximum(its) ≤ growth * minimum(its) + 2  # nearly mesh-independent (plus two for the smallest meshes)
    end

    @testset "non-convergence is reported" begin
      x, ns = s1_solve(make(:cg; rtol=1e-14, maxiter=2, nullspace=N), A, b)
      @test !linear_converged(ns)
      @test linear_iterations(ns) == 2
    end

    @testset "lagged preconditioner (reuse)" begin
      B = copy(A); B.nzval .*= 2.0
      ls = make(:cg; rtol=1e-10, reuse=3, nullspace=N)
      ns = numerical_setup(symbolic_setup(ls, A), A)
      @test preconditioner_builds(ns) == 1
      numerical_setup!(ns, B); @test preconditioner_builds(ns) == 1   # setup 2: reused
      numerical_setup!(ns, B); @test preconditioner_builds(ns) == 1   # setup 3: reused
      numerical_setup!(ns, B); @test preconditioner_builds(ns) == 2   # setup 4: rebuilt
      x = zeros(length(b)); solve!(x, ns, b)
      @test linear_converged(ns)
      @test norm(x - xlu / 2) ≤ 1e-8 * norm(xlu)
      # with a stale preconditioner the Krylov still converges to the solution of the current operator
      ls = make(:cg; rtol=1e-10, reuse=10, nullspace=N)
      ns = numerical_setup(symbolic_setup(ls, A), A)
      numerical_setup!(ns, B)
      fill!(x, 0.0); solve!(x, ns, b)
      @test linear_converged(ns)
      @test norm(x - xlu / 2) ≤ 1e-8 * norm(xlu)
      @test preconditioner_builds(ns) == 1
      # values updated in place in the same matrix object (what the Newton cache does)
      ls = make(:cg; rtol=1e-10, nullspace=N)
      ns = numerical_setup(symbolic_setup(ls, A), A)
      A.nzval .*= 2.0
      numerical_setup!(ns, A)
      fill!(x, 0.0); solve!(x, ns, b)
      @test norm(x - xlu / 2) ≤ 1e-8 * norm(xlu)
      A.nzval ./= 2.0
    end

    @testset "Newton trajectory of LU (nonlinear mechanics)" begin
      p = s1_mech_problem()
      Nm = near_nullspace(p.U)
      for backend in S1_BACKENDS
        @testset "$backend" begin
          ref = s1_solve_nl(LUSolver(), p; backend)
          @test ref.flag
          @test ref.its ≥ 2
          # the Krylov tolerance (1e-8) is well below the Newton's (1e-6) but not at round-off, where a Krylov
          # method stagnates: same iterations; both solutions satisfy the Newton tolerance, so they agree to the
          # accuracy that tolerance gives through the conditioning of the tangent, not to the Krylov tolerance
          for krylov in (:cg, :gmres)
            r = s1_solve_nl(make(krylov; rtol=1e-8, nullspace=Nm), p; backend)
            @test r.flag
            @test r.its == ref.its
            @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
          end
          # a Krylov solve that cannot converge fails the Newton like a line search (no plan B: the step fails)
          bad = s1_solve_nl(make(:cg; rtol=1e-8, maxiter=1, nullspace=Nm), p; backend)
          @test !bad.flag
          @test bad.reason === :linear_failed
        end
      end
    end
  end
end
