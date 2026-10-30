# StaticLinearModel: one constructor for the three forms of the problem, and a numerical setup of the
# linear solver only when the matrix was re-assembled.
using Gridap, Gridap.FESpaces, Gridap.Algebra
using HyperFEM
using LinearAlgebra
using Test

mutable struct CountingLinearSolver <: Gridap.Algebra.LinearSolver
  setups::Int
  updates::Int
end
struct CountingLinearSS <: Gridap.Algebra.SymbolicSetup
  ls::CountingLinearSolver
end
mutable struct CountingLinearNS <: Gridap.Algebra.NumericalSetup
  ls::CountingLinearSolver
  ns::Gridap.Algebra.LUNumericalSetup
end
Gridap.Algebra.symbolic_setup(ls::CountingLinearSolver, A::AbstractMatrix) = CountingLinearSS(ls)
Gridap.Algebra.numerical_setup(ss::CountingLinearSS, A::AbstractMatrix) =
  (ss.ls.setups += 1; CountingLinearNS(ss.ls, numerical_setup(symbolic_setup(LUSolver(), A), A)))
Gridap.Algebra.numerical_setup!(ns::CountingLinearNS, A::AbstractMatrix) = (ns.ls.updates += 1; numerical_setup!(ns.ns, A); ns)
Gridap.Algebra.solve!(x::AbstractVector, ns::CountingLinearNS, b::AbstractVector) = solve!(x, ns.ns, b)

function linear_test_problem()
  geometry = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(BoundaryCondition("left", 1.0))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 1.0)
  a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
  l(v) = ∫(1.0 * v)dΩ
  l2(v) = ∫(2.0 * v)dΩ
  (; a, l, l2, U, V, bc, dΩ)
end

@testset "StaticLinearModel: setup only when K changed" begin
  p = linear_test_problem()
  # reference: Gridap's affine problem
  xref = get_free_dof_values(solve(LUSolver(), AffineFEOperator(p.a, p.l, p.U, p.V)))
  ls = CountingLinearSolver(0, 0)
  m = StaticLinearModel(p.l, p.a, p.U, p.V, p.bc; ls)
  @test (ls.setups, ls.updates) == (1, 0) && !m.caches.stale
  x = solve!(m; Assembly=false)                      # the system of the constructor, no new setup
  @test x == xref && (ls.setups, ls.updates) == (1, 0)
  x = solve!(m; Assembly=true)                       # re-assembled: one update
  @test x == xref && (ls.setups, ls.updates) == (1, 1) && !m.caches.stale
  x = solve!(m; Assembly=false)
  @test x == xref && (ls.setups, ls.updates) == (1, 1)
  # a right-hand side given to solve!, matrix untouched
  b = copy(m.caches.b)
  x = solve!(m, b; Assembly=false)
  @test x == xref && (ls.setups, ls.updates) == (1, 1)
  x = solve!(m, b; Assembly=true)
  @test x == xref && (ls.setups, ls.updates) == (1, 2)
  # the functor over free values: the matrix of the last assembly
  y = copy(b)
  @test m(y; Assembly=false) == xref && (ls.setups, ls.updates) == (1, 2)
  # a matrix changed by other means: the caller marks the cache
  m.caches.stale = true
  x = solve!(m; Assembly=false)
  @test x == xref && (ls.setups, ls.updates) == (1, 3)

  # matrix-only model and the model with several linear forms: the same constructor
  ls = CountingLinearSolver(0, 0)
  m0 = StaticLinearModel(p.a, p.U, p.V, p.bc; ls)
  @test m0.lf === nothing && (ls.setups, ls.updates) == (1, 0)
  @test solve!(m0, b; Assembly=false) == xref && (ls.setups, ls.updates) == (1, 0)
  ls = CountingLinearSolver(0, 0)
  mv = StaticLinearModel([p.l, p.l2], p.a, p.U, p.V, p.bc; ls)
  @test mv isa StaticLinearModel{<:AbstractVector} && (ls.setups, ls.updates) == (1, 0)
  xv = solve!(mv; Assembly=false)
  @test (ls.setups, ls.updates) == (1, 0)
  @test length(HyperFEM.ComputationalModels.blocks(xv)) == 2
  x1, x2 = HyperFEM.ComputationalModels.blocks(xv)
  # without the Dirichlet lift (load vectors alone); the second load is twice the first
  @test norm(x1) > 0 && x2 ≈ 2 .* x1
  xv = solve!(mv; Assembly=true)
  @test (ls.setups, ls.updates) == (1, 1)
  @test_throws MethodError StaticLinearModel(1.0, p.a, p.U, p.V, p.bc)

  # the functor over an FE function (L2 projection through the model's backend, no Dirichlet lift)
  ls = CountingLinearSolver(0, 0)
  m0 = StaticLinearModel(p.a, p.U, p.V, p.bc; ls)
  fh() = interpolate_everywhere(x -> x[1], p.U)       # the functor solves into the free values of its argument
  bref = assemble_vector(v -> ∫(fh() * v)p.dΩ, p.V)
  xf = copy(m0(fh(); Measure=p.dΩ))
  @test xf ≈ m0.caches.K \ bref && (ls.setups, ls.updates) == (1, 0)
  @test copy(m0(fh(); Measure=p.dΩ, Assembly=true)) == xf && (ls.setups, ls.updates) == (1, 1)
end
