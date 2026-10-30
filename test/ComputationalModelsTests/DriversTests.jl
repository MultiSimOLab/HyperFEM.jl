# Drivers: the load stepping of `solve!(::StaticNonlinearModel)` after a failed attempt.
using Gridap, Gridap.FESpaces
using HyperFEM
using LinearAlgebra
using Test

# Small 2D hyperelastic block stretched by a Dirichlet condition
function drivers_test_problem()
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (4, 2))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  _, ∂Ψ∂F, ∂Ψ∂FF = physmodel()
  F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(
    BoundaryCondition("left", [0.0, 0.0]),
    BoundaryCondition("right", [0.6, 0.1], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F ∘ ∇(u)')))dΩ
  jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψ∂FF ∘ (F ∘ ∇(u)')) ⊙ ∇(du)'))dΩ
  (; res, jac, U, V, bc)
end

# A solver whose first `nfail` attempts fail without touching the state (to force bisections)
mutable struct FailFirst{N<:NewtonRaphson,L} <: Gridap.Algebra.NonlinearSolver
  nls::N
  nfail::Int
  calls::Int
  ls::L                       # the drivers read the linear solver of the Dirichlet projection here
end
FailFirst(nls, nfail, calls) = FailFirst(nls, nfail, calls, nls.ls)
HyperFEM.ComputationalModels.instantiate_caches(x, s::FailFirst, op) = HyperFEM.ComputationalModels.instantiate_caches(x, s.nls, op)
function Gridap.Algebra.solve!(x::AbstractVector, s::FailFirst, op::Gridap.Algebra.NonlinearOperator, cache)
  s.calls += 1
  s.calls <= s.nfail && return cache
  solve!(x, s.nls, op, cache)
end
HyperFEM.ComputationalModels.nls_converged(s::FailFirst) = s.calls > s.nfail && converged(s.nls)
HyperFEM.ComputationalModels.nls_message(s::FailFirst, msg) = nothing

@testset "Dirichlet values after a bisection" begin
  p = drivers_test_problem()
  dirichlet_at(Λ) = get_dirichlet_dof_values(TrialFESpace(p.V, p.bc, Λ))
  for ProjectDirichlet in (false, true)
    # every attempt fails (maxiter = 0): the driver bisects maxbisec times and gives up; the state, the
    # Dirichlet values of U and the load factor must be back at the last converged step (Λ = 0)
    nls = NewtonRaphson(LUSolver(); maxiter=0)
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
    x, flag = @test_logs (:warn, r"Maximum number of bisections") match_mode = :any begin
      solve!(m; stepping=(nsteps=2, maxbisec=1), ProjectDirichlet=ProjectDirichlet)
    end
    @test flag == 0
    @test x == zeros(length(x))
    @test get_dirichlet_dof_values(p.U) == dirichlet_at(0.0)
    @test get_dirichlet_dof_values(p.U) != dirichlet_at(0.25)
  end
  # a first failed attempt followed by converged steps: with the Dirichlet values reset to the last
  # converged load, the retry starts exactly as the run with the halved step, so both give the same x
  p = drivers_test_problem()
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=nls)
  xref, _ = solve!(m; stepping=(nsteps=4, maxbisec=0), ProjectDirichlet=true)
  xref = copy(xref)
  p = drivers_test_problem()
  ff = FailFirst(NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-12), 1, 0)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=ff)
  Λs = Float64[]
  post = PostProcessor(m, post -> push!(Λs, post.Λ[post.iter]); is_vtk=false, filepath=mktempdir())
  x, flag = @test_logs (:warn, r"Bisection") match_mode = :any solve!(m; stepping=(nsteps=2, maxbisec=5), ProjectDirichlet=true, post=post)
  @test flag == 1 && Λs ≈ [0.25, 0.5, 0.75, 1.0]
  @test x == xref
end
