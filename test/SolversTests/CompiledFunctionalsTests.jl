# Energy and injectivity step of the drivers with the compiled backend (solver stage S3): the line searches of
# `StaticNonlinearModel(…; backend = :compiled)` evaluate the model's energy and the injectivity step with
# HyperFEMKernels' `KernelFunctional` (kept in the step cache, re-captured per load factor) instead of Gridap's
# lazy evaluation; same values and the same Newton/line-search history as the lazy backend. Needs HyperFEMKernels.
using Gridap, Gridap.Algebra, Gridap.FESpaces
using HyperFEM
using HyperFEMKernels
using HyperFEMKernels: KernelFunctional
using HyperFEM.ComputationalModels: model_operator
using HyperFEM.Solvers: energy, max_step
using LinearAlgebra, Random, Test

function cf_problem(; neumann=false, stretch=0.2)
  geometry = CartesianDiscreteModel((0, 2, 0, 1), (6, 3))
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "left", [1, 3, 7])
  add_tag_from_tags!(labels, "right", [2, 4, 8])
  add_tag_from_tags!(labels, "top", [6])
  physmodel = NonlinearMooneyRivlin2D(λ=2.0, μ1=1.0, μ2=1.0, α1=1.0, α2=1.0)
  km = Kinematics(Mechano, Solid)
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, 2)
  bc = BoundaryConditions(BoundaryCondition("left", [0.0, 0.0]),
                          BoundaryCondition("right", [2stretch, 0.3], EvolutionFunctions.ramp()))
  N_bc = BoundaryConditions(BoundaryCondition("top", [0.05, -0.3], EvolutionFunctions.ramp()))
  reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  res(Λ) = (u, v) -> residual(physmodel, km, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, km, u, du, v, dΩ)
  en(Λ) = u -> energy(physmodel, km, u, dΩ)
  nc = neumann ? NeumannConditions(geometry, N_bc, 2) : nothing
  (; res, jac, en, U, V, bc, dΩ, neumann=nc)
end

@testset "compiled energy and injectivity step (S3)" begin
  rng = MersenneTwister(5)
  @testset "same values as the lazy evaluation (neumann = $neumann)" for neumann in (false, true)
    p = cf_problem(; neumann)
    nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10)
    ml = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls, neumann=p.neumann, energy=p.en, injectivity=p.dΩ)
    mc = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls, neumann=p.neumann, energy=p.en, injectivity=p.dΩ, backend=:compiled)
    @test ml.caches.functionals[] === nothing
    @test mc.caches.functionals[].energy isa KernelFunctional          # built with the model (Λ = 1)
    x, _ = solve!(ml; stepping=(nsteps=2, maxbisec=0))
    for (Λ, load) in ((1.0, 1.0), (0.5, 0.5))
      opl = model_operator(ml, ml.caches.op, Λ, load)
      opc = model_operator(mc, mc.caches.op, Λ, load)
      for _ in 1:3
        x0 = x .+ 0.01 .* randn(rng, length(x))
        @test energy(opc, x0) ≈ energy(opl, x0) rtol = 1e-12
        dx = 0.1 .* randn(rng, length(x))
        for (β, jmin) in ((1 - 1e-6, 0.0), (0.5, 1e-3))
          @test max_step(opc, x0, dx, β, jmin) ≈ max_step(opl, x0, dx, β, jmin) rtol = 1e-12
        end
      end
    end
    # the energy functional is updated per load factor, not rebuilt: same tables
    f1 = mc.caches.functionals[].energy
    model_operator(mc, mc.caches.op, 0.3, 0.3)
    f2 = mc.caches.functionals[].energy
    @test f2 !== f1 && f2.groups[1].tables === f1.groups[1].tables
    @test mc.caches.functionals[].injectivity isa KernelFunctional && mc.caches.functionals[].injectivity.reduce === :min
  end

  @testset "Armijo + injectivity line search: same history as the lazy backend" begin
    p = cf_problem()
    hist = Dict{Symbol,Any}()
    for backend in (:lazy, :compiled)
      ls = InjectivityLS(armijo=true, c1=1e-4, maxiter=20)
      nls = NewtonRaphson(LUSolver(); maxiter=30, atol=1e-10, rtol=1e-10, linesearch=ls)
      m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls, injectivity=p.dΩ, energy=p.en, backend)
      x, flag = solve!(m; stepping=(nsteps=2, maxbisec=0), ProjectDirichlet=true)
      @test flag == 1 && converged(nls)
      hist[backend] = (x=copy(x), steps=copy(step_history(nls)), its=num_iterations(nls))
    end
    @test hist[:compiled].its == hist[:lazy].its
    @test hist[:compiled].steps ≈ hist[:lazy].steps rtol = 1e-8
    @test hist[:compiled].x ≈ hist[:lazy].x rtol = 1e-8
    @test all(0 .< hist[:compiled].steps .<= 1)
  end
end
