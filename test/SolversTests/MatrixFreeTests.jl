# Matrix-free tangent inside the Newton (design/solvers.md, stage S3): `CompiledBackend(jacobian = :matrixfree)`
# hands the Newton a `MatrixFreeJacobian` of HyperFEMKernels, solved by its `MatrixFreeSolver` (Krylov.jl on the
# operator) with the slots `LaggedAssembled` (the tangent assembled every k iterations, a HyperFEM solver on it:
# `AMGSolver` here, `PETScSolver` in production) and `BlockJacobi`. The S1/S2 batteries are the reference: same Newton
# trajectory and solution as LU on the mechanics and EM problems, the forcing term through `set_tolerance!`, the
# Dirichlet projection through the operator, the verbose row with the Krylov iterations. Needs HyperFEMKernels.
include("krylov_common.jl")
import AlgebraicMultigrid                      # activates HyperFEMAMGExt (AMGSolver inside LaggedAssembled)
using HyperFEMKernels: MatrixFreeSolver, LaggedAssembled, BlockJacobi, MatrixFreeJacobian, sparsity_pattern

const MF_BACKEND = CompiledBackend(jacobian=:matrixfree)

function mf_capture_stdout(f)
  path, io = mktemp()
  try
    redirect_stdout(f, io)
  finally
    close(io)
  end
  read(path, String)
end

@testset "matrix-free Newton (S3)" begin
  @testset "CompiledBackend(jacobian)" begin
    @test CompiledBackend().jacobian === :assembled && as_backend(:compiled).jacobian === :assembled
    @test MF_BACKEND.jacobian === :matrixfree && HyperFEM.Backends.backend_name(MF_BACKEND) === :compiled
    @test_throws ErrorException CompiledBackend(jacobian=:dense)
  end

  @testset "the Newton receives the operator; an assembled-matrix solver is refused" begin
    p = s1_mech_problem()
    nls = NewtonRaphson(LUSolver(); maxiter=20, rtol=1e-6)
    @test_throws ErrorException StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend=MF_BACKEND)
    nls = NewtonRaphson(MatrixFreeSolver(; preconditioner=LaggedAssembled(LUSolver())); maxiter=20, rtol=1e-6)
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend=MF_BACKEND)
    @test m.caches.nls_cache.A isa MatrixFreeJacobian
    @test m.caches.op.pattern.A0 === nothing                 # no sparsity pattern before the first solve
    nls2 = NewtonRaphson(MatrixFreeSolver(); maxiter=20, rtol=1e-6)
    @test_throws ErrorException StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls=nls2, backend=:compiled)   # needs the operator
  end

  @testset "Newton trajectory of LU (nonlinear mechanics, Dirichlet projection through the operator)" begin
    p = s1_mech_problem()
    Nm = near_nullspace(p.U)
    ref = s1_solve_nl(LUSolver(), p; backend=:compiled)
    @test ref.flag && ref.its ≥ 2
    for (name, prec) in (("LaggedAssembled(LU)", LaggedAssembled(LUSolver())),
                         ("LaggedAssembled(AMG, every 2)", LaggedAssembled(AMGSolver(:cg; rtol=1e-12, nullspace=Nm); every=2)),
                         ("BlockJacobi", BlockJacobi()))
      @testset "$name" begin
        for krylov in (:gmres, :fgmres)
          ls = MatrixFreeSolver(; krylov, rtol=1e-8, maxiter=2000, restart=100, preconditioner=prec)
          r = s1_solve_nl(ls, p; backend=MF_BACKEND)
          @test r.flag
          @test r.its == ref.its
          @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
          @test all(>(0), r.lin)                             # Krylov iterations reported for every Newton iteration
          @test tolerance(r.ns) == 1e-8
        end
      end
    end
    # the lag: fewer assemblies than Newton iterations
    ls = MatrixFreeSolver(; rtol=1e-8, maxiter=2000, restart=100, preconditioner=LaggedAssembled(LUSolver(); every=3))
    r = s1_solve_nl(ls, p; backend=MF_BACKEND)
    @test r.flag && preconditioner_builds(r.ns) < r.niters
    # a Krylov solve that cannot converge fails the Newton like a line search
    bad = s1_solve_nl(MatrixFreeSolver(; rtol=1e-8, maxiter=1, restart=1), p; backend=MF_BACKEND)
    @test !bad.flag && bad.reason === :linear_failed
  end

  @testset "inexact Newton: the forcing term drives set_tolerance! of the operator solver" begin
    p = s1_mech_problem()
    make(rtol) = MatrixFreeSolver(; rtol, maxiter=2000, restart=100, preconditioner=LaggedAssembled(LUSolver(); every=3))
    ref = s1_solve_nl(make(1e-10), p; backend=MF_BACKEND)
    ew = s1_solve_nl(make(1e-10), p; backend=MF_BACKEND, forcing=EisenstatWalker())
    @test ref.flag && ew.flag
    @test ew.niters ≤ 1.5 * ref.niters + 1
    @test ew.nlinear < ref.nlinear
    @test norm(ew.x - ref.x) ≤ 1e-5 * norm(ref.x)
    @test all(η -> 1e-10 ≤ η ≤ 1e-2, ew.etas) && ew.etas[1] == 1e-2
    @test tolerance(ew.ns) == 1e-10
  end

  @testset "electro-mechanics: the saddle point through LaggedAssembled(AMGSolver blocks)" begin
    p = s1_em_problem()
    ref = s1_solve_nl(LUSolver(), p; backend=:compiled)
    @test ref.flag
    amg = AMGSolver(p.U; rtol=1e-12, maxiter=500)                        # S2 block preconditioner, assembled every 2 iterations
    ls = MatrixFreeSolver(; krylov=:fgmres, rtol=1e-8, maxiter=3000, restart=200, preconditioner=LaggedAssembled(amg; every=2))
    r = s1_solve_nl(ls, p; backend=MF_BACKEND)
    @test r.flag
    @test r.its == ref.its
    @test norm(r.x - ref.x) ≤ 1e-5 * norm(ref.x)
    @test all(>(0), r.lin)
  end

  @testset "verbose row names the operator solver" begin
    p = s1_mech_problem()
    ls = MatrixFreeSolver(; rtol=1e-8, maxiter=2000, restart=100, preconditioner=LaggedAssembled(LUSolver()))
    nls = NewtonRaphson(ls; maxiter=20, rtol=1e-6, verbose=2)
    m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend=MF_BACKEND)
    out = mf_capture_stdout(() -> solve!(m; stepping=(nsteps=1, maxbisec=0)))
    @test occursin("MF-GMRES", out)
  end
end
