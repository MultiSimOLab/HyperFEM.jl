# Backend types: the `backend` keywords accept a symbol or a backend, unknown backends fail clearly.
using Gridap, Gridap.FESpaces
using HyperFEM
using HyperFEM.Backends
using Test

struct NoSuchBackend <: AbstractBackend end

@testset "Backends" begin
  @test as_backend(:lazy) === LazyBackend()
  @test as_backend(:compiled) === CompiledBackend()
  @test as_backend(LazyBackend()) === LazyBackend()
  @test_throws ErrorException as_backend(:fast)
  @test_throws ErrorException as_backend("lazy")
  @test Backends.backend_name(LazyBackend()) === :lazy
  @test Backends.backend_name(CompiledBackend()) === :compiled
  @test Backends.backend_name(NoSuchBackend()) === :NoSuchBackend

  p = drivers_test_problem()
  m1 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); verbose=0))
  x1, _ = solve!(m1; stepping=(nsteps=2, maxbisec=0))
  p = drivers_test_problem()
  m2 = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); verbose=0), backend=LazyBackend())
  x2, _ = solve!(m2; stepping=(nsteps=2, maxbisec=0))
  @test x1 == x2
  # a backend without implementation fails at construction with a message naming it
  err = try
    StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; backend=NoSuchBackend()); nothing
  catch e
    e
  end
  @test err isa ErrorException && occursin("NoSuchBackend", err.msg)
  @test_throws ErrorException StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; backend=:fast)
  dΩ = Measure(Triangulation(CartesianDiscreteModel((0, 1), (2,))), 2)
  @test HyperFEM.PhysicalModels.new_state(1.0, dΩ, :lazy) isa Gridap.CellData.CellState
  @test_throws ErrorException HyperFEM.PhysicalModels.new_state(1.0, dΩ, NoSuchBackend())
  @test_throws ErrorException HyperFEM.PhysicalModels.state_updater(identity, NoSuchBackend())
end

@testset "Typed caches" begin
  p = drivers_test_problem()
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); verbose=0))
  c = m.caches
  @test c isa NonlinearStepCache
  @test isconcretetype(typeof(c)) && c.backend === LazyBackend() && c.projection === nothing
  @test c.x === get_free_dof_values(get_state(m)) && get_assemblers(m) === c.assem
  op0 = c.op
  x, _ = solve!(m; stepping=(nsteps=2, maxbisec=0), ProjectDirichlet=true)
  @test x === c.x && typeof(c.op) === typeof(op0)    # one operator per step, of the same type
  @test fieldtype(typeof(c), :op) === typeof(op0)
  @test c.projection === nothing                                       # the lazy projection keeps no operator
  # DynamicNonlinearModel (deprecated, removed in the next release): the constructor only
  p = drivers_test_problem()
  Uold = TrialFESpace(p.V, p.bc, 0.0)
  vel = HyperFEM.ComputationalModels.SingleFieldTC(x -> VectorValue(0.0, 0.0), p.V)
  resd(t, uh⁻, vh) = (u, v) -> p.res(t)(u, v)
  jacd(t, uh⁻, vh) = (u, du, v) -> p.jac(t)(u, du, v)
  mkd() = DynamicNonlinearModel(resd, jacd, p.U, Uold, p.V, p.bc, vel; nls=NewtonRaphson(LUSolver(); verbose=0))
  md = Base.JLOptions().depwarn == 0 ? mkd() : (@test_logs (:warn, r"deprecated") match_mode = :any mkd())
  @test md.caches isa NonlinearStepCache && isconcretetype(typeof(md.caches)) && md.caches.backend === LazyBackend()
  @test md.state⁻ isa FEFunction && get_free_dof_values(md.state⁻) === md.caches.x⁻ && get_assemblers(md) === md.caches.assem
end

@testset "Custom assemblers" begin
  p = drivers_test_problem()
  assem = SparseMatrixAssembler(p.U, p.V)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); verbose=0), assem_U=assem)
  @test get_assemblers(m) === assem
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; nls=NewtonRaphson(LUSolver(); verbose=0))
  @test get_assemblers(m) isa SparseMatrixAssembler
  err = try
    StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.bc; backend=NoSuchBackend(), assem_U=assem); nothing
  catch e
    e
  end
  @test err isa ErrorException && occursin("takes no assembler", err.msg) && occursin("NoSuchBackend", err.msg)
  @test HyperFEM.ComputationalModels.assembler(CompiledBackend(), nothing, p.U, p.V) === nothing
  @test_throws ErrorException HyperFEM.ComputationalModels.assembler(CompiledBackend(), assem, p.U, p.V)
end
