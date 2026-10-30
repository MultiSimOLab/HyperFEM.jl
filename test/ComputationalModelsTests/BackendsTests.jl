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
