# Dirichlet values of a space from its conditions (DirichletApplicator): the case of the staggered
# solid–vacuum problems, a coupling with a field living only on the faces of the interface, on the
# mesh of the mesh-movement simulation (test_BC2.msh: air domain, 2D, order 1).
using Gridap.FESpaces

function interface_coupling_setup()
  geomodel = GmshDiscreteModel(projdir("test/models/test_BC2.msh"))
  Ωair = Interior(geomodel, tags=["Air"])
  Γ = BoundaryTriangulation(Ωair, tags="Interface")
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
  L = 0.1
  mapping(x, Λ) = VectorValue(0.0, Λ * 0.98 * L * sin(2 * π * (x[1] + L / 2) / L))
  uh⁺ = interpolate_everywhere(x -> mapping(x, 1.0), FESpace(Γ, reffeu))
  uh⁻ = interpolate_everywhere(x -> mapping(x, 0.0), FESpace(Γ, reffeu))
  bc = BoundaryConditions(
    BoundaryCondition("uair_fixed", [0.0, 0.0], Λ -> Λ),
    BoundaryCondition("Interface", Coupling(τ -> uh⁻ + (uh⁺ - uh⁻) * τ)))
  V = TestFESpace(Ωair, reffeu, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  U, bc, dirichlet_applicator(U, bc)
end

SUITE["BoundaryConditions"] = BenchmarkGroup()
SUITE["BoundaryConditions"]["TrialFESpace! interface coupling"] = @benchmarkable TrialFESpace!(U, app, 0.7; stage=0.4) setup = ((U, bc, app) = interface_coupling_setup())
SUITE["BoundaryConditions"]["TrialFESpace! increment"] = @benchmarkable TrialFESpace!(U, app, 0.7, 0.1; stage=(0.4, 0.1)) setup = ((U, bc, app) = interface_coupling_setup())
SUITE["BoundaryConditions"]["dirichlet_applicator"] = @benchmarkable dirichlet_applicator(U, bc) setup = ((U, bc, app) = interface_coupling_setup())
