using Gridap
using Gridap.FESpaces, Gridap.CellData
using HyperFEM
using HyperFEM.ComputationalModels: _TagTransfer, _TagCells, DirichletInterpolator, apply!, is_gather
using LinearAlgebra
using Test

# Fields prescribed by a Coupling: the nodes of the tag are located once in the mesh of the field; a finite-element
# function goes through its dof values (algebraic path), any other field is evaluated in the located cells.

@testset "Coupling: located once, algebraic and general paths" begin

  function square(n)
    model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (n, n))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "bottom", [1, 2, 5])
    add_tag_from_tags!(labels, "top", [3, 4, 6])
    model
  end
  f(x) = VectorValue(x[1]^2 + 0.3, x[1] * x[2] - 0.2)
  g(x) = VectorValue(sin(2.0 * x[1]), 0.5 - x[1]^2)
  zero2 = VectorValue(0.0, 0.0)
  coupled(c; kwargs...) = BoundaryConditions(BoundaryCondition("bottom", [0.0, 0.0]), BoundaryCondition("top", c; kwargs...))
  top(V) = get_dirichlet_dof_tag(V) .== 2
  blend(a, b, τ) = a .+ (b .- a) .* τ

  for order in (1, 2)
    model = square(5)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    Vsrc = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["bottom"])      # the source has Dirichlet dofs
    uh⁻ = interpolate_everywhere(g, TrialFESpace(Vsrc, g)); uh⁺ = interpolate_everywhere(f, TrialFESpace(Vsrc, f))
    V = TestFESpace(model, reffe, coupled(Coupling(τ -> uh⁺)))
    exact⁺ = get_dirichlet_dof_values(TrialFESpace(V, (zero2, f)))
    exact⁻ = get_dirichlet_dof_values(TrialFESpace(V, (zero2, g)))

    @testset "order $order, coincident nodes" begin
      # a finite-element function: a gather of its dof values, exact
      bc = coupled(Coupling(τ -> uh⁺))
      U = TrialFESpace(V, bc, 0.0)
      app = dirichlet_applicator(U, bc)
      TrialFESpace!(U, app, 1.0)
      @test get_dirichlet_dof_values(U) == exact⁺
      c = app.cache[2]
      @test c isa _TagTransfer && is_gather(c.location)
      @test sizeof(c.location.col) == 4 * count(top(V))                    # 4 bytes per dof of the tag
      # the same inside an Interpolable
      bcI = coupled(Coupling(τ -> Interpolable(uh⁺)))
      @test get_dirichlet_dof_values(TrialFESpace(V, bcI, 1.0)) == exact⁺
      # field updated in place (staggered loop): no new location, new values
      get_free_dof_values(uh⁺) .*= 2.0; uh⁺.dirichlet_values .*= 2.0
      TrialFESpace!(U, app, 1.0)
      @test app.cache[2] === c
      @test get_dirichlet_dof_values(U) == 2.0 .* exact⁺
      get_free_dof_values(uh⁺) ./= 2.0; uh⁺.dirichlet_values ./= 2.0
      @test @allocated(TrialFESpace!(U, app, 1.0)) < 1_000

      # StageBlend: algebraic, the blend of the two dof vectors, exact
      bcB = coupled(Coupling(StageBlend(uh⁻, uh⁺)))
      UB = TrialFESpace(V, bcB, 0.0)
      appB = dirichlet_applicator(UB, bcB)
      for τ in (0.0, 0.3, 1.0)
        TrialFESpace!(UB, appB, 0.7; stage=τ)                              # the load is not the parameter of a coupling
        @test get_dirichlet_dof_values(UB)[top(V)] == blend(exact⁻, exact⁺, τ)[top(V)]
        @test all(==(0.0), get_dirichlet_dof_values(UB)[.!top(V)])
      end
      cB = appB.cache[2]
      @test cB isa _TagTransfer && is_gather(cB.location)
      println("  order $order: bytes per apply!, FE function ", @allocated(TrialFESpace!(U, app, 1.0)), ", StageBlend ", @allocated(TrialFESpace!(UB, appB, 0.7; stage=0.3)))
      @test @allocated(TrialFESpace!(UB, appB, 0.7; stage=0.3)) < 2_000
      # increment between two stages
      TrialFESpace!(UB, appB, 0.7, 0.1; stage=(0.6, 0.25))
      @test isapprox(get_dirichlet_dof_values(UB), blend(exact⁻, exact⁺, 0.6) .- blend(exact⁻, exact⁺, 0.35); atol=1e-14)
      @test appB.cache[2] === cB

      # the closure: a new CellField at every call, evaluated in the located cells (general path), located once
      bcC = coupled(Coupling(τ -> uh⁻ + (uh⁺ - uh⁻) * τ))
      UC = TrialFESpace(V, bcC, 0.0)
      appC = dirichlet_applicator(UC, bcC)
      TrialFESpace!(UC, appC, 0.7; stage=0.3)
      cC = appC.cache[2]
      @test cC isa _TagCells
      @test isapprox(get_dirichlet_dof_values(UC), blend(exact⁻, exact⁺, 0.3); atol=1e-13)
      TrialFESpace!(UC, appC, 0.7; stage=0.8)
      @test appC.cache[2] === cC                                           # no new point search
      @test isapprox(get_dirichlet_dof_values(UC), blend(exact⁻, exact⁺, 0.8); atol=1e-13)
      TrialFESpace!(UC, appC, 0.7, 0.1; stage=(0.6, 0.25))
      @test isapprox(get_dirichlet_dof_values(UC), blend(exact⁻, exact⁺, 0.6) .- blend(exact⁻, exact⁺, 0.35); atol=1e-13)

      # every other way to the values of the StageBlend gives the same
      @test isapprox(get_dirichlet_dof_values(TrialFESpace(V, bcB, 0.7; stage=0.3)), blend(exact⁻, exact⁺, 0.3); atol=1e-14)
      UI = TrialFESpace(V, bcB, 0.0)
      apply!(UI, DirichletInterpolator(bcB), 0.7, 0.3)                     # Gridap's interpolation of the same field
      @test isapprox(get_dirichlet_dof_values(UI), blend(exact⁻, exact⁺, 0.3); atol=1e-12)
      @test get_space_functions(bcB, 0.7, 0.3)[2] isa CellField
      d = get_time_derivative(bcB, 0.7, 0.3)[2]                            # exact: uh⁺ - uh⁻
      x = Point(0.37, 0.61)
      @test d isa CellField && isapprox(d(x), uh⁺(x) - uh⁻(x); atol=1e-13)

      # a mask: only one component constrained on the tag
      for mask in ((true, false), (false, true))
        bcM = coupled(Coupling(StageBlend(uh⁻, uh⁺)); mask=mask)
        VM = TestFESpace(model, reffe, bcM)
        ref = get_dirichlet_dof_values(TrialFESpace!(TrialFESpace(VM, bcM, 0.0), coupled(Coupling(τ -> uh⁻ + (uh⁺ - uh⁻) * τ); mask=mask), 0.0; stage=0.4))
        got = get_dirichlet_dof_values(TrialFESpace(VM, bcM, 0.0; stage=0.4))
        @test isapprox(got, ref; atol=1e-13) && norm(got) > 0
      end
    end

    @testset "order $order, field on the faces of the tag only" begin
      Γ = BoundaryTriangulation(model, tags="top")
      uΓ = interpolate_everywhere(f, FESpace(Γ, reffe))
      bcΓ = coupled(Coupling(τ -> uΓ))
      UΓ = TrialFESpace(V, bcΓ, 0.0)
      appΓ = dirichlet_applicator(UΓ, bcΓ)
      TrialFESpace!(UΓ, appΓ, 1.0)
      @test get_dirichlet_dof_values(UΓ) == exact⁺                         # coincident nodes: a gather from the interface space
      @test appΓ.cache[2] isa _TagTransfer && is_gather(appΓ.cache[2].location)
      bcΓ2 = coupled(Coupling(τ -> τ * uΓ))                                # general path on the interface
      UΓ2 = TrialFESpace(V, bcΓ2, 0.0)
      appΓ2 = dirichlet_applicator(UΓ2, bcΓ2)
      TrialFESpace!(UΓ2, appΓ2, 0.0; stage=0.5)
      @test appΓ2.cache[2] isa _TagCells
      @test isapprox(get_dirichlet_dof_values(UΓ2), 0.5 .* exact⁺; atol=1e-13)
    end
  end

  @testset "Non-matching source mesh" begin
    model = square(5)
    reffe2 = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    uhA = interpolate(f, FESpace(CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (3, 3)), reffe2))   # f is quadratic
    for c in (Coupling(τ -> uhA), Coupling(τ -> uhA; match=:search), Coupling(τ -> Interpolable(uhA)), Coupling(τ -> 1.0 * uhA))
      bc = coupled(c)
      V = TestFESpace(model, reffe2, bc)
      exact = get_dirichlet_dof_values(TrialFESpace(V, (zero2, f)))
      U = TrialFESpace(V, bc, 0.0)
      app = dirichlet_applicator(U, bc)
      TrialFESpace!(U, app, 1.0)
      @test isapprox(get_dirichlet_dof_values(U), exact; rtol=1e-12, atol=1e-12)
      loc = app.cache[2]
      TrialFESpace!(U, app, 1.0)
      @test app.cache[2] === loc
    end
    @test dirichlet_applicator(TrialFESpace(TestFESpace(model, reffe2, coupled(Coupling(τ -> uhA))), coupled(Coupling(τ -> uhA)), 1.0),
                               coupled(Coupling(τ -> uhA))) isa DirichletApplicator
    # a field in the physical domain
    Ω = Triangulation(model)
    bcP = coupled(Coupling(τ -> CellField(x -> τ * f(x), Ω)))
    VP = TestFESpace(model, reffe2, bcP)
    @test isapprox(get_dirichlet_dof_values(TrialFESpace(VP, bcP, 0.0; stage=0.5)),
                   0.5 .* get_dirichlet_dof_values(TrialFESpace(VP, (zero2, f))); rtol=1e-12, atol=1e-12)
    # a source that does not reach the tag is an error, not an extrapolation
    uhalf = interpolate(f, FESpace(CartesianDiscreteModel((0.0, 1.0, 0.0, 0.5), (3, 3)), reffe2))
    bcH = coupled(Coupling(τ -> uhalf))
    @test_throws ErrorException TrialFESpace(TestFESpace(model, reffe2, bcH), bcH, 1.0)
  end

end


@testset "StaggeredModel with StageBlend" begin

  # The two staggered problems of BoundaryConditionsTests, the second coupled to the first on its top edge, with the
  # coupling written as a closure (general path) and as a StageBlend (algebraic path): same solution.
  function staggered(couple; project)
    model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (4, 4))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "bottom", [1, 2, 5]); add_tag_from_tags!(labels, "top", [3, 4, 6]); add_tag_from_tags!(labels, "left", [7])
    dΩ = Measure(Triangulation(model), 2)
    res(Λ) = (u, v) -> ∫(∇(v) ⋅ ∇(u))dΩ
    jac(Λ) = (u, du, v) -> ∫(∇(v) ⋅ ∇(du))dΩ
    nls = NewtonRaphson(LUSolver(); maxiter=5, rtol=1e-10, verbose=false)
    reffe = ReferenceFE(lagrangian, Float64, 1)
    bc1 = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("left", x -> x[2], Λ -> Λ))
    neumann1 = NeumannConditions(model, BoundaryConditions(BoundaryCondition("top", 1.0, Λ -> Λ)), 2)
    V1 = TestFESpace(model, reffe, bc1)
    U1 = TrialFESpace(V1, bc1, 0.0)
    u1⁺ = FEFunction(U1, zero_free_values(U1))
    u1⁻ = FEFunction(TrialFESpace(V1, bc1, 0.0), zero_free_values(U1))
    m1 = StaticNonlinearModel(res, jac, U1, V1, bc1; neumann=neumann1, nls=nls, xh=u1⁺)
    bc2 = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("top", couple(u1⁻, u1⁺)))
    V2 = TestFESpace(model, reffe, bc2)
    U2 = TrialFESpace(V2, bc2, 0.0)
    u2⁺ = FEFunction(U2, zero_free_values(U2))
    u2⁻ = FEFunction(TrialFESpace(V2, bc2, 0.0), zero_free_values(U2))
    m2 = StaticNonlinearModel(res, jac, U2, V2, bc2; nls=nls, xh=u2⁺)
    sm = StaggeredModel((m1, m2), (u1⁺, u2⁺), (u1⁻, u2⁻))
    a = project ? Dict(:stepping => (nsteps=2, maxbisec=1), :ProjectDirichlet => true) : Dict(:stepping => (nsteps=2, maxbisec=1))
    solve!(sm; stepping=StaggeredStepping(nsteps=4), kargsolve=(a, a), verbose=false)
    (u1=copy(get_free_dof_values(u1⁺)), u2=copy(get_free_dof_values(u2⁺)), d2=copy(get_dirichlet_dof_values(U2)),
     paths=(typeof(m2.applicators[1].cache[2]),))
  end
  closure(u⁻, u⁺) = Coupling(τ -> u⁻ + (u⁺ - u⁻) * τ)
  blended(u⁻, u⁺) = Coupling(StageBlend(u⁻, u⁺))
  for project in (false, true)
    a = staggered(closure; project); b = staggered(blended; project)
    @test a.paths[1] <: _TagCells && b.paths[1] <: _TagTransfer
    @test isapprox(a.u1, b.u1; atol=1e-13) && isapprox(a.u2, b.u2; atol=1e-13) && isapprox(a.d2, b.d2; atol=1e-13)
    @test maximum(b.d2) ≈ 1.0
  end

end
