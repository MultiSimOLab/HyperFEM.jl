using Gridap
using Gridap.FESpaces, Gridap.Geometry, Gridap.CellData
using HyperFEM
using HyperFEM.ComputationalModels: PointLocation, Transfer, transfer!, transfer_transpose!, is_gather, _free_dof_targets
using LinearAlgebra
using SparseArrays
using Test

# A vector type that is not an Array: the application loops must be generic in the array type (what a GPU or a
# distributed vector needs); only getindex/setindex!/size are provided.
struct WrappedVector{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(a::WrappedVector) = size(a.data)
Base.getindex(a::WrappedVector, i::Int) = a.data[i]
Base.setindex!(a::WrappedVector, v, i::Int) = (a.data[i] = v)

fes(model, T, order; kwargs...) = TestFESpace(model, ReferenceFE(lagrangian, T, order); conformity=:H1, kwargs...)
apply(T, uh) = transfer!(zeros(size(T, 1)), T, uh)

@testset "Transfer" begin

  lin(x) = 2.0 + 3.0 * x[1] - 1.5 * x[2]
  quad(x) = 1.0 + x[1] * x[2] - 2.0 * x[1]^2 + 0.5 * x[2]
  vlin(x) = VectorValue(1.0 + x[1] - 2.0 * x[2], -0.5 + 0.25 * x[1] + 4.0 * x[2])
  wild(x) = VectorValue(sin(3.0 * x[1]) * exp(x[2]), cos(5.0 * x[2] + x[1]))

  @testset "Coincident nodes: a gather, exact" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (8, 6))
    for order in (1, 2), (T, f) in ((Float64, x -> wild(x)[1]), (VectorValue{2,Float64}, wild))
      Vs = fes(model, T, order; dirichlet_tags=["tag_5"])          # source: Dirichlet dofs on the bottom
      Vd = fes(model, T, order; dirichlet_tags=["tag_7", "tag_8"])  # destination: another Dirichlet boundary
      uh = interpolate_everywhere(f, Vs)
      vh = interpolate_everywhere(f, Vd)
      Tr = Transfer(Vs, Vd)
      @test is_gather(Tr)
      @test Tr.location.nmatched == num_free_dofs(Vd)
      @test apply(Tr, uh) == get_free_dof_values(vh)              # a copy of dof values: no rounding
      @test sizeof(Tr.location.col) == 4 * num_free_dofs(Vd)       # 4 bytes per destination dof
      @test isempty(Tr.location.rowptr) && isempty(Tr.location.val)
      # the mask: a destination with one component constrained only
      if T <: VectorValue
        Vm = fes(model, T, order; dirichlet_tags=["tag_6"], dirichlet_masks=[(true, false)])
        @test apply(Transfer(Vs, Vm), uh) == get_free_dof_values(interpolate_everywhere(f, Vm))
      end
    end
  end

  @testset "Coordinates perturbed by one ulp still match" begin
    model = CartesianDiscreteModel((0, 1, 0, 1), (7, 5))
    Vs = fes(model, Float64, 1)
    uh = interpolate_everywhere(x -> wild(x)[1], Vs)
    points, node, comp, ncomp = _free_dof_targets(Vs)
    for bump in (nextfloat, prevfloat)
      moved = [VectorValue(bump(x[1]), bump(x[2])) for x in points]
      L = PointLocation(Vs, moved, node, comp; ncomp)
      @test is_gather(L)
      @test transfer!(zeros(length(node)), L, uh) == get_free_dof_values(uh)
    end
  end

  @testset "Non-matching meshes: interpolation" begin
    cases = (("quads", m -> m), ("simplices", simplexify))
    for (name, change) in cases
      ms = change(CartesianDiscreteModel((0, 1, 0, 1), (10, 13)))   # no shared nodes but the corners
      md = change(CartesianDiscreteModel((0, 1, 0, 1), (9, 7)))
      for (os, od, f) in ((1, 1, lin), (2, 1, quad), (1, 2, lin), (2, 2, quad))
        name == "simplices" && f === quad && os == 2 || name == "quads" || f === lin || continue
        Vs = fes(ms, Float64, os); Vd = fes(md, Float64, od)
        uh = interpolate_everywhere(f, Vs)
        Tr = Transfer(Vs, Vd)
        @test !is_gather(Tr)
        exact = get_free_dof_values(interpolate_everywhere(f, Vd))
        @test isapprox(apply(Tr, uh), exact; rtol=1e-12, atol=1e-12)
        A = sparse(Tr)
        @test isapprox(A * ones(size(A, 2)), ones(size(A, 1)); rtol=1e-12)          # partition of unity
        @test A * vcat(get_free_dof_values(uh), uh.dirichlet_values) ≈ apply(Tr, uh)
        # against Gridap's interpolation through an Interpolable
        ref = get_free_dof_values(interpolate_everywhere(Interpolable(uh), Vd))
        @test isapprox(apply(Tr, uh), ref; rtol=1e-12, atol=1e-12)
      end
      # vector valued, source with Dirichlet dofs: the Dirichlet values of the function are read too
      Vs = fes(ms, VectorValue{2,Float64}, 1; dirichlet_tags=["tag_5", "tag_7"]); Vd = fes(md, VectorValue{2,Float64}, 1)
      uh = interpolate_everywhere(vlin, Vs)
      @test isapprox(apply(Transfer(Vs, Vd), uh), get_free_dof_values(interpolate_everywhere(vlin, Vd)); rtol=1e-12, atol=1e-12)
      # match = :search on coincident meshes: same values up to rounding, not a gather
      Tsearch = Transfer(Vs, Vs; match=:search)
      @test !is_gather(Tsearch)
      @test isapprox(apply(Tsearch, uh), get_free_dof_values(uh); rtol=1e-12, atol=1e-12)
    end
  end

  @testset "Non-affine cells" begin
    warp(x) = VectorValue(x[1] + 0.08 * x[1] * x[2], x[2] + 0.05 * x[1]^2)
    ms = CartesianDiscreteModel((0, 1, 0, 1), (9, 8); map=warp)
    Vs = fes(ms, Float64, 1)
    uh = interpolate_everywhere(lin, Vs)                 # isoparametric Q1 reproduces linear fields
    pts = [warp(VectorValue(0.137 + 0.61 * s, 0.211 + 0.53 * s^2)) for s in range(0, 1; length=17)]
    L = PointLocation(Vs, pts, collect(1:length(pts)), ones(Int, length(pts)))
    y = transfer!(zeros(length(pts)), L, uh)
    println("  non-affine cells: max error ", maximum(abs, y .- lin.(pts)))
    @test isapprox(y, lin.(pts); rtol=1e-12)
  end

  @testset "Source on a boundary triangulation" begin
    # 2D: the field lives on the top edge only
    model = CartesianDiscreteModel((0, 1, 0, 1), (6, 4))
    Γ = BoundaryTriangulation(model, tags=["tag_6"])
    VΓ = FESpace(Γ, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1))
    uΓ = interpolate_everywhere(vlin, VΓ)
    pts = [VectorValue(s, 1.0) for s in (0.0, 0.07, 1 / 6, 0.31, 0.5, 0.83, 1.0)]     # nodes and points inside the edges
    node = repeat(1:length(pts), inner=2); comp = repeat(1:2, outer=length(pts))
    L = PointLocation(VΓ, pts, node, comp)
    @test !is_gather(L) && L.nmatched == 2 * 4            # 0, 1/6, 0.5 and 1 are nodes of the edge
    y = transfer!(zeros(length(node)), L, uΓ)
    @test isapprox(y, [vlin(pts[n])[c] for (n, c) in zip(node, comp)]; rtol=1e-12, atol=1e-12)
    # a point off the edge is not projected on it
    @test_throws ErrorException PointLocation(VΓ, [VectorValue(0.31, 0.97)], [1, 1], [1, 2])
    # 3D: a face of the cube
    model3 = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (3, 4, 2))
    Γ3 = BoundaryTriangulation(model3, tags=["tag_22"])
    V3 = FESpace(Γ3, ReferenceFE(lagrangian, Float64, 1))
    g3(x) = 1.0 + 2.0 * x[1] - x[2] + 0.5 * x[3]
    u3 = interpolate_everywhere(g3, V3)
    xΓ = get_cell_coordinates(Γ3)[1]
    xc = sum(xΓ) / length(xΓ)                               # centre of a face of the boundary
    L3 = PointLocation(V3, [xc], [1], [1])
    @test isapprox(transfer!(zeros(1), L3, u3)[1], g3(xc); rtol=1e-12)
  end

  @testset "Errors" begin
    ms = CartesianDiscreteModel((0, 1, 0, 1), (4, 4))
    Vs = fes(ms, Float64, 1); Vv = fes(ms, VectorValue{2,Float64}, 1)
    @test_throws ErrorException Transfer(Vs, Vv)                                           # numbers of components
    @test_throws ErrorException PointLocation(Vs, [VectorValue(1.5, 0.5)], [1], [1])      # outside the source mesh
    @test_throws ErrorException PointLocation(Vs, [VectorValue(0.5, 0.5)], [1], [1]; match=:project)
    Vdg = TestFESpace(ms, ReferenceFE(lagrangian, Float64, 1); conformity=:L2)
    @test_throws ErrorException PointLocation(Vdg, [VectorValue(0.25, 0.25)], [1], [1])   # several dofs at one node
  end

  @testset "Application: no allocations, inferred, generic arrays" begin
    ms = CartesianDiscreteModel((0, 1, 0, 1), (10, 13)); md = CartesianDiscreteModel((0, 1, 0, 1), (9, 7))
    Vs = fes(ms, VectorValue{2,Float64}, 1; dirichlet_tags=["tag_5"])
    uh = interpolate_everywhere(vlin, Vs)
    xf, xd = get_free_dof_values(uh), uh.dirichlet_values
    for Vd in (fes(md, VectorValue{2,Float64}, 1), fes(ms, VectorValue{2,Float64}, 1))    # CSR and gather
      Tr = Transfer(Vs, Vd)
      L = Tr.location
      @test isconcretetype(typeof(L)) && L.col isa Vector{Int32} && L.val isa Vector{Float64}
      y = zeros(size(Tr, 1)); f = zeros(length(xf)); g = rand(size(Tr, 1))
      transfer!(y, L, xf, xd); transfer_transpose!(f, L, g)
      @test (@allocated transfer!(y, L, xf, xd)) == 0
      @test (@allocated transfer_transpose!(f, L, g)) == 0
      @test (@inferred transfer!(y, L, xf, xd)) === y
      @test (@inferred transfer_transpose!(f, L, g)) === f
      # other array and number types
      yw = WrappedVector(zeros(length(y)))
      transfer!(yw, L, WrappedVector(xf), WrappedVector(xd))
      @test yw.data == y
      y32 = zeros(Float32, length(y))
      transfer!(y32, L, Float32.(xf), Float32.(xd))
      @test isapprox(y32, y; rtol=1e-5, atol=1e-5)
      Lw = PointLocation(WrappedVector(L.rowptr), WrappedVector(L.col), WrappedVector(L.val), L.ntargets, L.nfree, L.ndirichlet, L.nmatched)
      @test transfer!(zeros(length(y)), Lw, xf, xd) == y
      # transpose: (Lᵀ g) on the free dofs, and the same as the sparse matrix
      A = sparse(L)
      @test isapprox(f, (A'*g)[1:length(xf)]; rtol=1e-13, atol=1e-13)
      # triplets with global numbering rebuild the same map (what a distributed assembly consumes)
      I, J, W = findnz(A)
      @test sparse(I, J, W, size(A)...) * vcat(xf, xd) ≈ y
    end
  end

  @testset "Loads: the transpose keeps the resultant" begin
    ms = CartesianDiscreteModel((0, 1, 0, 1), (10, 13)); md = CartesianDiscreteModel((0, 1, 0, 1), (9, 7))
    Vs = fes(ms, Float64, 1); Vd = fes(md, Float64, 1)
    Tr = Transfer(Vs, Vd)
    g = rand(num_free_dofs(Vd))
    f = transfer_transpose!(zeros(num_free_dofs(Vs)), Tr, g)
    @test isapprox(sum(f), sum(g); rtol=1e-12)
    # the transfer of the nodal loads as if they were a field does not
    Tback = Transfer(Vd, Vs)
    @test !isapprox(sum(transfer!(zeros(num_free_dofs(Vs)), Tback, g, Float64[])), sum(g); rtol=1e-3)
  end

end
