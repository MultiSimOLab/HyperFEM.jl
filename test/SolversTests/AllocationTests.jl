# The Newton loop and the line searches do not allocate beyond the assembly and the linear solve:
# a diagonal model problem R(x) = d .* x + x.^3 - f with an allocation-free operator and solver.
using Gridap, Gridap.Algebra
using HyperFEM
using HyperFEM.Solvers: search!, LineSearch
using LinearAlgebra
using Test

struct DiagonalCubicOperator <: NonlinearOperator
  d::Vector{Float64}
  f::Vector{Float64}
end

function Gridap.Algebra.residual!(b::AbstractVector, op::DiagonalCubicOperator, x::AbstractVector)
  @inbounds for i in eachindex(b)
    b[i] = op.d[i] * x[i] + x[i]^3 - op.f[i]
  end
  b
end
function Gridap.Algebra.jacobian!(A::Diagonal, op::DiagonalCubicOperator, x::AbstractVector)
  @inbounds for i in eachindex(x)
    A.diag[i] = op.d[i] + 3x[i]^2
  end
  A
end
Gridap.Algebra.allocate_residual(op::DiagonalCubicOperator, x::AbstractVector) = similar(x)
Gridap.Algebra.allocate_jacobian(op::DiagonalCubicOperator, x::AbstractVector) = Diagonal(similar(x))
Gridap.Algebra.zero_initial_guess(op::DiagonalCubicOperator) = zeros(length(op.d))

# A direct solver for diagonal tangents (Gridap's LUSolver allocates on every numerical setup)
struct DiagonalSolver <: LinearSolver end
struct DiagonalSS <: SymbolicSetup end
mutable struct DiagonalNS <: NumericalSetup
  A::Diagonal{Float64,Vector{Float64}}
end
Gridap.Algebra.symbolic_setup(::DiagonalSolver, A::AbstractMatrix) = DiagonalSS()
Gridap.Algebra.numerical_setup(::DiagonalSS, A::Diagonal) = DiagonalNS(A)
Gridap.Algebra.numerical_setup!(ns::DiagonalNS, A::Diagonal) = (ns.A = A; ns)
Gridap.Algebra.solve!(x::AbstractVector, ns::DiagonalNS, b::AbstractVector) = (x .= b ./ ns.A.diag; x)

n = 1000
d = 1.0 .+ rand(n)
f = rand(n)
op = DiagonalCubicOperator(d, f)

function solve_twice(nls, op)
  x = zero_initial_guess(op)
  cache = solve!(x, nls, op, nothing)
  fill!(x, 0.0)
  @allocated solve!(x, nls, op, cache)
end

@testset "Newton loop without allocations" begin
  for ls in (LineSearch(), Roman_LS(; maxiter=10))
    nls = NewtonRaphson(DiagonalSolver(); maxiter=30, atol=1e-13, rtol=1e-14, verbose=0)
    nls = NewtonRaphson(nls.ls, nls.tols, nls.log, ls)
    bytes = solve_twice(nls, op)
    @test bytes == 0
    @test converged(nls) && num_iterations(nls) > 2
    x = zero_initial_guess(op)
    solve!(x, nls, op, nothing)
    @test norm(d .* x .+ x .^ 3 .- f) < 1e-12
  end
  # curvature criterion active: still no allocations
  nls = NewtonRaphson(DiagonalSolver(); maxiter=30, atol=0.0, rtol=0.0, ctol=1e-20, verbose=0)
  @test solve_twice(nls, op) == 0
end

@testset "Line searches without allocations" begin
  x = zero_initial_guess(op)
  nls = NewtonRaphson(DiagonalSolver(); maxiter=30, atol=1e-13, rtol=1e-14)
  cache = solve!(x, nls, op, nothing)
  w = similar(x)
  b, dx = cache.b, cache.dx
  fill!(x, 0.1)
  residual!(b, op, x)
  rmul!(b, -1)
  dx .= b ./ (d .+ 3 .* x .^ 2)
  for ls in (LineSearch(), Roman_LS(; maxiter=10))
    search!(ls, w, op, x, dx, b)
    residual!(b, op, x); rmul!(b, -1)
    @test (@allocated search!(ls, w, op, x, dx, b)) == 0
    residual!(b, op, x); rmul!(b, -1)
  end
end
