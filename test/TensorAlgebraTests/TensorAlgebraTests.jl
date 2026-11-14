using Gridap.TensorValues
using Gridap.Arrays
using HyperFEM.TensorAlgebra
using Test
import LinearAlgebra
import Random


digits1(D) = [Float64(i) for i in 1:D]
digits2(D) = [Float64(10i + j) for i in 1:D, j in 1:D]
digits3(D) = [Float64(100i + 10j + k) for i in 1:D, j in 1:D, k in 1:D]
digits4(D) = [Float64(1000i + 100j + 10k + l) for i in 1:D, j in 1:D, k in 1:D, l in 1:D]

third(a::Array{Float64,3}) = ThirdOrderTensorValue{size(a)...}(a...)
fourth(a::Array{Float64,4}) = FourthOrderTensorValue(Tuple(a))


@testset "transpose" begin
  for D in (2, 3)
    # third-order tensors: cyclic index shift Wᵀ[i,j,k] = W[k,i,j]
    w = digits3(D)
    W = third(w)
    @test W' == third([w[k, i, j] for i in 1:D, j in 1:D, k in 1:D])
    @test W''' == W
    # fourth-order tensors: major transpose Hᵀ[i,j,k,l] = H[k,l,i,j]
    h = digits4(D)
    H = fourth(h)
    @test H' == fourth([h[k, l, i, j] for i in 1:D, j in 1:D, k in 1:D, l in 1:D])
    @test H'' == H
    @test Tuple(H') == Tuple(TensorValue(Tuple(H))')  # the transpose of the D²×D² matrix
  end
end


@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; threshold=0.01) == 0.014870878346353422
  @test ForwardDiff.derivative(logreg, J) ≈ ∂log∂J(J)
  @test ForwardDiff.derivative(∂log∂J, J) ≈ ∂∂log∂JJ(J)
end


@testset "outer" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  B = TensorValue(5.0, 6.0, 7.0, 8.0)
  u = VectorValue(1.0, 2.0)
  v = VectorValue(3.0, 4.0)
  @test u ⊗ v == TensorValue(3.0, 6.0, 4.0, 8.0)
  @test Tuple(A ⊗ B)     == (5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0)
  @test Tuple(A ⊗ᵢₖʲˡ B) == (5.0, 10.0, 6.0, 12.0, 15.0, 20.0, 18.0, 24.0, 7.0, 14.0, 8.0, 16.0, 21.0, 28.0, 24.0, 32.0)
  @test Tuple(A ⊗ᵢₗʲᵏ B) == (5.0, 10.0, 6.0, 12.0, 7.0, 14.0, 8.0, 16.0, 15.0, 20.0, 18.0, 24.0, 21.0, 28.0, 24.0, 32.0)
  @test Tuple(u ⊗ A)     == (1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0)
  @test Tuple(A ⊗ u)     == (1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0)
  @test Tuple(A ⊗ᵢₖʲ u)  == (1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0)
end


@testset "cross" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  D = ThirdOrderTensorValue(Tuple([4.6 2.1 1.7 3.2 6.5 1.4 9.2 8.0 9.0;
                          1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
                          5.3 2.0 3.1 1.9 5.4 9.8 0.4 8.8 3.1] * 1e-3))
  @test sqrt(×ᵢ⁴(A) ⊙ ×ᵢ⁴(A)) == 0.033763886032268264
  @test norm(A × B) == 6.246230863488799e-5
  @test sqrt((C × B) ⊙ (C × B)) == 2.4491455542698976e-6
  @test sqrt((B × C) ⊙ (B × C)) == 1.104276381618298e-6
  @test sqrt((D × A) ⊙ (D × A)) == 0.00012378691368638284
  @test norm(get_array(A) × get_array(B))== 6.246230863488799e-5
end


@testset "inner" begin
  H = fourth(reshape(collect(1.0:16.0), 2, 2, 2, 2))
  G = third(reshape(collect(1.0:8.0), 2, 2, 2))
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  V = VectorValue(5.0, 6.0)
  @test inner(H,H) == 1496.0
  @test inner(G,G) == 204.0
  @test inner(A,A) == 30.0
  @test inner(V,V) == 61.0
  @test inner(H,A) == TensorValue(90.0, 100.0, 110.0, 120.0)
  @test inner(G,A) == VectorValue(50.0, 60.0)
  @test inner(G,V) == TensorValue(35.0, 46.0, 57.0, 68.0)
  @test H ⊙ A     == TensorValue(90.0, 100.0, 110.0, 120.0)
  @test G ⊙ A     == VectorValue(50.0, 60.0)
  @test G ⊙ V     == TensorValue(35.0, 46.0, 57.0, 68.0)
end


@testset "sum" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
  B = TensorValue(4.1, 5.2, 6.3, 7.4, 8.5, 9.6, 1.7, 2.8, 3.9)
  @test A + B == TensorValue(5.1, 7.2, 9.3, 11.4, 13.5, 15.6, 8.7, 10.8, 12.9)
  @test norm(A + B) ≈ 32.842807431765024
  @test B - A ≈ TensorValue(3.1, 3.2, 3.3, 3.4, 3.5, 3.6, -5.3, -5.2, -5.1)
  @test -A == TensorValue(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0)
  C = TensorValue{2,3}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  @test C + C == TensorValue{2,3}(2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
  @test TensorValue{2,2}(1, 2, 3, 4) + TensorValue{2,2}(0.5, 0.5, 0.5, 0.5) isa TensorValue{2,2,Float64}
  E = TensorValue{0,3,Float64}(())  # empty tensor, built by Gridap for vertex grids
  @test E + E == E
  @test E - E == E
  @test -E == E
  # high-order tensors
  for D in (2, 3)
    h, g = digits4(D), reverse(digits4(D))
    H, G = fourth(h), fourth(g)
    @test H + G == fourth(h + g)
    @test H - G == fourth(h - g)
    @test -H == fourth(-h)
    W = third(digits3(D))
    @test W + W == third(2 * digits3(D))
  end
end


@testset "identity" begin
  @test I2D == TensorValue(1.0, 0.0, 0.0, 1.0)
  @test I3D == TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  for (D, 𝕀) in ((2, 𝕀2D), (3, 𝕀3D))
    @test 𝕀 isa FourthOrderTensorValue{D,Float64}
    @test 𝕀 == fourth([Float64(i == k && j == l) for i in 1:D, j in 1:D, k in 1:D, l in 1:D])
    @test Tuple(TensorValue(Tuple(𝕀))) == Tuple(one(TensorValue{D^2,D^2,Float64}))  # the identity matrix
    A = TensorValue{D,D}(digits2(D)...)
    @test 𝕀 ⊙ A == A
  end
end


@testset "matrix representation and constructors" begin
  for D in (2, 3)
    H = fourth(digits4(D))
    M = TensorValue(Tuple(H))
    @test M isa TensorValue{D^2,D^2,Float64}
    @test M[1 + (2 - 1) * D, D] == H[1, 2, D, 1]  # rows (i,j), columns (k,l)
    @test FourthOrderTensorValue(Tuple(M)) == H
    @test FourthOrderTensorValue(Tuple(get_array(M))) == H
    w = digits3(D)
    @test ThirdOrderTensorValue(Tuple(reshape(w, D, D^2))) == third(w)
    @test FourthOrderTensorValue(Tuple(digits4(D))...) == H  # varargs
  end
end


@testset "products in index notation" begin
  # each product against its Einstein summation on plain arrays
  for D in (2, 3)
    a, b, h, v = randn(D, D), randn(D, D), randn(D, D, D, D), randn(D)
    A, B, H, V = TensorValue{D,D}(a...), TensorValue{D,D}(b...), fourth(h), VectorValue{D}(v...)
    r = 1:D
    @test A ⊗ᵢₖʲ V == third([a[i, k] * v[j] for i in r, j in r, k in r])
    @test A ⊗ᵢₖʲˡ B == fourth([a[i, k] * b[j, l] for i in r, j in r, k in r, l in r])
    @test A ⊗ᵢₗʲᵏ B == fourth([a[i, l] * b[j, k] for i in r, j in r, k in r, l in r])
    @test A ⊗ᵢₘʲᵐᵏˡ H ≈ fourth([sum(a[i, m] * h[j, m, k, l] for m in r) for i in r, j in r, k in r, l in r])
  end
  # the operands must have all their dimensions equal
  @test_throws DimensionMismatch I2D ⊗ᵢₖʲˡ I3D
end


@testset "contraction" begin
  for D in (2, 3)
    a, w, h = digits2(D), digits3(D), digits4(D)
    A, W, H, V = TensorValue{D,D}(a...), third(w), fourth(h), VectorValue{D}(digits1(D)...)
    r = 1:D
    @test A ⋅ H == fourth([sum(a[i, m] * h[m, j, k, l] for m in r) for i in r, j in r, k in r, l in r])
    @test W ⋅ W == fourth([sum(w[i, j, m] * w[m, k, l] for m in r) for i in r, j in r, k in r, l in r])
    @test W ⋅ A == third([sum(w[i, j, m] * a[m, k] for m in r) for i in r, j in r, k in r])
    @test W ⋅ V == TensorValue{D,D}([sum(w[i, j, m] * V[m] for m in r) for i in r, j in r]...)
    @test V ⋅ W == TensorValue{D,D}([sum(V[m] * w[m, i, j] for m in r) for i in r, j in r]...)
    # double contraction
    g = reverse(h)
    G = fourth(g)
    @test H ⋅² G == fourth([sum(h[i, j, m, n] * g[m, n, k, l] for m in r, n in r) for i in r, j in r, k in r, l in r])
    @test Tuple(H ⋅² G) == Tuple(TensorValue(Tuple(H)) ⋅ TensorValue(Tuple(G)))  # the product of the D²×D² matrices
    @test H ⋅² A == H ⊙ A
    @test A ⋅² H == TensorValue{D,D}([sum(a[m, n] * h[m, n, i, j] for m in r, n in r) for i in r, j in r]...)
    @test W ⋅² A == W ⊙ A
  end
  @test 𝕀2D ⋅² 𝕀2D == 𝕀2D
end


@testset "IIsym" begin
  A  = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
  B  = TensorValue(1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0)
  # 1. IIsym(I3D) ⊙ B yields the symmetric part 0.5 * (B + Bᵀ)
  @test IIsym(I3D) ⊙ B == 0.5 * (B + B')
  # 2. IIsym(A) ⊙ B yields 0.5 * (A ⋅ B ⋅ Aᵀ + A ⋅ Bᵀ ⋅ Aᵀ)
  @test IIsym(A) ⊙ B ≈ 0.5 * (A ⋅ B ⋅ A' + A ⋅ B' ⋅ A')
  # 3. IIsym(I3D) == ½ (δᵢₖδⱼₗ + δᵢₗδⱼₖ)
  @test IIsym(I3D) == 0.5 * (I3D ⊗ᵢₖʲˡ I3D + I3D ⊗ᵢₗʲᵏ I3D)
  @test I3D ⊗ᵢₖʲˡ I3D == 𝕀3D
end


@testset "push_forward_C_to_F" begin
  H = fourth(rand(3, 3, 3, 3))
  H = H' ⋅² H
  F = TensorValue(rand(9)...) * 1e-2 + I3D
  DCDF = F' ⊗ᵢₖʲˡ I3D + I3D ⊗ᵢₗʲᵏ F'
  S_ref = 0.5 * DCDF' ⋅² H ⋅² DCDF
  @test push_forward_C_to_F(F, H) ≈ S_ref
end


@testset "sqrt" begin
  A = TensorValue(1.:9...)
  A = A'*A + I3D
  sqrtA = TensorValue(sqrt(get_array(A)))
  @test isapprox(sqrt(A), sqrtA, rtol=1e-14)
end


@testset "cofactor" begin
  A = TensorValue(1.:9...) + I3D
  cofA = det(A) * inv(A')
  @test isapprox(cof(A), cofA)
end


@testset "regularize_hessian" begin
  # TensorValue from a Matrix by splatting (TensorValue(::Matrix) fails for 9x9 on Julia 1.14.0-alpha1)
  tensor(M::AbstractMatrix) = TensorValue{size(M, 1),size(M, 2)}(M...)
  # Reference: the former implementation of HessianRegularization (LAPACK eigen on the upper triangle)
  function reference_regularization(C::TensorValue, δ)
    vecval = LinearAlgebra.eigen(LinearAlgebra.Symmetric(get_array(C)))
    vec, val = vecval.vectors, vecval.values
    tensor(vec * LinearAlgebra.diagm(max.(δ, val)) * vec')
  end
  function random_symmetric(N, λ)
    Q = Matrix(LinearAlgebra.qr(randn(N, N)).Q)
    A = Q * LinearAlgebra.Diagonal(λ) * Q'
    tensor((A + A') / 2)
  end
  allocs(C, δ) = @allocated regularize_hessian(C, δ)
  # Both computations are backward stable: the eigenvectors carry an absolute error ~ ε‖C‖, which is the
  # error to expect in the result, whatever its own size (an indefinite C with large negative eigenvalues
  # clamped to a small δ gives ‖R‖ ≪ ‖C‖, and a tolerance relative to ‖R‖ alone fails on those draws)
  close(X, Y, C) = isapprox(X, Y; rtol=1e-12, atol=1e3 * eps() * LinearAlgebra.norm(get_array(C)))

  Random.seed!(20261030)
  for N in (4, 9), _ in 1:200
    scale = 10.0^rand(-2:5)
    δ = 1e-6 * scale
    spectra = (scale .* (0.1 .+ rand(N)),                          # positive definite
               scale .* randn(N),                                   # indefinite
               scale .* [rand((-1.0, 0.5, 2.0)) for _ in 1:N],      # repeated eigenvalues
               δ .* (1 .+ 1e-3 .* randn(N)),                        # eigenvalues around δ
               scale .* [i <= N ÷ 2 ? 0.0 : rand() for i in 1:N])  # singular
    for λ in spectra
      C = random_symmetric(N, λ)
      R = reference_regularization(C, δ)
      @test close(regularize_hessian(C, δ), R, C)
    end
    # only the upper triangle counts, as with Symmetric
    C = random_symmetric(N, scale .* randn(N))
    A = get_array(C) + 1e-3 * scale * LinearAlgebra.triu(randn(N, N), 1)'
    @test close(regularize_hessian(tensor(A), δ), reference_regularization(tensor(A), δ), tensor(A))
  end

  # positive definite above δ: returned unchanged
  C = random_symmetric(9, 1.0 .+ rand(9))
  @test regularize_hessian(C, 1e-6) == C

  # No allocations, on the fast path (positive definite) and on the eigen-decomposition path. Skipped when
  # bounds checking is forced (Pkg.test runs with --check-bounds=yes): @inbounds is then ignored and the
  # MMatrix scratch arrays can escape to the heap through the BoundsError path (depending on the Julia
  # version). The allocation-free behaviour is also tracked by the "regularize_hessian" benchmarks.
  forced_boundscheck = Base.JLOptions().check_bounds == 1
  for N in (4, 9)
    Cpd = random_symmetric(N, 1.0 .+ rand(N))
    Cind = random_symmetric(N, randn(N))
    regularize_hessian(Cpd, 1e-6); regularize_hessian(Cind, 1e-6)
    @test allocs(Cpd, 1e-6) == 0 skip = forced_boundscheck
    @test allocs(Cind, 1e-6) == 0 skip = forced_boundscheck
  end
end


@testset "regularize_hessian of fourth-order tensors" begin
  # same as the regularization of the D²×D² matrix
  for D in (2, 3)
    Q = Matrix(LinearAlgebra.qr(randn(D^2, D^2)).Q)
    M = Q * LinearAlgebra.Diagonal(randn(D^2)) * Q'
    H = FourthOrderTensorValue(Tuple((M + M') / 2))
    @test regularize_hessian(H, 1e-6) == FourthOrderTensorValue(Tuple(regularize_hessian(TensorValue(Tuple(H)), 1e-6)))
  end
end
