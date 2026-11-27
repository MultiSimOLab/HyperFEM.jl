# Time of the first call (compilation + run) of the tensor operations, in 3D.
# The same file runs in Gridap and in HyperFEM: if HyperFEM is in the environment, it is loaded
# and its methods replace Gridap's.
#
# Run from the repository root:
#   julia --project=benchmark benchmark/first_call_times.jl
#
# Every operation is called only once: after the first call the method is compiled and cached.
# All the cases run in the same Julia session, so the code shared by several operations is
# compiled only in the first of them.

using Gridap, Gridap.TensorValues

const HYPERFEM = Base.find_package("HyperFEM") !== nothing
HYPERFEM && @eval using HyperFEM

# `invokelatest` makes the method be compiled inside the timed region
function measure(label, f, args...)
  stats = try
    @timed Base.invokelatest(f, args...)
  catch
    println(rpad(label, 34), "unsupported")
    return 0.0
  end
  compilation = round(100 * stats.compile_time / stats.time; digits = 1)
  println(rpad(label, 34), lpad(round(stats.time; digits = 3), 8), " s   compilation ", lpad(compilation, 5), " %")
  stats.time
end

D = 3
v = VectorValue{D,Float64}(ntuple(i -> Float64(i), D))
u = VectorValue{D,Float64}(ntuple(i -> Float64(D + i), D))
A = TensorValue{D,D,Float64}(ntuple(i -> 1.0 + i / 10, D^2))
B = TensorValue{D,D,Float64}(ntuple(i -> 2.0 - i / 10, D^2))
W = ThirdOrderTensorValue{D,D,D,Float64}(ntuple(i -> 1.0 + i / 100, D^3))
H = HighOrderTensorValue{Tuple{D,D,D,D},Float64}(ntuple(i -> 1.0 + i / 1000, D^4))

println("Gridap $(pkgversion(Gridap))", HYPERFEM ? " + HyperFEM $(pkgversion(HyperFEM))" : "", ", Julia $VERSION")

total = 0.0

total += measure("outer / 1⊗1", ⊗, v, u)
total += measure("outer / 1⊗2", ⊗, v, A)
total += measure("outer / 2⊗1", ⊗, A, v)
total += measure("outer / 2⊗2", ⊗, A, B)
total += measure("outer / 1⊗3", ⊗, v, W)
total += measure("outer / 3⊗1", ⊗, W, v)

total += measure("inner / 1⊙1", ⊙, v, u)
total += measure("inner / 2⊙2", ⊙, A, B)
total += measure("inner / 3⊙3", ⊙, W, W)
total += measure("inner / 4⊙4", ⊙, H, H)

total += measure("dot / 1⋅1", ⋅, v, u)
total += measure("dot / 1⋅2", ⋅, v, A)
total += measure("dot / 2⋅1", ⋅, A, v)
total += measure("dot / 2⋅2", ⋅, A, B)
total += measure("dot / 1⋅3", ⋅, v, W)
total += measure("dot / 3⋅1", ⋅, W, v)
total += measure("dot / 2⋅3", ⋅, A, W)
total += measure("dot / 3⋅2", ⋅, W, A)
total += measure("dot / 3⋅3", ⋅, W, W)
total += measure("dot / 2⋅4", ⋅, A, H)
total += measure("dot / 4⋅2", ⋅, H, A)

total += measure("double dot / 2⋅²2", ⋅², A, B)
total += measure("double dot / 4⋅²2", ⋅², H, A)
total += measure("double dot / 2⋅²4", ⋅², A, H)
total += measure("double dot / 4⋅²4", ⋅², H, H)
total += measure("double dot / 3⋅²2", ⋅², W, A)
total += measure("double dot / 3⋅²4", ⋅², W, H)

total += measure("cross / 1×1", ×, v, u)

for (name, X) in (("1", v), ("2", A), ("3", W), ("4", H))
  global total += measure("arithmetic / $name+$name", +, X, X)
  global total += measure("arithmetic / $name-$name", -, X, X)
  global total += measure("arithmetic / -$name", -, X)
  global total += measure("arithmetic / α*$name", *, 2.0, X)
  global total += measure("arithmetic / $name/α", /, X, 2.0)
  global total += measure("arithmetic / $name'", transpose, X)
end

println(rpad("total", 34), lpad(round(total; digits = 2), 8), " s")
