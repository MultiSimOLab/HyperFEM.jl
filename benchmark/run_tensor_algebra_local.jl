
using BenchmarkTools
SUITE = BenchmarkGroup()
include("TensorAlgebraBenchmarks/TensorAlgebraBenchmarksArrays.jl")
tune!(SUITE)
r = median(run(SUITE; verbose=true))
for (k,t) in sort(collect(BenchmarkTools.leaves(r)); by=first)
    println(
        rpad(join(k," / "),45),
        lpad(BenchmarkTools.prettytime(time(t)/NELEMS),12),
        lpad(allocs(t),6), " allocs  ",
        BenchmarkTools.prettymemory(memory(t))
    )
end
