"""
    GridapPerformancePatches

Temporary methods that restore, on Gridap 0.20, the performance of Gridap 0.19.9 in two places. They add new, more
specific methods (no method of Gridap is overwritten, so the package precompiles) and give bit-for-bit the same
results. To be removed once Gridap ships the fixes upstream.

1. `Gridap.Arrays._common_size`: Gridap 0.20 replaced `all(map(f, a))` by `all(f, a)` on the tuple of arrays of every
   `lazy_map`; with arrays of different types `all(f, a)` does not specialise and dispatches at run time. Methods for
   1 to 8 arrays with the 0.19.9 body.
2. `OperationField` whose operation is itself a `Field` (e.g. an `Interpolable` composed with the cell map, as in
   `interpolate_everywhere!(Interpolable(uh), …)`), evaluated on an array of points: Gridap 0.20 evaluates it with
   `Broadcasting(op)`, which calls `evaluate(op, x)` point by point and builds the cache of the operation at every
   point (for an `Interpolable`, the KD-tree search structure and the active model of its triangulation). As in 0.19.9,
   the cache of the operation is built once and reused.
"""
module GridapPerformancePatches

using Gridap
using Gridap.Arrays: CachedArray, setsize!, testitem
using Gridap.Fields: Field, OperationField
import Gridap.Arrays: _common_size, return_cache, evaluate!

# 1. _common_size, 1 to 8 arrays
for N in 1:8
  as = [Symbol(:a, i) for i in 1:N]
  sig = [:($a::AbstractArray) for a in as]
  @eval function _common_size($(sig...))
    a = ($(as...),)
    a1 = a[1]
    if all(map(ai -> size(a1) == size(ai), a))
      size(a1)
    else
      (length(a1),)
    end
  end
end

# 2. OperationField with a Field as operation, on an array of points
function return_cache(c::OperationField{<:Field}, x::AbstractArray{<:Point})
  cf = map(fi -> return_cache(fi, x), c.fields)
  lx = map((ci, fi) -> evaluate!(ci, fi, x), cf, c.fields)
  ck = return_cache(c.op, map(testitem, lx)...)
  r = [evaluate!(ck, c.op, map(lxi -> lxi[i], lx)...) for i in eachindex(x)]
  CachedArray(reshape(r, size(x))), ck, cf
end

function evaluate!(cache, c::OperationField{<:Field}, x::AbstractArray{<:Point})
  ca, ck, cf = cache
  setsize!(ca, size(x))
  lx = map((ci, fi) -> evaluate!(ci, fi, x), cf, c.fields)
  r = ca.array
  for i in eachindex(x)
    @inbounds r[i] = evaluate!(ck, c.op, map(lxi -> lxi[i], lx)...)
  end
  r
end

end # module
