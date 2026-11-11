# Code generation kernel of the tensor algebra.
#
# Every operation of the tensor algebra is an `@generated` function whose body is built, at compile time,
# by the functions of this file as a Julia expression: a fully unrolled constructor call, with zero
# allocations. The tensors use the flattened convention: a tensor of order `o` in dimension `D` keeps its
# `D^o` components in column-major order (first index fastest), stored in
#   order 1  `VectorValue{D}`
#   order 2  `TensorValue{D,D}`
#   order 3  `TensorValue{D,D²}`
#   order 4  `TensorValue{D²,D²}`
#
# The functions of this file run at compile time only (inside the generators), never at run time.

"""Column-major linear index of the cartesian index `I` of a tensor of dimension `D`."""
_linear_index(I::Tuple, D::Int) = 1 + sum(((I[k] - 1) * D^(k - 1) for k in eachindex(I)); init=0)

"""Cartesian indices of a tensor of order `o` in dimension `D`, in storage (column-major) order."""
_cartesian_indices(o::Int, D::Int) = vec(collect(Iterators.product(ntuple(_ -> 1:D, o)...)))

"""Expression reading the component `I` of the tensor bound to the symbol `s`."""
_component(s::Symbol, I::Tuple, D::Int) = :($s.data[$(_linear_index(I, D))])

"""Shape of a tensor of order `o` in dimension `D` (flattened convention)."""
function _flat_shape(o::Int, D::Int)
  o == 1 && return (D,)
  o == 2 && return (D, D)
  o == 3 && return (D, D^2)
  o == 4 && return (D^2, D^2)
  error("Tensors of order $o are not supported")
end

"""Constructor of a tensor of order `o` in dimension `D` (flattened convention)."""
function _flat_constructor(o::Int, D::Int)
  o == 1 && return :(VectorValue{$D})
  S = _flat_shape(o, D)
  :(TensorValue{$(S[1]),$(S[2])})
end

"""Shape and number of components of a `MultiValue` type."""
_shape(::Type{<:MultiValue{S}}) where {S} = Tuple(S.parameters)
_num_components(::Type{<:MultiValue{S,T,N,L}}) where {S,T,N,L} = L

"""
    _dimension(T::Type{<:MultiValue}, o::Int)

Dimension `D` of the type `T` storing a tensor of order `o`, or `nothing` when `T` does not store
a tensor of order `o` in the flattened convention.
"""
function _dimension(T::Type{<:MultiValue}, o::Int)
  D = round(Int, _num_components(T)^(1 / o))
  _shape(T) == _flat_shape(o, D) ? D : nothing
end

"""
    _construct(constructor, entries)

Expression calling `constructor` with the tuple of the component expressions `entries`.
"""
_construct(constructor, entries) = Expr(:call, constructor, Expr(:tuple, entries...))

"""
    _tensor_expr(o, D, entry; constructor)

Expression of the tensor of order `o` in dimension `D` whose component `I` is the expression `entry(I)`.
A scalar when `o == 0`.
"""
function _tensor_expr(o::Int, D::Int, entry; constructor=o == 0 ? nothing : _flat_constructor(o, D))
  o == 0 && return entry(())
  _construct(constructor, [entry(I) for I in _cartesian_indices(o, D)])
end

"""
    _sum_expr(terms, T)

Sum of the expressions `terms`, evaluated from left to right (n-ary `+`); `zero(T)` when empty.
"""
function _sum_expr(terms, T)
  isempty(terms) && return :(zero($T))
  length(terms) == 1 && return terms[1]
  Expr(:call, :+, terms...)
end

"""
    _map_expr(op, constructor, N, operands)

Expression of the tensor whose `N` components are `op` applied component-wise to the `operands` (symbols).
"""
function _map_expr(op::Symbol, constructor, N::Int, operands)
  _construct(constructor, [Expr(:call, op, (:($s.data[$i]) for s in operands)...) for i in 1:N])
end

"""
    _einsum_expr(out, operands, D, T; coefficient, constructor, reorder)

Expression of the tensor `R[out...] = Σ coefficient(labels) * Π operand[labels...]` in Einstein notation.

- `operands` is a list of `symbol => labels`, where `labels` is a tuple of `Char`, one per index.
- `out` are the labels of the result, in order. Every other label of the operands is summed.
- The summed labels run in column-major order: the first summed label (alphabetically) is the fastest.
- The factors of each term keep the order of the operands and the terms are added from left to right.
- `coefficient(values)` gives a constant factor from the values of the labels (a `Dict{Char,Int}`).
  The terms with a zero factor are skipped and a factor `-1` negates the term (`-(a*b)`).
- `reorder(terms)` may change the order of the terms of each component. Each term is a named tuple
  `(coefficient, index, expr)`, where `index` is the linear index of the component of the first operand.
"""
function _einsum_expr(out::Tuple, operands, D::Int, T;
    coefficient=_ -> 1, constructor=length(out) == 0 ? nothing : _flat_constructor(length(out), D),
    reorder=identity)
  summed = sort(setdiff(unique(collect(Iterators.flatten(last.(operands)))), out))
  function entry(I)
    values = Dict{Char,Int}(zip(out, I))
    terms = []
    for J in _cartesian_indices(length(summed), D)
      merge!(values, Dict(zip(summed, J)))
      c = coefficient(values)
      iszero(c) && continue
      indices = [map(l -> values[l], labels) for (_, labels) in operands]
      factors = [_component(s, I, D) for ((s, _), I) in zip(operands, indices)]
      p = length(factors) == 1 ? factors[1] : Expr(:call, :*, factors...)
      expr = c == 1 ? p : c == -1 ? :(-$p) : :($c * $p)
      push!(terms, (coefficient=c, index=_linear_index(indices[1], D), expr=expr))
    end
    _sum_expr([t.expr for t in reorder(terms)], T)
  end
  _tensor_expr(length(out), D, entry; constructor=constructor)
end

"""Levi-Civita symbol in three dimensions."""
_levi_civita(i::Int, j::Int, k::Int) = (i - j) * (j - k) * (k - i) ÷ 2

"""
Order of the terms of the cross products kept from their former hand-written implementation, so that the
results are bitwise identical: terms sorted by the component of the first operand, with the first positive
term moved to the front.
"""
function _cross_terms_order(terms)
  sorted = sort(terms; by=t -> t.index)
  k = findfirst(t -> t.coefficient > 0, sorted)
  k === nothing ? sorted : [sorted[k]; sorted[1:k-1]; sorted[k+1:end]]
end


# =====================
# Index notation
# =====================

const _SUBSCRIPT_LABELS = Dict('ᵢ' => 'i', 'ⱼ' => 'j', 'ₖ' => 'k', 'ₗ' => 'l', 'ₘ' => 'm', 'ₙ' => 'n', 'ₚ' => 'p')
const _SUPERSCRIPT_LABELS = Dict('ⁱ' => 'i', 'ʲ' => 'j', 'ᵏ' => 'k', 'ˡ' => 'l', 'ᵐ' => 'm', 'ⁿ' => 'n', 'ᵖ' => 'p')
const _FREE_LABELS = ('i', 'j', 'k', 'l')
const _SUMMED_LABELS = ('m', 'n', 'p')

"""
    _index_labels(op::Symbol) -> (labels of the first operand, labels of the second operand)

Parse the index notation of a product operator, e.g. `⊗ᵢⱼₘₙᵐⁿᵏˡ` → `(('i','j','m','n'), ('m','n','k','l'))`.
The subscripts are the indices of the first operand and the superscripts the indices of the second one.
The notation must be canonical, so that each operation has a single name:
- the free indices are `i, j, k, l`, they appear once and give the order of the indices of the result;
- the summed indices are `m, n, p`, in order of first appearance, and they appear once in each operand.
"""
function _index_labels(op::Symbol)
  chars = collect(String(op))
  @assert first(chars) == '⊗' "$op is not a product in index notation"
  marks = chars[2:end]
  nsub = count(c -> haskey(_SUBSCRIPT_LABELS, c), marks)
  @assert all(c -> haskey(_SUBSCRIPT_LABELS, c), marks[1:nsub]) &&
          all(c -> haskey(_SUPERSCRIPT_LABELS, c), marks[nsub+1:end]) "$op: subscripts must precede superscripts"
  la = Tuple(_SUBSCRIPT_LABELS[c] for c in marks[1:nsub])
  lb = Tuple(_SUPERSCRIPT_LABELS[c] for c in marks[nsub+1:end])
  labels = (la..., lb...)
  free = filter(l -> l in _FREE_LABELS, labels)
  summed = unique(filter(l -> l in _SUMMED_LABELS, labels))
  @assert sort(collect(free)) == collect(_FREE_LABELS[1:length(free)]) "$op: the free indices must be $(join(_FREE_LABELS[1:length(free)], ", ")), once each"
  @assert summed == collect(_SUMMED_LABELS[1:length(summed)]) "$op: the summed indices must be $(join(_SUMMED_LABELS[1:length(summed)], ", ")), in order of appearance"
  @assert all(l -> count(==(l), la) == 1 && count(==(l), lb) == 1, summed) "$op: a summed index must appear once in each operand"
  (la, lb)
end

"""
    _einsum(Val(la), Val(lb), A, B)

Product of `A` and `B` in index notation with the labels `la` and `lb` (see [`_index_labels`](@ref)).
The free labels give the indices of the result, in alphabetical order.
"""
@inline @generated function _einsum(::Val{la}, ::Val{lb}, A::MultiValue, B::MultiValue) where {la,lb}
  D = _dimension(A, length(la))
  if D === nothing || _dimension(B, length(lb)) != D
    msg = "Tensors of order $(length(la)) and $(length(lb)) in the same dimension expected, got $A and $B"
    return :(throw(DimensionMismatch($msg)))
  end
  labels = (la..., lb...)
  out = Tuple(sort(filter(l -> count(==(l), labels) == 1, collect(labels))))
  _einsum_expr(out, (:A => la, :B => lb), D, promote_type(eltype(A), eltype(B)))
end
