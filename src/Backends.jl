"""
Assembly backends of HyperFEM: who evaluates and assembles the weak forms in the drivers and who
stores and updates the state variables. [`LazyBackend`](@ref) (`:lazy`, the default) is Gridap's lazy
cell-by-cell evaluation; [`CompiledBackend`](@ref) (`:compiled`) the compiled, threaded kernels of
HyperFEMKernels, enabled by its extension (`using HyperFEMKernels`). Every `backend` keyword accepts
the symbol or the backend itself (`as_backend`).
"""
module Backends

export AbstractBackend, LazyBackend, CompiledBackend, as_backend, backend_name

"Supertype of the assembly backends; a backend is a singleton the drivers dispatch on."
abstract type AbstractBackend end

"Gridap's lazy evaluation of the weak forms (`FEOperator`, `SparseMatrixAssembler`); `backend = :lazy`."
struct LazyBackend <: AbstractBackend end

"""
The weak forms compiled into fused, threaded kernels by HyperFEMKernels (`KernelOperator`);
`backend = :compiled`. Its methods are provided by the extension HyperFEMKernelsExt
(`using HyperFEMKernels`); without it every use fails with a clear error.
"""
struct CompiledBackend <: AbstractBackend end

"""
    as_backend(b) -> AbstractBackend

The backend named by the symbol `:lazy` or `:compiled`, or `b` itself when it is already a backend.
"""
as_backend(b::AbstractBackend) = b
as_backend(s::Symbol) = s === :lazy ? LazyBackend() : s === :compiled ? CompiledBackend() :
    error("HyperFEM: unknown backend :$s (use :lazy or :compiled)")
as_backend(b) = error("HyperFEM: `backend` must be :lazy, :compiled or an AbstractBackend; got $(typeof(b))")

"Name of a backend as the symbol of its keyword (`:lazy`, `:compiled`)."
backend_name(::LazyBackend) = :lazy
backend_name(::CompiledBackend) = :compiled
backend_name(b::AbstractBackend) = nameof(typeof(b))

"Error of a backend whose implementation is not loaded or does not cover `what`."
not_available(b::AbstractBackend, what::String) =
    error("HyperFEM: backend :$(backend_name(b)) is not available for $what")
not_available(::CompiledBackend, what::String) =
    error("HyperFEM: the compiled backend is not loaded for $what; `using HyperFEMKernels` enables it")

end # module
