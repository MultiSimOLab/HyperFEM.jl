
"""
    IterativeSolver(solver::String; kwargs...)

Wrapper of the solvers of the IterativeSolvers package (`solver` is the name of the function, e.g.
`"cg"`, `kwargs` its keyword arguments; `Pl` a function of the matrix building the preconditioner).

!!! warning "Deprecated"
    It parses and evaluates `solver` at every solve, does not warm-start and rebuilds the
    preconditioner at every Newton iteration. It will be removed in the next release; the Krylov
    solvers with algebraic multigrid of the solver plan (stage S1) replace it.
"""
#constructor
struct IterativeSolver <: Gridap.Algebra.LinearSolver
  solver::String
  kwargs::Dict
end

function IterativeSolver(solver::String; kwargs...)
  Base.depwarn("`IterativeSolver` is deprecated and will be removed in the next release", :IterativeSolver)
  IterativeSolver(solver, Dict{Symbol,Any}(kwargs))
end


# SymbolicSetup
struct IterativeSolverss <: Gridap.Algebra.SymbolicSetup
  itsolver::IterativeSolver
end


function Gridap.Algebra.symbolic_setup(solver::IterativeSolver, mat::AbstractMatrix)
  IterativeSolverss(solver)
end

# NumericalSetup
mutable struct IterativeSolverns{T<:AbstractMatrix} <: Gridap.Algebra.NumericalSetup
  A::T
  itsolver::IterativeSolver
  kwargs::Dict
end

function Gridap.Algebra.numerical_setup(ss::IterativeSolverss, mat::AbstractMatrix)
  kwargs_ = copy(ss.itsolver.kwargs)
  if haskey(kwargs_, :Pl)
    kwargs_[:Pl] = kwargs_[:Pl](mat)
  end
  IterativeSolverns(mat, ss.itsolver, kwargs_)
end

function Gridap.Algebra.numerical_setup!(ns::IterativeSolverns, mat::AbstractMatrix)
  ns.A = mat
  if haskey(ns.kwargs, :Pl)
    ns.kwargs[:Pl] = ns.itsolver.kwargs[:Pl](mat)
  end
  ns
end


function Gridap.Algebra.solve!(
  x::AbstractVector, ns::IterativeSolverns, b::AbstractVector)
  fexp = eval(Meta.parse(ns.itsolver.solver))
  copy_entries!(x, fexp(ns.A, b; ns.kwargs...))
end

 
