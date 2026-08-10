
"""Return the linear index of a N-dimensional tensor."""
@inline _flat_idx(i::Int, j::Int, N::Int) = i + N*(j-1)
@inline _flat_idx(i::Int, j::Int, k::Int, N::Int) = _flat_idx(_flat_idx(i,j,N), k, N*N)
@inline _flat_idx(i::Int, j::Int, k::Int, l::Int, N::Int) = _flat_idx(_flat_idx(i,j,N), _flat_idx(k,l,N), N*N)

"""Return the cartesian indices of an N-dimensional second-order tensor."""
@inline _ij(α::Int, N::Int) = ((α-1)%N+1 ,(α-1)÷N+1)

"""Return the cartesian indices of an N-dimensional third-order tensor."""
@inline _ijk(α::Int, N::Int) = ((α-1)%N+1, ((α-1)÷N)%N+1, (α-1)÷(N*N)+1)

"""Return the cartesian indices of an N-dimensional fourth-order tensor."""
@inline _ijkl(α::Int, β::Int, N::Int) = (_ij(α,N)..., _ij(β,N)...)
@inline _ijkl(α::Int, N::Int) = _ijkl(_ij(α,N*N)...,N)
