
"""
Return the aspect ratio of the underlying cartesian elements as a string.
This function is only available for an underlying `CartesianGrid`.

# Example
    aspect_ratio(Ω)             # "51:51:5"
    aspect_ratio(Ω, tol=0.05)   # "10:10:1"
    aspect_ratio(uh⁺, tol=0.1)  # "10:10:1" 
"""
function aspect_ratio(grid::CartesianGrid; tol=1e-6)
  descriptor = Gridap.Geometry.get_cartesian_descriptor(grid)
  sizes = descriptor.sizes

  rel = sizes ./ minimum(sizes)
  best = nothing
  best_error = Inf
  best_complexity = Inf

  for d in 0:20
    candidate = round.(Int, rel .* d)
    any(candidate .== 0) && continue
    approx = candidate ./ d
    err = maximum(abs.(approx .- rel) ./ rel)
    if err < best_error
      complexity = max(candidate...)
      if complexity < best_complexity
        best = candidate
        best_error = err
        best_complexity = complexity
      end
    end
  end

  if best_error > tol  # exact fallback
      denominators = denominator.(rationalize.(sizes))
      least_mult = lcm(denominators...)
      best = round.(Int, sizes .* least_mult)
  end

  join(best, ":")
end

function aspect_ratio(model::CartesianDiscreteModel; kwargs...)
  aspect_ratio(get_grid(model); kwargs...)
end

function aspect_ratio(triangulation::Triangulation; kwargs...)
  aspect_ratio(get_background_model(triangulation); kwargs...)
end

function aspect_ratio(f::CellField; kwargs...)
  aspect_ratio(get_triangulation(f); kwargs...)
end


"""
Return the element size for a cartesian mesh.
This function is only available for an underlying `CartesianGrid`.

# Example
    element_size(model)   # Compute the diagonal
    element_size(uh, :x)  # Get the x-size of the underlying grid
"""
function element_size(grid::CartesianGrid)
  descriptor = Gridap.Geometry.get_cartesian_descriptor(grid)
  sizes = descriptor.sizes
  sqrt(sum(abs2, sizes))
end

function element_size(grid::CartesianGrid, direction)
  descriptor = Gridap.Geometry.get_cartesian_descriptor(grid)
  direction_indices = Dict(:x => 1, :y => 2, :z => 3)
  index = direction_indices[direction]
  descriptor.sizes[index]
end

function element_size(model::CartesianDiscreteModel, args...)
  element_size(get_grid(model), args...)
end

function element_size(triangulation::Triangulation, args...)
  element_size(get_background_model(triangulation), args...)
end

function element_size(f::CellField, args...)
  element_size(get_triangulation(f), args...)
end
