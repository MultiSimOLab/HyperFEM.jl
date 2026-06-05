
"""
Create a new tag from a geometry and a coordinate-based filter function.
The filter function takes in vertex coordinates and returns a boolean values. A geometrical 
entity is tagged if all its vertices pass the filter.

# See also
- `Gridap.Geometry.face_labeling_from_vertex_filter`
- `Gridap.Geometry.merge!`
"""
function add_tag_from_vertex_filter!(
            labels::Gridap.Geometry.FaceLabeling,
            tag::String,
            geometry::Gridap.Geometry.DiscreteModel,
            filter::Function)
  new_labels = Gridap.Geometry.face_labeling_from_vertex_filter(geometry.grid_topology, tag, filter)
  merge!(labels, new_labels)
end


function add_tag_from_vertex_filter!(
            labels::Gridap.Geometry.FaceLabeling,
            geometry::Gridap.Geometry.DiscreteModel,
            tag::String,
            filter::Function)
  @warn "This method is deprecated. Call the function with: (labels, tag, geometry, filter)"
  add_tag_from_vertex_filter!(labels, tag, geometry, filter)
end
