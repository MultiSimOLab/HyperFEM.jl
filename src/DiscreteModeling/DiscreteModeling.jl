
"""
A bundle of helper tools to work with discrete models in space/time.
"""
module DiscreteModeling

using Gridap

export CartesianTags
export EvolutionFunctions
export add_tag_from_vertex_filter!
export aspect_ratio
export element_size

include("CartesianTags.jl")
include("EvolutionFunctions.jl")
include("FaceLabeling.jl")
include("MeshDescriptor.jl")

end
