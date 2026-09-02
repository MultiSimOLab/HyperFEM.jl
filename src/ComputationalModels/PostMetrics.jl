module PostMetrics

using Gridap
using HyperFEM.PhysicalModels

export volume_diff


"""
Calculate the variation of the volume with respect to the undeformed configuration.
"""
function volume_diff(uh, dΩ)
  @warn "The function volume_diff is deprecated and it will be removed after release 0.0.7."
  F, _, J = Kinematics(Mechano).metrics
  sum(∫(J ∘ F ∘ ∇(uh) -1.0)dΩ) / sum(∫(1.0)dΩ)
end


end