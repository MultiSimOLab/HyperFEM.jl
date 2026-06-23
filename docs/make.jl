using Documenter
using HyperFEM

readme_path = joinpath(@__DIR__, "..", "README.md")
index_path = joinpath(@__DIR__, "src", "index.md")

readme_content = read(readme_path, String)
readme_content = replace(
    readme_content, 
    r"(<p align=\"center\"><img.*?</p>)" => s"```@raw html\n\1\n```"
)
write(index_path, readme_content)

makedocs(
  sitename = "HyperFEM.jl",
  modules = [
    HyperFEM,
    HyperFEM.TensorAlgebra,
    HyperFEM.PhysicalModels,
    HyperFEM.WeakForms,
    HyperFEM.Solvers,
    HyperFEM.DiscreteModeling,
    HyperFEM.ComputationalModels
  ],
  pages = [
    "HyperFEM" => "index.md",   # Inject README.md (previously cloned into index.md)
    "Tutorials" => "tutorials.md",
    "Modules" => [
      "Overview"            => "api/overview.md",
      "TensorAlgebra"       => "api/tensor_algebra.md",
      "PhysicalModels"      => "api/physical_models.md",
      "WeakForms"           => "api/weak_forms.md",
      "Solvers"             => "api/solvers.md",
      "DiscreteModeling"    => "api/discrete_modeling.md",
      "ComputationalModels" => "api/computational_models.md",
    ]
  ],
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://MultiSimOLab.github.io/HyperFEM.jl",
  ),
  checkdocs = :none,
  warnonly = true
)

deploydocs(
  repo = "github.com/MultiSimOLab/HyperFEM.jl.git",
  devbranch = "main"
)
