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
    "HyperFEM" => "index.md",       # Inject README.md (previously cloned into index.md)
    "API reference" => "api.md"  # Inject the docstrings from the code
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
