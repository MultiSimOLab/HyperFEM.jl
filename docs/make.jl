using Documenter
using HyperFEM

readme_path = joinpath(@__DIR__, "..", "README.md")
index_path = joinpath(@__DIR__, "src", "index.md")

readme_content = read(readme_path, String)
header_start = findfirst("# Multiphysics", readme_content)
readme_content = header_start !== nothing ? readme_content[header_start.start:end] : readme_content
readme_content = replace(readme_content, r"(<p align=\"center\"><img.*?</p>)" => s"```@raw html\n\1\n```")
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
    HyperFEM.DiscreteModeling.EvolutionFunctions,
    HyperFEM.DiscreteModeling.CartesianTags,
    HyperFEM.ComputationalModels
  ],
  pages = [
    "Home" => "index.md",           # Inject README.md (previously cloned into index.md)
    "Tutorials" => "tutorials.md",  # Point to the tutorials repository
    "API reference" => "api.md"     # Inject the docstrings from the code
  ],
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://MultiSimOLab.github.io/HyperFEM.jl",
  ),
  checkdocs = :none,
  warnonly = false,
  linkcheck = get(ENV, "CI", "false") == "true", # check links only if running in ci
  linkcheck_ignore = [r"^mailto:"],
)

deploydocs(
  repo = "github.com/MultiSimOLab/HyperFEM.jl.git",
  devbranch = "main"
)
