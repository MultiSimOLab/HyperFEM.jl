using Documenter
using HyperFEM

readme_path = joinpath(@__DIR__, "..", "README.md")
index_path = joinpath(@__DIR__, "src", "index.md")
cp(readme_path, index_path, force=true)

makedocs(
  sitename = "HyperFEM.jl",
  modules = [HyperFEM],
  pages = [
    "HyperFEM" => "index.md",       # Inject README.md (previously cloned into index.md)
    "API reference" => "api.md"  # Inject the docstrings from the code
  ],
  checkdocs = :none,
  warnonly = true
)

deploydocs(
  repo = "github.com/MultiSimOLab/HyperFEM.jl.git",
  devbranch = "main"
)
