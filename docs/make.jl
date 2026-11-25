using Documenter
using HyperFEM

readme_path = joinpath(@__DIR__, "..", "README.md")
index_path = joinpath(@__DIR__, "src", "index.md")

readme_content = read(readme_path, String)
header_start = findfirst("# Multiphysics", readme_content)
readme_content = header_start !== nothing ? readme_content[header_start.start:end] : readme_content
readme_content = replace(readme_content, "docs/src/assets/" => "assets/")  # README paths are relative to the repo root
# links of the README to pages of this documentation become relative links (checked by Documenter, not by linkcheck)
readme_content = replace(readme_content, r"https://multisimolab\.github\.io/HyperFEM\.jl/dev/(\w+)/" => s"\1.md")
readme_content = replace(readme_content, r"(<p align=\"center\"><img.*?</p>)" => s"```@raw html\n\1\n```")
write(index_path, readme_content)

# Tutorials page: the intro in tutorials_intro.md plus a list generated from the tutorials repository, one entry
# per script src/E<NN>_<Name>.jl. The link text and the summary come from the script's Literate header:
#   # # Title
#   # One-line summary
# Without a header, the file name is the link text and the summary is an empty string. Each title links to the
# page rendered by the tutorials' own documentation, and "source" to the script at the commit that was listed.
const TUTORIALS_REPO = "https://github.com/MultiSimOLab/HyperFEMTutorials"
const TUTORIALS_SITE = "https://multisimolab.github.io/HyperFEMTutorials/dev"

function tutorial_header(path)
  lines = readlines(path)
  start = findfirst(!isempty ∘ strip, lines)
  start === nothing && return ("", "")
  m = match(r"^#\s*#\s+(.+?)\s*$", lines[start])
  m === nothing && return ("", "")
  summary = String[]
  for line in lines[start+1:end]
    s = match(r"^#\s?(.*?)\s*$", line)
    (s === nothing || isempty(s[1]) || startswith(s[1], "#")) && break
    push!(summary, s[1])
  end
  return (String(m[1]), join(summary, " "))
end

function tutorials_list(repo_dir, sha)
  io = IOBuffer()
  for file in sort(filter(f -> occursin(r"^E\d+_.*\.jl$", f), readdir(joinpath(repo_dir, "src"))))
    title, summary = tutorial_header(joinpath(repo_dir, "src", file))
    name = splitext(file)[1]
    print(io, "* [", isempty(title) ? file : title, "](", TUTORIALS_SITE, "/tutorials/", name, "/)")
    print(io, " ([source](", TUTORIALS_REPO, "/blob/", sha, "/src/", file, "))")
    println(io, isempty(summary) ? "" : ": " * summary)
  end
  return String(take!(io))
end

function write_tutorials_page(page_path)
  intro = read(joinpath(@__DIR__, "tutorials_intro.md"), String)
  list = mktempdir() do dir
    try
      run(pipeline(`git clone --quiet --depth 1 $TUTORIALS_REPO $dir`; stderr = devnull))
      tutorials_list(dir, readchomp(`git -C $dir rev-parse HEAD`))
    catch e
      @warn "Could not fetch the tutorials repository; the tutorials page will only link to it." exception = e
      "See the [tutorials documentation]($TUTORIALS_SITE/).
"
    end
  end
  write(page_path, rstrip(intro) * "

" * list)
end

write_tutorials_page(joinpath(@__DIR__, "src", "tutorials.md"))

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
    "Tutorials" => "tutorials.md",  # Generated from the tutorials repository
    "Load stepping and solvers" => "solvers.md",
    "Staggered models" => "staggered.md",
    "Time integration" => "transient.md",
    "Tensor algebra" => "tensor_algebra.md",
    "API reference" => "api.md"     # Inject the docstrings from the code
  ],
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://MultiSimOLab.github.io/HyperFEM.jl",
    size_threshold = 400 * 2^10,        # the API reference is one page with every docstring (~210 KiB)
    size_threshold_warn = 250 * 2^10,
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
