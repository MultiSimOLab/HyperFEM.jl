module IO

using HyperFEM

export setupfolder
export projdir
export stem
export MockPVD
export mockpvd

"""
Make sure the specified folder exists.

# Examples
    setupfolder("output")                    # Remove everything inside "output"
    setupfolder("output", ".*")              # Remove everything inside "output"
    setupfolder("output", "all")             # Remove everything inside "output"
    setupfolder("results", ".vtu")           # Remove only .vtu files
    setupfolder("results", [".vtu", ".pvd"]) # Remove multiple file types
    setupfolder("data", nothing)             # Keep existing contents
    setupfolder("data", remove=nothing)      # Keep existing contents
"""
function setupfolder(path::String; remove::Any="all")
  if isdir(path)
    cleandir(path, remove)
  end
  mkpath(path)
end

function cleandir(path::String, remove::String)
  if remove === "all"
    rm(path,recursive=true)
  else
    foreach(rm, filter(endswith(remove), readdir(path,join=true)))
  end
end

function cleandir(path::String, remove::AbstractVector{String})
  foreach(r -> cleandir(path,r), remove)
end

function cleandir(::String, ::Nothing)
end

"""
Return the path to the specified folders relative to the HyperFEM path.

# Examples
    folder = projdir("data", "sims")
    folder = projdir("test/data/mesh.msh")
"""
function projdir(folders::String...)
  base_folder = dirname(dirname(pathof(HyperFEM)))
  joinpath(base_folder, folders...)
end

"""
Return the file name without extension.

# Examples
    stem("a/b/c")          → "c"
    stem("a/b/c.jl")       → "c"
    stem("a/b/.gitignore") → ".gitignore"
    stem("a/b/foo.tar.gz") → "foo.tar"
"""
function stem(path::AbstractString)
  name = basename(path)
  root, ext = splitext(name)
  root
end

"""
Return the full path without extension.

# Examples
    julia > nosuffix("home/user/example.jl")
    "home/user/example"
    
    julia > nosuffix("home/user/example.a.jl")
    "home/user/example.a"
    
    julia > nosuffix("home/user/example")
    "home/user/example"
"""
function nosuffix(path::AbstractString)
  splitext(path)[1]
end

"""
Mock struct to emulate pvd files.
"""
struct MockPVD end

"""
Mock function to emulate the creation of pvd files.

# Example
    pvdstrategy = writevtk ? createpvd : mockpvd
    pvdstrategy(outpath) do pvd
      ...
    end
"""
mockpvd(f, args...; kwargs...) = f(MockPVD())

end