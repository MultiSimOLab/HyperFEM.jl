
"Shortcuts for the tags of cartesian discrete models."
module CartesianTags

export face0YZ, face1YZ, faceX0Z, faceX1Z, faceXY0, faceXY1

export edgeX00, edgeX10, edgeX01, edgeX11
export edge0Y0, edge1Y0, edge0Y1, edge1Y1
export edge00Z, edge10Z, edge01Z, edge11Z

export corner000, corner100, corner010, corner110
export corner001, corner101, corner011, corner111

export edgeX00⁺, edgeX10⁺, edgeX01⁺, edgeX11⁺
export edge0Y0⁺, edge1Y0⁺, edge0Y1⁺, edge1Y1⁺
export edge00Z⁺, edge10Z⁺, edge01Z⁺, edge11Z⁺

export face0YZ⁺, face1YZ⁺, faceX0Z⁺, faceX1Z⁺, faceXY0⁺, faceXY1⁺

# Deprecated tags
export faceX0, faceX1, faceY0, faceY1, faceZ0, faceZ1

# --- Face tags ---

"Tags indicating the face at plane X0."
const face0YZ = [25]

"Tags indicating the face at plane X1."
const face1YZ = [26]

"Tags indicating the face at plane Y0."
const faceX0Z = [23]

"Tags indicating the face at plane Y1."
const faceX1Z = [24]

"Tags indicating the face at plane Z0."
const faceXY0 = [21]

"Tags indicating the face at plane Z1."
const faceXY1 = [22]

# --- Edge tags ---

"Tag indicating the edge at  X, Y0, Z0."
const edgeX00 = [9]

"Tag indicating the edge at  X, Y1, Z0."
const edgeX10 = [10]

"Tag indicating the edge at  X, Y0, Z1."
const edgeX01 = [11]

"Tag indicating the edge at  X, Y1, Z1."
const edgeX11 = [12]

"Tag indicating the edge at  X0, Y, Z0."
const edge0Y0 = [13]

"Tag indicating the edge at  X1, Y, Z0."
const edge1Y0 = [14]

"Tag indicating the edge at  X0, Y, Z1."
const edge0Y1 = [15]

"Tag indicating the edge at  X1, Y, Z1."
const edge1Y1 = [16]

"Tag indicating the edge at  X0, Y0, Z."
const edge00Z = [17]

"Tag indicating the edge at  X1, Y0, Z."
const edge10Z = [18]

"Tag indicating the edge at  X0, Y1, Z."
const edge01Z = [19]

"Tag indicating the edge at  X1, Y1, Z."
const edge11Z = [20]

# --- Corner tags ---

"Tag indicating the point at corner X0, Y0, Z0."
const corner000 = [1]

"Tag indicating the point at corner X1, Y0, Z0."
const corner100 = [2]

"Tag indicating the point at corner X0, Y1, Z0."
const corner010 = [3]

"Tag indicating the point at corner X1, Y1, Z0."
const corner110 = [4]

"Tag indicating the point at corner X0, Y0, Z1."
const corner001 = [5]

"Tag indicating the point at corner X1, Y0, Z1."
const corner101 = [6]

"Tag indicating the point at corner X0, Y1, Z1."
const corner011 = [7]

"Tag indicating the point at corner X1, Y1, Z1."
const corner111 = [8]

# --- Edge & corner tags ---

"Tags indicating points and edge at X, Y0, Z0."
const edgeX00⁺ = [edgeX00; corner000; corner100]

"Tags indicating points and edge at X, Y1, Z0."
const edgeX10⁺ = [edgeX10; corner010; corner110]

"Tags indicating points and edge at X, Y0, Z1."
const edgeX01⁺ = [edgeX01; corner001; corner101]

"Tags indicating points and edge at X, Y1, Z1."
const edgeX11⁺ = [edgeX11; corner011; corner111]

"Tags indicating points and edge at X0, Y, Z0."
const edge0Y0⁺ = [edge0Y0; corner000; corner010]

"Tags indicating points and edge at X1, Y, Z0."
const edge1Y0⁺ = [edge1Y0; corner100; corner110]

"Tags indicating points and edge at X0, Y, Z1."
const edge0Y1⁺ = [edge0Y1; corner001; corner011]

"Tags indicating points and edge at X1, Y, Z1."
const edge1Y1⁺ = [edge1Y1; corner101; corner111]

"Tags indicating points and edge at X0, Y0, Z."
const edge00Z⁺ = [edge00Z; corner000; corner001]

"Tags indicating points and edge at X1, Y0, Z."
const edge10Z⁺ = [edge10Z; corner100; corner101]

"Tags indicating points and edge at X0, Y1, Z."
const edge01Z⁺ = [edge01Z; corner010; corner011]

"Tags indicating points and edge at X1, Y1, Z."
const edge11Z⁺ = [edge11Z; corner110; corner111]

# --- Face & edge & corner tags ---

"Tags indicating points, edges and faces at plane X0."
const face0YZ⁺ = [face0YZ; edge00Z⁺; edge01Z⁺; edge0Y0⁺; edge0Y1⁺]

"Tags indicating points, edges and faces at plane X1."
const face1YZ⁺ = [face1YZ; edge10Z⁺; edge11Z⁺; edge1Y0⁺; edge1Y1⁺]

"Tags indicating points, edges and faces at plane Y0."
const faceX0Z⁺ = [faceX0Z; edgeX00⁺; edgeX01⁺; edge00Z⁺; edge10Z⁺]

"Tags indicating points, edges and faces at plane Y1."
const faceX1Z⁺ = [faceX1Z; edgeX10⁺; edgeX11⁺; edge01Z⁺; edge11Z⁺]

"Tags indicating points, edges and faces at plane Z0."
const faceXY0⁺ = [faceXY0; edgeX00⁺; edgeX10⁺; edge0Y0⁺; edge1Y0⁺]

"Tags indicating points, edges and faces at plane Z1."
const faceXY1⁺ = [faceXY1; edgeX01⁺; edgeX11⁺; edge0Y1⁺; edge1Y1⁺]

# --- Deprecations ---

Base.@deprecate_binding faceX0 face0YZ
Base.@deprecate_binding faceX1 face1YZ
Base.@deprecate_binding faceY0 faceX0Z
Base.@deprecate_binding faceY1 faceX1Z
Base.@deprecate_binding faceZ0 faceXY0
Base.@deprecate_binding faceZ1 faceXY1

end
