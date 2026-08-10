 

function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end


function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end


@inline @generated function (+)(A::TensorValue{D,D}, B::TensorValue{D,D}) where {D}
  str = ""
  for i in 1:D*D
    str *= "A.data[$i] + B.data[$i], "
  end
  Meta.parse("TensorValue{D,D}($str)")
end


function Gridap.TensorValues.outer(A::TensorValue{D,D}, B::TensorValue{D,D}) where {D}
  return (A ⊗₁₂³⁴ B)
end

function Gridap.TensorValues.outer(A::VectorValue{D}, B::VectorValue{D}) where {D}
  return (A ⊗₁² B)
end

function Gridap.TensorValues.outer(A::VectorValue{D}, B::TensorValue{D,D}) where {D}
  return (A ⊗₁²³ B)
end

function Gridap.TensorValues.outer(A::TensorValue{D,D}, B::VectorValue{D}) where {D}
  return (A ⊗₁₂³ B)
end

function Gridap.TensorValues.outer(A::TensorValue{4,2}, B::VectorValue{2})
  return (A ⊗₁₂₃⁴ B)
end

function Gridap.TensorValues.outer(A::TensorValue{9,3}, B::VectorValue{3})
  return (A ⊗₁₂₃⁴ B)
end


"""
    ⊗₁²(A::VectorValue{D}, B::VectorValue{D})::TensorValue{D,D}

Outer product of two first-order tensors (vectors), returning a second-order tensor (matrix).
"""
@inline @generated function (⊗₁²)(A::VectorValue{D}, B::VectorValue{D}) where {D}
  str = ""
  for iB in 1:D
    for iA in 1:D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D,D}($str)")
end


"""
    ⊗₁₃²⁴(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}

Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `D² x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₂³⁴)(A::TensorValue{D,D}, B::TensorValue{D,D}) where {D}
  str = ""
  for iB in 1:D*D
    for iA in 1:D*D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D*D,D*D}($str)")
end


"""
    ⊗₁₃²⁴(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}

Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `D² x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₃²⁴)(A::TensorValue{D}, B::TensorValue{D}) where D
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          str *= "A[$i,$k]*B[$j,$l],"
        end
      end
    end
  end
  Meta.parse("TensorValue{D*D}($str)")
end


"""
    ⊗₁₄²³(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}

Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
represented in a `D² x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₄²³)(A::TensorValue{D}, B::TensorValue{D}) where D
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          str *= "A[$i,$l]*B[$j,$k],"
        end
      end
    end
  end
  Meta.parse("TensorValue{D*D}($str)")
end


"""
    ⊗₁²³(A::VectorValue{D}, B::TensorValue{D})::TensorValue{D,D*D}

Outer product of a first-order and second-order tensors (vector and matrix),
returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁²³)(V::VectorValue{D}, A::TensorValue{D,D}) where {D}
  str = ""
  for iA in 1:D*D
    for iV in 1:D
      str *= "A.data[$iA] * V.data[$iV], "
    end
  end
  Meta.parse("TensorValue{D,D*D}($str)")
end


"""
    ⊗₁₂³(A::TensorValue{D}, B::VectorValue{D})::TensorValue{D,D*D}

Outer product of a second-order and first-order tensors (matrix and vector),
returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₂³)(A::TensorValue{D,D}, V::VectorValue{D}) where {D}
  str = ""
  for iV in 1:D
    for iA in 1:D*D
      str *= "A.data[$iA] * V.data[$iV], "
    end
  end
  Meta.parse("TensorValue{D,D*D}($str)")
end


"""
    ⊗₁₃²(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D,D*D}

Outer product of a second-order and first-order tensors (matrix and vector),
returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₃²)(A::TensorValue{D}, V::VectorValue{D}) where D
  str = ""
  for k in 1:D
    for j in 1:D
      for i in 1:D
        str *= "A[$i,$k]*V[$j],"
      end
    end
  end
  Meta.parse("TensorValue{D,D*D}($str)")
end


"""
    ⊗₁₂₃⁴(A::TensorValue{D²,D}, B::TensorValue{D})::TensorValue{D,D*D}

Outer product of a third-order and first-order tensors (tensor and vector),
returning a fourth-order tensor represented in a `D² x D²` flattened matrix using combined indices.
"""
@inline @generated function (⊗₁₂₃⁴)(A::TensorValue{D,D²}, V::VectorValue{D}) where {D,D²}
  @assert D*D == D² "Third- and first-order tensors size mismatch with $D² × $D and $D"
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          a = _flat_idx(i,j,D)
          str *= "A[$k,$a]*V[$l],"
        end
      end
    end
  end
  Meta.parse("TensorValue{D*D,D*D}($str)")
end

@inline @generated function IIsym(A::TensorValue{D,D}) where {D}
  str = ""
  for a in 1:D^4
    i, j, k, l = _ijkl(a, D)
    str *= "0.5 * (A[$i, $k] * A[$j, $l] + A[$i, $l] * A[$j, $k]),"
  end
  Meta.parse("TensorValue{D*D,D*D}($str)")
end

@inline function (×ᵢ⁴)(A::TensorValue{3,3})
  TensorValue{9,9}(0.0, 0.0, 0.0, 0.0, A[9], -A[8], 0.0, -A[6], A[5], 0.0, 0.0, 0.0, -A[9],
    0.0, A[7], A[6], 0.0, -A[4], 0.0, 0.0, 0.0, A[8], -A[7], 0.0, -A[5], A[4], 0.0, 0.0, -A[9],
    A[8], 0.0, 0.0, 0.0, 0.0, A[3], -A[2], A[9], 0.0, -A[7], 0.0, 0.0, 0.0, -A[3], 0.0,
    A[1], -A[8], A[7], 0.0, 0.0, 0.0, 0.0, A[2], -A[1], 0.0, 0.0, A[6], -A[5], 0.0,
    -A[3], A[2], 0.0, 0.0, 0.0, -A[6], 0.0, A[4], A[3], 0.0, -A[1],
    0.0, 0.0, 0.0, A[5], -A[4], 0.0, -A[2], A[1], 0.0, 0.0, 0.0, 0.0)
end


@inline function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue{3,3}(A[5] * B[9] - A[6] * B[8] - A[8] * B[6] + A[9] * B[5],
    A[6] * B[7] - A[4] * B[9] + A[7] * B[6] - A[9] * B[4],
    A[4] * B[8] - A[5] * B[7] - A[7] * B[5] + A[8] * B[4],
    A[3] * B[8] - A[2] * B[9] + A[8] * B[3] - A[9] * B[2],
    A[1] * B[9] - A[3] * B[7] - A[7] * B[3] + A[9] * B[1],
    A[2] * B[7] - A[1] * B[8] + A[7] * B[2] - A[8] * B[1],
    A[2] * B[6] - A[3] * B[5] - A[5] * B[3] + A[6] * B[2],
    A[3] * B[4] - A[1] * B[6] + A[4] * B[3] - A[6] * B[1],
    A[1] * B[5] - A[2] * B[4] - A[4] * B[2] + A[5] * B[1])
end


@inline function Gridap.TensorValues.cross(H::TensorValue{9,9,T1}, A::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue{9,9}(A[9] * H[37] - A[8] * H[46] - A[6] * H[64] + A[5] * H[73],
    A[9] * H[38] - A[8] * H[47] - A[6] * H[65] + A[5] * H[74],
    A[9] * H[39] - A[8] * H[48] - A[6] * H[66] + A[5] * H[75],
    A[9] * H[40] - A[8] * H[49] - A[6] * H[67] + A[5] * H[76],
    A[9] * H[41] - A[8] * H[50] - A[6] * H[68] + A[5] * H[77],
    A[9] * H[42] - A[8] * H[51] - A[6] * H[69] + A[5] * H[78],
    A[9] * H[43] - A[8] * H[52] - A[6] * H[70] + A[5] * H[79],
    A[9] * H[44] - A[8] * H[53] - A[6] * H[71] + A[5] * H[80],
    A[9] * H[45] - A[8] * H[54] - A[6] * H[72] + A[5] * H[81],
    A[7] * H[46] - A[9] * H[28] + A[6] * H[55] - A[4] * H[73],
    A[7] * H[47] - A[9] * H[29] + A[6] * H[56] - A[4] * H[74],
    A[7] * H[48] - A[9] * H[30] + A[6] * H[57] - A[4] * H[75],
    A[7] * H[49] - A[9] * H[31] + A[6] * H[58] - A[4] * H[76],
    A[7] * H[50] - A[9] * H[32] + A[6] * H[59] - A[4] * H[77],
    A[7] * H[51] - A[9] * H[33] + A[6] * H[60] - A[4] * H[78],
    A[7] * H[52] - A[9] * H[34] + A[6] * H[61] - A[4] * H[79],
    A[7] * H[53] - A[9] * H[35] + A[6] * H[62] - A[4] * H[80],
    A[7] * H[54] - A[9] * H[36] + A[6] * H[63] - A[4] * H[81],
    A[8] * H[28] - A[7] * H[37] - A[5] * H[55] + A[4] * H[64],
    A[8] * H[29] - A[7] * H[38] - A[5] * H[56] + A[4] * H[65],
    A[8] * H[30] - A[7] * H[39] - A[5] * H[57] + A[4] * H[66],
    A[8] * H[31] - A[7] * H[40] - A[5] * H[58] + A[4] * H[67],
    A[8] * H[32] - A[7] * H[41] - A[5] * H[59] + A[4] * H[68],
    A[8] * H[33] - A[7] * H[42] - A[5] * H[60] + A[4] * H[69],
    A[8] * H[34] - A[7] * H[43] - A[5] * H[61] + A[4] * H[70],
    A[8] * H[35] - A[7] * H[44] - A[5] * H[62] + A[4] * H[71],
    A[8] * H[36] - A[7] * H[45] - A[5] * H[63] + A[4] * H[72],
    A[8] * H[19] - A[9] * H[10] + A[3] * H[64] - A[2] * H[73],
    A[8] * H[20] - A[9] * H[11] + A[3] * H[65] - A[2] * H[74],
    A[8] * H[21] - A[9] * H[12] + A[3] * H[66] - A[2] * H[75],
    A[8] * H[22] - A[9] * H[13] + A[3] * H[67] - A[2] * H[76],
    A[8] * H[23] - A[9] * H[14] + A[3] * H[68] - A[2] * H[77],
    A[8] * H[24] - A[9] * H[15] + A[3] * H[69] - A[2] * H[78],
    A[8] * H[25] - A[9] * H[16] + A[3] * H[70] - A[2] * H[79],
    A[8] * H[26] - A[9] * H[17] + A[3] * H[71] - A[2] * H[80],
    A[8] * H[27] - A[9] * H[18] + A[3] * H[72] - A[2] * H[81],
    A[9] * H[1] - A[7] * H[19] - A[3] * H[55] + A[1] * H[73],
    A[9] * H[2] - A[7] * H[20] - A[3] * H[56] + A[1] * H[74],
    A[9] * H[3] - A[7] * H[21] - A[3] * H[57] + A[1] * H[75],
    A[9] * H[4] - A[7] * H[22] - A[3] * H[58] + A[1] * H[76],
    A[9] * H[5] - A[7] * H[23] - A[3] * H[59] + A[1] * H[77],
    A[9] * H[6] - A[7] * H[24] - A[3] * H[60] + A[1] * H[78],
    A[9] * H[7] - A[7] * H[25] - A[3] * H[61] + A[1] * H[79],
    A[9] * H[8] - A[7] * H[26] - A[3] * H[62] + A[1] * H[80],
    A[9] * H[9] - A[7] * H[27] - A[3] * H[63] + A[1] * H[81],
    A[7] * H[10] - A[8] * H[1] + A[2] * H[55] - A[1] * H[64],
    A[7] * H[11] - A[8] * H[2] + A[2] * H[56] - A[1] * H[65],
    A[7] * H[12] - A[8] * H[3] + A[2] * H[57] - A[1] * H[66],
    A[7] * H[13] - A[8] * H[4] + A[2] * H[58] - A[1] * H[67],
    A[7] * H[14] - A[8] * H[5] + A[2] * H[59] - A[1] * H[68],
    A[7] * H[15] - A[8] * H[6] + A[2] * H[60] - A[1] * H[69],
    A[7] * H[16] - A[8] * H[7] + A[2] * H[61] - A[1] * H[70],
    A[7] * H[17] - A[8] * H[8] + A[2] * H[62] - A[1] * H[71],
    A[7] * H[18] - A[8] * H[9] + A[2] * H[63] - A[1] * H[72],
    A[6] * H[10] - A[5] * H[19] - A[3] * H[37] + A[2] * H[46],
    A[6] * H[11] - A[5] * H[20] - A[3] * H[38] + A[2] * H[47],
    A[6] * H[12] - A[5] * H[21] - A[3] * H[39] + A[2] * H[48],
    A[6] * H[13] - A[5] * H[22] - A[3] * H[40] + A[2] * H[49],
    A[6] * H[14] - A[5] * H[23] - A[3] * H[41] + A[2] * H[50],
    A[6] * H[15] - A[5] * H[24] - A[3] * H[42] + A[2] * H[51],
    A[6] * H[16] - A[5] * H[25] - A[3] * H[43] + A[2] * H[52],
    A[6] * H[17] - A[5] * H[26] - A[3] * H[44] + A[2] * H[53],
    A[6] * H[18] - A[5] * H[27] - A[3] * H[45] + A[2] * H[54],
    A[4] * H[19] - A[6] * H[1] + A[3] * H[28] - A[1] * H[46],
    A[4] * H[20] - A[6] * H[2] + A[3] * H[29] - A[1] * H[47],
    A[4] * H[21] - A[6] * H[3] + A[3] * H[30] - A[1] * H[48],
    A[4] * H[22] - A[6] * H[4] + A[3] * H[31] - A[1] * H[49],
    A[4] * H[23] - A[6] * H[5] + A[3] * H[32] - A[1] * H[50],
    A[4] * H[24] - A[6] * H[6] + A[3] * H[33] - A[1] * H[51],
    A[4] * H[25] - A[6] * H[7] + A[3] * H[34] - A[1] * H[52],
    A[4] * H[26] - A[6] * H[8] + A[3] * H[35] - A[1] * H[53],
    A[4] * H[27] - A[6] * H[9] + A[3] * H[36] - A[1] * H[54],
    A[5] * H[1] - A[4] * H[10] - A[2] * H[28] + A[1] * H[37],
    A[5] * H[2] - A[4] * H[11] - A[2] * H[29] + A[1] * H[38],
    A[5] * H[3] - A[4] * H[12] - A[2] * H[30] + A[1] * H[39],
    A[5] * H[4] - A[4] * H[13] - A[2] * H[31] + A[1] * H[40],
    A[5] * H[5] - A[4] * H[14] - A[2] * H[32] + A[1] * H[41],
    A[5] * H[6] - A[4] * H[15] - A[2] * H[33] + A[1] * H[42],
    A[5] * H[7] - A[4] * H[16] - A[2] * H[34] + A[1] * H[43],
    A[5] * H[8] - A[4] * H[17] - A[2] * H[35] + A[1] * H[44],
    A[5] * H[9] - A[4] * H[18] - A[2] * H[36] + A[1] * H[45])
end

@inline function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, H::TensorValue{9,9,T2}) where {T1,T2}

  TensorValue{9,9}(A[5] * H[9] - A[6] * H[8] - A[8] * H[6] + A[9] * H[5],
    A[6] * H[7] - A[4] * H[9] + A[7] * H[6] - A[9] * H[4],
    A[4] * H[8] - A[5] * H[7] - A[7] * H[5] + A[8] * H[4],
    A[3] * H[8] - A[2] * H[9] + A[8] * H[3] - A[9] * H[2],
    A[1] * H[9] - A[3] * H[7] - A[7] * H[3] + A[9] * H[1],
    A[2] * H[7] - A[1] * H[8] + A[7] * H[2] - A[8] * H[1],
    A[2] * H[6] - A[3] * H[5] - A[5] * H[3] + A[6] * H[2],
    A[3] * H[4] - A[1] * H[6] + A[4] * H[3] - A[6] * H[1],
    A[1] * H[5] - A[2] * H[4] - A[4] * H[2] + A[5] * H[1],
    A[5] * H[18] - A[6] * H[17] - A[8] * H[15] + A[9] * H[14],
    A[6] * H[16] - A[4] * H[18] + A[7] * H[15] - A[9] * H[13],
    A[4] * H[17] - A[5] * H[16] - A[7] * H[14] + A[8] * H[13],
    A[3] * H[17] - A[2] * H[18] + A[8] * H[12] - A[9] * H[11],
    A[1] * H[18] - A[3] * H[16] - A[7] * H[12] + A[9] * H[10],
    A[2] * H[16] - A[1] * H[17] + A[7] * H[11] - A[8] * H[10],
    A[2] * H[15] - A[3] * H[14] - A[5] * H[12] + A[6] * H[11],
    A[3] * H[13] - A[1] * H[15] + A[4] * H[12] - A[6] * H[10],
    A[1] * H[14] - A[2] * H[13] - A[4] * H[11] + A[5] * H[10],
    A[5] * H[27] - A[6] * H[26] - A[8] * H[24] + A[9] * H[23],
    A[6] * H[25] - A[4] * H[27] + A[7] * H[24] - A[9] * H[22],
    A[4] * H[26] - A[5] * H[25] - A[7] * H[23] + A[8] * H[22],
    A[3] * H[26] - A[2] * H[27] + A[8] * H[21] - A[9] * H[20],
    A[1] * H[27] - A[3] * H[25] - A[7] * H[21] + A[9] * H[19],
    A[2] * H[25] - A[1] * H[26] + A[7] * H[20] - A[8] * H[19],
    A[2] * H[24] - A[3] * H[23] - A[5] * H[21] + A[6] * H[20],
    A[3] * H[22] - A[1] * H[24] + A[4] * H[21] - A[6] * H[19],
    A[1] * H[23] - A[2] * H[22] - A[4] * H[20] + A[5] * H[19],
    A[5] * H[36] - A[6] * H[35] - A[8] * H[33] + A[9] * H[32],
    A[6] * H[34] - A[4] * H[36] + A[7] * H[33] - A[9] * H[31],
    A[4] * H[35] - A[5] * H[34] - A[7] * H[32] + A[8] * H[31],
    A[3] * H[35] - A[2] * H[36] + A[8] * H[30] - A[9] * H[29],
    A[1] * H[36] - A[3] * H[34] - A[7] * H[30] + A[9] * H[28],
    A[2] * H[34] - A[1] * H[35] + A[7] * H[29] - A[8] * H[28],
    A[2] * H[33] - A[3] * H[32] - A[5] * H[30] + A[6] * H[29],
    A[3] * H[31] - A[1] * H[33] + A[4] * H[30] - A[6] * H[28],
    A[1] * H[32] - A[2] * H[31] - A[4] * H[29] + A[5] * H[28],
    A[5] * H[45] - A[6] * H[44] - A[8] * H[42] + A[9] * H[41],
    A[6] * H[43] - A[4] * H[45] + A[7] * H[42] - A[9] * H[40],
    A[4] * H[44] - A[5] * H[43] - A[7] * H[41] + A[8] * H[40],
    A[3] * H[44] - A[2] * H[45] + A[8] * H[39] - A[9] * H[38],
    A[1] * H[45] - A[3] * H[43] - A[7] * H[39] + A[9] * H[37],
    A[2] * H[43] - A[1] * H[44] + A[7] * H[38] - A[8] * H[37],
    A[2] * H[42] - A[3] * H[41] - A[5] * H[39] + A[6] * H[38],
    A[3] * H[40] - A[1] * H[42] + A[4] * H[39] - A[6] * H[37],
    A[1] * H[41] - A[2] * H[40] - A[4] * H[38] + A[5] * H[37],
    A[5] * H[54] - A[6] * H[53] - A[8] * H[51] + A[9] * H[50],
    A[6] * H[52] - A[4] * H[54] + A[7] * H[51] - A[9] * H[49],
    A[4] * H[53] - A[5] * H[52] - A[7] * H[50] + A[8] * H[49],
    A[3] * H[53] - A[2] * H[54] + A[8] * H[48] - A[9] * H[47],
    A[1] * H[54] - A[3] * H[52] - A[7] * H[48] + A[9] * H[46],
    A[2] * H[52] - A[1] * H[53] + A[7] * H[47] - A[8] * H[46],
    A[2] * H[51] - A[3] * H[50] - A[5] * H[48] + A[6] * H[47],
    A[3] * H[49] - A[1] * H[51] + A[4] * H[48] - A[6] * H[46],
    A[1] * H[50] - A[2] * H[49] - A[4] * H[47] + A[5] * H[46],
    A[5] * H[63] - A[6] * H[62] - A[8] * H[60] + A[9] * H[59],
    A[6] * H[61] - A[4] * H[63] + A[7] * H[60] - A[9] * H[58],
    A[4] * H[62] - A[5] * H[61] - A[7] * H[59] + A[8] * H[58],
    A[3] * H[62] - A[2] * H[63] + A[8] * H[57] - A[9] * H[56],
    A[1] * H[63] - A[3] * H[61] - A[7] * H[57] + A[9] * H[55],
    A[2] * H[61] - A[1] * H[62] + A[7] * H[56] - A[8] * H[55],
    A[2] * H[60] - A[3] * H[59] - A[5] * H[57] + A[6] * H[56],
    A[3] * H[58] - A[1] * H[60] + A[4] * H[57] - A[6] * H[55],
    A[1] * H[59] - A[2] * H[58] - A[4] * H[56] + A[5] * H[55],
    A[5] * H[72] - A[6] * H[71] - A[8] * H[69] + A[9] * H[68],
    A[6] * H[70] - A[4] * H[72] + A[7] * H[69] - A[9] * H[67],
    A[4] * H[71] - A[5] * H[70] - A[7] * H[68] + A[8] * H[67],
    A[3] * H[71] - A[2] * H[72] + A[8] * H[66] - A[9] * H[65],
    A[1] * H[72] - A[3] * H[70] - A[7] * H[66] + A[9] * H[64],
    A[2] * H[70] - A[1] * H[71] + A[7] * H[65] - A[8] * H[64],
    A[2] * H[69] - A[3] * H[68] - A[5] * H[66] + A[6] * H[65],
    A[3] * H[67] - A[1] * H[69] + A[4] * H[66] - A[6] * H[64],
    A[1] * H[68] - A[2] * H[67] - A[4] * H[65] + A[5] * H[64],
    A[5] * H[81] - A[6] * H[80] - A[8] * H[78] + A[9] * H[77],
    A[6] * H[79] - A[4] * H[81] + A[7] * H[78] - A[9] * H[76],
    A[4] * H[80] - A[5] * H[79] - A[7] * H[77] + A[8] * H[76],
    A[3] * H[80] - A[2] * H[81] + A[8] * H[75] - A[9] * H[74],
    A[1] * H[81] - A[3] * H[79] - A[7] * H[75] + A[9] * H[73],
    A[2] * H[79] - A[1] * H[80] + A[7] * H[74] - A[8] * H[73],
    A[2] * H[78] - A[3] * H[77] - A[5] * H[75] + A[6] * H[74],
    A[3] * H[76] - A[1] * H[78] + A[4] * H[75] - A[6] * H[73],
    A[1] * H[77] - A[2] * H[76] - A[4] * H[74] + A[5] * H[73])
end

@inline function Gridap.TensorValues.cross(A::TensorValue{3,9,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue{3,9}(A[13] * B[9] - A[16] * B[8] - A[22] * B[6] + A[25] * B[5],
    A[14] * B[9] - A[17] * B[8] - A[23] * B[6] + A[26] * B[5],
    A[15] * B[9] - A[18] * B[8] - A[24] * B[6] + A[27] * B[5],
    A[16] * B[7] - A[10] * B[9] + A[19] * B[6] - A[25] * B[4],
    A[17] * B[7] - A[11] * B[9] + A[20] * B[6] - A[26] * B[4],
    A[18] * B[7] - A[12] * B[9] + A[21] * B[6] - A[27] * B[4],
    A[10] * B[8] - A[13] * B[7] - A[19] * B[5] + A[22] * B[4],
    A[11] * B[8] - A[14] * B[7] - A[20] * B[5] + A[23] * B[4],
    A[12] * B[8] - A[15] * B[7] - A[21] * B[5] + A[24] * B[4],
    A[7] * B[8] - A[4] * B[9] + A[22] * B[3] - A[25] * B[2],
    A[8] * B[8] - A[5] * B[9] + A[23] * B[3] - A[26] * B[2],
    A[9] * B[8] - A[6] * B[9] + A[24] * B[3] - A[27] * B[2],
    A[1] * B[9] - A[7] * B[7] - A[19] * B[3] + A[25] * B[1],
    A[2] * B[9] - A[8] * B[7] - A[20] * B[3] + A[26] * B[1],
    A[3] * B[9] - A[9] * B[7] - A[21] * B[3] + A[27] * B[1],
    A[4] * B[7] - A[1] * B[8] + A[19] * B[2] - A[22] * B[1],
    A[5] * B[7] - A[2] * B[8] + A[20] * B[2] - A[23] * B[1],
    A[6] * B[7] - A[3] * B[8] + A[21] * B[2] - A[24] * B[1],
    A[4] * B[6] - A[7] * B[5] - A[13] * B[3] + A[16] * B[2],
    A[5] * B[6] - A[8] * B[5] - A[14] * B[3] + A[17] * B[2],
    A[6] * B[6] - A[9] * B[5] - A[15] * B[3] + A[18] * B[2],
    A[7] * B[4] - A[1] * B[6] + A[10] * B[3] - A[16] * B[1],
    A[8] * B[4] - A[2] * B[6] + A[11] * B[3] - A[17] * B[1],
    A[9] * B[4] - A[3] * B[6] + A[12] * B[3] - A[18] * B[1],
    A[1] * B[5] - A[4] * B[4] - A[10] * B[2] + A[13] * B[1],
    A[2] * B[5] - A[5] * B[4] - A[11] * B[2] + A[14] * B[1],
    A[3] * B[5] - A[6] * B[4] - A[12] * B[2] + A[15] * B[1])
end

@inline function Gridap.TensorValues.cross(A::SMatrix, B::SMatrix)
  return get_array(TensorValue(A) × TensorValue(B))
end

@inline function Gridap.TensorValues.outer(A::SVector, B::SVector)
  return get_array(VectorValue(A) ⊗ VectorValue(B))
end


@inline @generated function ⊙₁₂₃₄³⁴(H::TensorValue{D²}, A::TensorValue{D}) where {D, D²}
  @assert D*D == D² "Fourth- and second-order tensors size mismatch"
  str = ""
  for j in 1:D
    for i in 1:D
      for l in 1:D
        for k in 1:D
          a = _flat_idx(i,j,k,l,D)
          str *= "+H[$a]*A[$k,$l]"
        end
      end
      str *= ","
    end
  end
  Meta.parse("TensorValue{$D}($str)")
end


@inline @generated function ⊙₁₂₃²³(H::TensorValue{D,D²}, A::TensorValue{D}) where {D, D²}
  @assert D*D == D² "Fourth- and second-order tensors size mismatch"
  str = ""
  for i in 1:D
    for k in 1:D
      for j in 1:D
        a = _flat_idx(j,k,D)
        str *= "+H[$i,$a]*A[$j,$k]"
      end
    end
    str *= ","
  end
  Meta.parse("VectorValue{$D}($str)")
end


@inline @generated function ⊙₁₂₃³(H::TensorValue{D,D²}, V::VectorValue{D}) where {D, D²}
  @assert D*D == D² "Fourth- and second-order tensors size mismatch"
  str = ""
  for j in 1:D
    for i in 1:D
      for k in 1:D
        a = _flat_idx(j,k,D)
        str *= "+H[$i,$a]*V[$k]"
      end
      str *= ","
    end
  end
  Meta.parse("TensorValue{$D}($str)")
end


Gridap.TensorValues.inner(H::TensorValue{4,4}, A::TensorValue{2,2}) = H ⊙₁₂₃₄³⁴ A
Gridap.TensorValues.inner(H::TensorValue{9,9}, A::TensorValue{3,3}) = H ⊙₁₂₃₄³⁴ A
Gridap.TensorValues.inner(H::TensorValue{2,4}, A::TensorValue{2,2}) = H ⊙₁₂₃²³ A
Gridap.TensorValues.inner(H::TensorValue{3,9}, A::TensorValue{3,3}) = H ⊙₁₂₃²³ A
Gridap.TensorValues.inner(H::TensorValue{2,4}, V::VectorValue{2}) = H ⊙₁₂₃³ V
Gridap.TensorValues.inner(H::TensorValue{3,9}, V::VectorValue{3}) = H ⊙₁₂₃³ V
Gridap.TensorValues.inner(V::VectorValue, H::TensorValue) = TensorValue(V.data) ⊙ H


"""
    contraction_IP_PJKL(A::TensorValue{D}, H::TensorValue{D*D})::TensorValue{D*D}

Performs a tensor contraction between a second-order tensor (of size `D × D`)
and a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation).
The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@inline @generated function contraction_IP_PJKL(A::TensorValue{D}, H::TensorValue{D²}) where {D, D²}
  @assert D*D == D² "Second and Fourth-order tensors size mismatch"
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          for p in 1:D
            a = _flat_idx(p,j,k,l,D)
            str *= "+A[$i,$p]*H[$a]"
          end
          str *= ","
        end
      end
    end
  end
  Meta.parse("TensorValue{D²}($str)")
end


"""
    contraction_IP_JPKL(A::TensorValue{D}, H::TensorValue{D*D})::TensorValue{D*D}

Performs a tensor contraction between a second-order tensor (of size `D × D`)
and a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation).
The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@inline @generated function contraction_IP_JPKL(A::TensorValue{D}, H::TensorValue{D²}) where {D, D²}
  @assert D*D == D² "Second and Fourth-order tensors size mismatch"
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          for p in 1:D
            a = _flat_idx(j,p,k,l,D)
            str *= "+A[$i,$p]*H[$a]"
          end
          str *= ","
        end
      end
    end
  end
  Meta.parse("TensorValue{D²}($str)")
end


"""
    contraction_IJKL_JL(A::TensorValue{D*D}, H::TensorValue{D})::TensorValue{D*D}

Performs a tensor contraction between a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation)
and a second-order tensor (of size `D × D`).
The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@inline @generated function contraction_IJKL_JL(H::TensorValue{D²}, A::TensorValue{D}) where {D, D²}
  @assert D*D == D² "Fourth- and Second-order tensors size mismatch"
  str = ""
  for i in 1:D
    for k in 1:D
      for j in 1:D
        for l in 1:D
          a = _flat_idx(i,j,k,l,D)
          str *= "+H[$a]*A[$j,$l]"
        end
      end
      str *= ","
    end
  end
  Meta.parse("TensorValue{D}($str)")
end

(⊗₁₂₃₄²⁴) = contraction_IJKL_JL


"""
    contraction_IJK_KLP(A::TensorValue{D,D*D}, B::TensorValue{D,D*D})::TensorValue{D*D,D*D}

Performs a tensor contraction between third-order tensors (represented as a `D × D²` matrix in flattened index notation).
The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@inline @generated function contraction_IJK_KLP(A::TensorValue{D,D²}, B::TensorValue{D,D²}) where {D, D²}
  @assert D*D == D² "Third-order tensor sizes mismatch"
  str = ""
  for p in 1:D
    for l in 1:D
      for j in 1:D
        for i in 1:D
          for k in 1:D
            a = _flat_idx(i,j,k,D)
            b = _flat_idx(k,l,p,D)
            str *= "+A[$a]*B[$b]"
          end
          str *= ","
        end
      end
    end
  end
  Meta.parse("TensorValue{$D²,$D²}($str)")
end

Gridap.TensorValues.dot(A::TensorValue{2,2}, B::TensorValue{4,4}) = contraction_IP_PJKL(A,B)
Gridap.TensorValues.dot(A::TensorValue{3,3}, B::TensorValue{9,9}) = contraction_IP_PJKL(A,B)
Gridap.TensorValues.dot(A::TensorValue{2,4}, B::TensorValue{2,4}) = contraction_IJK_KLP(A,B)
Gridap.TensorValues.dot(A::TensorValue{3,9}, B::TensorValue{3,9}) = contraction_IJK_KLP(A,B)


"""
    push_forward_C_to_F(F::TensorValue{D}, H::TensorValue{D²}) :: TensorValue

Assumming `C` is symmetric, compute directly `0.5 * DCDF' · H · DCDF` without
computing the 4th order tensor `DCDF`.
"""
@inline @generated function push_forward_C_to_F(F::TensorValue{D}, H::TensorValue{D²}) where {D, D²}
  @assert D*D == D² "Mismatch dimensions of F (D) and H (D²)."
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          term_str = ""
          for n in 1:D
            for m in 1:D
              a1 = _flat_idx(m, j, n, l, D)
              a2 = _flat_idx(m, j, l, n, D)
              a3 = _flat_idx(j, m, n, l, D)
              a4 = _flat_idx(j, m, l, n, D)
              term_str *= "+ 0.5 * F[$i,$m] * F[$k,$n] * (H[$a1] + H[$a2] + H[$a3] + H[$a4])"
            end
          end
          str *= "($term_str),"
        end
      end
    end
  end
  Meta.parse("TensorValue{D²}($str)")
end
