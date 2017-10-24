float4 mat44_multiply(float4 pnt, __global const float* mat) {
    // mat considered to be in column-major order
    return (float4)(
        mat[0] * pnt.x + mat[4] * pnt.y + mat[ 8] * pnt.z + mat[12] * pnt.w,
        mat[1] * pnt.x + mat[5] * pnt.y + mat[ 9] * pnt.z + mat[13] * pnt.w,
        mat[2] * pnt.x + mat[6] * pnt.y + mat[10] * pnt.z + mat[14] * pnt.w,
        mat[3] * pnt.x + mat[7] * pnt.y + mat[11] * pnt.z + mat[15] * pnt.w
    );
}
