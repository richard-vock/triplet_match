__kernel void icp_projection(__global const float4* pnts, int n,
                                 __global const float4* image,
                                 int2 img_size,
                                 __global const float* mat_align,
                                 __global const float* mat_proj,
                                 __global const float* mat_norm,
                                 float max_corr_dist,
                                 __global float4* out_positions,
                                 __global int* model_indices,
                                 __global int* scene_indices) {
    // index corresponds to point in scene cloud
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    float4 pnt = pnts[index];

    // transform scene surface to model surface
    // for a good guess relevant scene point subset
    // now coincides with model surface
    float4 loc = mat44_multiply(pnt, mat_align);

    // project into projector space
    float4 uv = uv_project(loc, mat_proj);

    float2 uv_nrm = mat44_multiply(uv, mat_norm).xy;


    int2 px = convert_int2(uv_nrm * convert_float2(img_size));
    // image v coordinate of height is possible
    if (px.y == img_size.y) {
        px.y = img_size.y - 1;
    }

    model_indices[index] = -1;
    scene_indices[index] = -1;
    out_positions[index] = (float4)(-1.f, -1.f, -1.f, -1.f);

    // check for in-bound pixel coordinate
    if (px.x >= 0 && px.x < img_size.x && px.y >= 0 && px.y < img_size.y) {
        int idx = px.y*img_size.x + px.x;
        float dist = length(image[idx].xyz - loc.xyz);
        if (dist < max_corr_dist) {
            model_indices[index] = idx;
            scene_indices[index] = index;
            out_positions[index] = (float4)(uv_nrm, dist, 0.f);
        }
    }
}
