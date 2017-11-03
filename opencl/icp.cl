__kernel void icp_projection(__global const float4* pnts, int n,
                                 __global const float4* image,
                                 int2 img_size,
                                 int2 img_margin,
                                 __global const float* mat_align,
                                 __global const float* mat_uvw,
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

    float4 uv_nrm = mat44_multiply(mat44_multiply(uv, mat_norm), mat_uvw);

    // ext chosen such that uv * ext is inside the image without margin for uv in [0,1]
    float2 ext = convert_float2(img_size - 2 * img_margin - (int2)(1,1));
    int2 px = convert_int2(uv_nrm.xy * ext) + img_margin;
    // image v coordinate of height is possible
    if (px.y == img_size.y) {
        px.y = img_size.y - 1;
    }

    model_indices[index] = -1;
    scene_indices[index] = -1;
    out_positions[index] = (float4)(0.f, 0.f, 0.f, 0.f);

    // check for in-bound pixel coordinate
    if (px.x >= 0 && px.x < img_size.x && px.y >= 0 && px.y < img_size.y) {
        int idx = px.y*img_size.x + px.x;
        float dist = length(image[idx].xy - uv_nrm.xy);
        out_positions[index].w = dist;
        if (dist < max_corr_dist) {
            model_indices[index] = idx;
            scene_indices[index] = index;
            out_positions[index] = uv_nrm;
        }
    }
}

__kernel void icp_correlation(__global const float4* scene,
                              __global const float4* model,
                              __global const int* indices_scene,
                              __global const int* indices_model,
                              int n,
                              float4 centroid_scene,
                              float4 centroid_model,
                              __global float16* output) {
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    float4 spt = scene[indices_scene[index]] - centroid_scene;
    float4 mpt = model[indices_model[index]] - centroid_model;
    float norm = 1.f / (n-1);
    output[index].s0 = spt.x * mpt.x * norm;
    output[index].s1 = spt.y * mpt.x * norm;
    output[index].s2 = spt.z * mpt.x * norm;
    output[index].s3 = spt.x * mpt.y * norm;
    output[index].s4 = spt.y * mpt.y * norm;
    output[index].s5 = spt.z * mpt.y * norm;
    output[index].s6 = spt.x * mpt.z * norm;
    output[index].s7 = spt.y * mpt.z * norm;
    output[index].s8 = spt.z * mpt.z * norm;
    output[index].s9 = 0.f;
    output[index].sa = 0.f;
    output[index].sb = 0.f;
    output[index].sc = 0.f;
    output[index].sd = 0.f;
    output[index].se = 0.f;
    output[index].sf = 0.f;
}
