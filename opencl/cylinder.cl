float4 uv_project(float4 loc, __global const float* cyl2ncoord) {
    // transform points into normalized model
    // cylinder coordinates, i.e.
    // x-axis = angle reference direction
    // y-axis = angle reference co-direction
    // z-axis = cylinder axis
    // radius = 1
    float4 nc = mat44_multiply(loc, cyl2ncoord);

    // compute angle from xy-coordinates
    float u = atan2pi(nc.y, nc.x);
    // angle range correction...
    if (u < 0.f) u += 2.f;
    // ... and normalization such that u in [0,1]
    u /= 2.f;

    // second coordinate is the cylinder height coord
    float v = nc.z;

    // w = height deviation on cylinder surface = radius deviation
    float w = length(nc.xy) - 1.0;

    // return as homogeneous coordinate
    return (float4)(u, v, w, 1.f);
}
