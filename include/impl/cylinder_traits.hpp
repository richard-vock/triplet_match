#include <pcl/sample_consensus/msac.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>

namespace triplet_match {

template <typename Point>
inline typename cylinder_traits<Point>::handle_t
cylinder_traits<Point>::init_from_model(typename cloud_t::ConstPtr cloud,
                                        const subset_t& subset) {
    typedef pcl::SampleConsensusModelCylinder<Point, Point> pcl_model_t;

    std::vector<int> indices(subset.begin(), subset.end());
    if (indices.empty()) {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }
    typename pcl_model_t::Ptr cyl(new pcl_model_t(cloud->native()));
    auto handle = std::make_shared<state_t>();
    handle->threshold = 5.f * cloud->resolution();
    cyl->setIndices(indices);
    cyl->setInputNormals(cloud->native());
    pcl::MEstimatorSampleConsensus<Point> msac(cyl, handle->threshold);
    msac.setProbability(0.99);
    msac.computeModel();
    Eigen::VectorXf coeffs;
    msac.getModelCoefficients(coeffs);

    vec3f_t axis = coeffs.segment<3>(3).normalized();
    vec3f_t ref = axis.unitOrthogonal();
    handle->origin = coeffs.head<3>();
    handle->radius = coeffs[6];
    handle->g2l = mat4f_t::Identity();
    handle->g2l.template block<1, 3>(0, 0) = ref.transpose();
    handle->g2l.template block<1, 3>(1, 0) = ref.cross(axis).transpose();
    handle->g2l.template block<1, 3>(2, 0) = axis.transpose();
    handle->g2l.template block<3, 1>(0, 3) =
        handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    // compute uvw bounds
    //for (auto idx : indices) {
        //std::optional<vec3f_t> uvw = project(handle, cloud->points[idx].getVector3fMap());
        //if (!uvw) continue;
        //handle->uvw_bounds.extend(*uvw);
    //}

    //pdebug("model radius: {}", handle->radius);
    //pdebug("model diameter: {}", cloud->diameter());

    return handle;
}

template <typename Point>
inline typename cylinder_traits<Point>::handle_t
cylinder_traits<Point>::init_from_samples(
    const_handle_t model_handle, const samples_t& samples) {
    // given normals 2 points suffice to derive a cylinder model
    Point sample1 = std::get<0>(samples);
    Point sample2 = std::get<1>(samples);

    vec3f_t p1 = sample1.getVector3fMap();
    vec3f_t p2 = sample2.getVector3fMap();
    vec3f_t n1 = sample1.getNormalVector3fMap();
    vec3f_t n2 = sample2.getNormalVector3fMap();

    vec3f_t w = n1 + p1 - p2;
    float a = n1.dot(n1);
    float b = n1.dot(n2);
    float c = n2.dot(n2);
    float d = n1.dot(w);
    float e = n2.dot(w);
    float denominator = a * c - b * b;
    float sc, tc;
    if (denominator < 1e-8) {
        sc = 0.0f;
        tc = (b > c ? d / b : e / c);
    } else {
        sc = (b * e - c * d) / denominator;
        tc = (a * e - b * d) / denominator;
    }

    auto handle = std::make_shared<state_t>();
    handle->threshold = model_handle->threshold;
    handle->origin = p1 + n1 + sc * n1;
    vec3f_t axis = (p2 + tc * n2 - handle->origin).normalized();
    vec3f_t ref = axis.unitOrthogonal();
    handle->radius = axis.cross(handle->origin - p1).norm();

    handle->g2l = mat4f_t::Identity();
    handle->g2l.template block<1, 3>(0, 0) = ref.transpose();
    handle->g2l.template block<1, 3>(1, 0) = ref.cross(axis).transpose();
    handle->g2l.template block<1, 3>(2, 0) = axis.transpose();
    handle->g2l.template block<3, 1>(0, 3) =
        handle->g2l.template topLeftCorner<3, 3>() * (-handle->origin);
    handle->l2g = handle->g2l.inverse();

    return handle;
}

template <typename Point>
inline std::optional<vec3f_t>
cylinder_traits<Point>::project(const_handle_t handle, const vec3f_t& xyz) {
    vec4f_t loc = handle->g2l * xyz.homogeneous();
    vec3f_t uvw;
    float height = loc.head(2).norm() - handle->radius;
    if (fabs(height) > handle->threshold) return std::nullopt;
    uvw[1] = loc[2];
    uvw[2] = height / handle->radius;
    uvw[0] = atan2(loc[1], loc[0]);
    if (uvw[0] < 0.f) uvw[0] += 2.0 * M_PI;
    uvw[0] *= handle->radius;

    return uvw;
}

template <typename Point>
inline vec3f_t
cylinder_traits<Point>::unproject(const_handle_t handle, const vec3f_t& uvw) {
    vec4f_t loc = vec4f_t::UnitW();
    loc[2] = uvw[1];

    float r = (uvw[2] + 1.f) * handle->radius;
    float angle = uvw[0] / handle->radius;

    loc[0] = r * std::cos(angle);
    loc[1] = r * std::sin(angle);

    vec3f_t xyz = (handle->l2g * loc).head(3);
    return xyz;
}

template <typename Point>
inline vec3f_t
cylinder_traits<Point>::tangent(const_handle_t handle, const Point& pnt) {
    vec4f_t loc = handle->g2l * pnt.getVector3fMap().homogeneous();
    vec3f_t loc_t = handle->g2l.template topLeftCorner<3, 3>() *
                    vec3f_t(pnt.data_c[1], pnt.data_c[2], pnt.data_c[3]);
    loc_t.normalize();

    float r = handle->radius;
    float n = loc.head(2).norm();
    float s = n * n;
    vec3f_t t;
    t[0] = loc_t[0] * (-r * loc[1] / s) + loc_t[1] * r * loc[0] / s;
    t[1] = loc_t[2];
    t[2] = loc_t[0] * loc[0] / n + loc_t[1] * loc[1] / n;
    // mat3f_t grad;
    // grad <<
    // radius * -loc[1] / s, radius * loc[0] / s, 0.f,
    // 0.f,                 0.f, 1.f,
    // loc[0] / n,          loc[1] / n, 0.f;
    // vec3f_t t = grad * loc_t;
    return t.normalized();
}

template <typename Point>
inline vec3f_t
cylinder_traits<Point>::normal(const_handle_t handle, const Point& pnt) {
    vec3f_t glb_t_1 = vec3f_t(pnt.data_c[1], pnt.data_c[2], pnt.data_c[3]);
    vec3f_t glb_t_2 = glb_t_1.cross(pnt.getNormalVector3fMap()).normalized();
    vec4f_t loc = handle->g2l * pnt.getVector3fMap().homogeneous();
    vec3f_t loc_t_1 = handle->g2l.template topLeftCorner<3, 3>() * glb_t_1;
    vec3f_t loc_t_2 = handle->g2l.template topLeftCorner<3, 3>() * glb_t_2;
    loc_t_1.normalize();
    loc_t_2.normalize();

    float r = handle->radius;
    float n = loc.head(2).norm();
    float s = n * n;
    vec3f_t t_1, t_2;
    t_1[0] = loc_t_1[0] * (-r * loc[1] / s) + loc_t_1[1] * r * loc[0] / s;
    t_1[1] = loc_t_1[2];
    t_1[2] = loc_t_1[0] * loc[0] / n + loc_t_1[1] * loc[1] / n;
    t_2[0] = loc_t_2[0] * (-r * loc[1] / s) + loc_t_2[1] * r * loc[0] / s;
    t_2[1] = loc_t_2[2];
    t_2[2] = loc_t_2[0] * loc[0] / n + loc_t_2[1] * loc[1] / n;
    return t_1.cross(t_2).normalized();
}

template <typename Point>
inline float
cylinder_traits<Point>::intrinsic_distance(const_handle_t handle,
                                           const vec3f_t& p0,
                                           const vec3f_t& p1) {
    float d_u = fabs(p1[0] - p0[0]);
    float d_v = fabs(p1[1] - p0[1]);
    d_u = std::min(d_u, static_cast<float>(2.0 * M_PI * handle->radius) - d_u);
    return sqrtf(d_u * d_u + d_v * d_v);
}

}  // namespace triplet_match
