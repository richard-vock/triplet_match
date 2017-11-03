namespace triplet_match {

template <typename Point>
inline
cylinder<Point>::cylinder(float threshold) : threshold_(threshold) {
}

template <typename Point>
inline
cylinder<Point>::~cylinder() {
}

template <typename Point>
inline void
cylinder<Point>::fit(typename cloud_t::ConstPtr cloud, const subset_t& subset) {
    std::vector<int> indices(subset.begin(), subset.end());
    if (indices.empty()) {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }
    typename pcl_model_t::Ptr cyl(new pcl_model_t(cloud));
    cyl->setIndices(indices);
    cyl->setInputNormals(cloud);
    pcl::MEstimatorSampleConsensus<Point> msac(cyl, threshold_);
    msac.setProbability(0.99);
    msac.computeModel();
    Eigen::VectorXf coeffs;
    msac.getModelCoefficients(coeffs);

    origin_ = coeffs.head<3>();
    axis_ = coeffs.segment<3>(3).normalized();
    radius_ = coeffs[6];

    vec3f_t ref = axis_.unitOrthogonal();

    mat3f_t base;
    base <<
        ref.transpose(),
        ref.cross(axis_).transpose(),
        axis_.transpose();

    //float rinv = 1.f / radius_;

    //mat3f_t scale = vec3f_t(rinv, rinv, rinv).asDiagonal();
    proj_ = mat4f_t::Identity();
    //proj_.topLeftCorner<3, 3>() = scale * base;
    proj_.topLeftCorner<3, 3>() = base;
    proj_.block<3,1>(0,3) = proj_.topLeftCorner<3, 3>() * (-origin_);
    inv_ = proj_.inverse();
}

template <typename Point>
inline void
cylinder<Point>::fit(const Point& sample1, const Point& sample2) {
    vec3f_t p1 = sample1.getVector3fMap();
    vec3f_t p2 = sample2.getVector3fMap();
    vec3f_t n1 = sample1.getNormalVector3fMap();
    vec3f_t n2 = sample2.getNormalVector3fMap();

    vec3f_t w = n1 + p1 - p2;
    float a = n1.dot (n1);
    float b = n1.dot (n2);
    float c = n2.dot (n2);
    float d = n1.dot (w);
    float e = n2.dot (w);
    float denominator = a*c - b*b;
    float sc, tc;
    if (denominator < 1e-8) {
        sc = 0.0f;
        tc = (b > c ? d / b : e / c);
    } else {
        sc = (b*e - c*d) / denominator;
        tc = (a*e - b*d) / denominator;
    }
    origin_  = p1 + n1 + sc * n1;
    axis_ = p2 + tc * n2 - origin_;
    axis_.normalize ();
    radius_ = axis_.cross(origin_ - p1).norm();

    vec3f_t ref = axis_.unitOrthogonal();

    mat3f_t base;
    base <<
        ref.transpose(),
        ref.cross(axis_).transpose(),
        axis_.transpose();

    //float rinv = 1.f / radius_;

    //mat3f_t scale = vec3f_t::Constant(1.f / radius_).asDiagonal();
    proj_ = mat4f_t::Identity();
    //proj_.topLeftCorner<3, 3>() = scale * base;
    proj_.topLeftCorner<3, 3>() = base;
    proj_.block<3,1>(0,3) = proj_.topLeftCorner<3, 3>() * (-origin_);
    inv_ = proj_.inverse();
}

template <typename Point>
inline mat4f_t
cylinder<Point>::transform_to(const cylinder<Point>& other, const vec3f_t& ref1, const vec3f_t& ref2) const {
    vec3f_t r1 = ref1 - origin_;
    vec3f_t r2 = ref2 - other.origin_;
    float l1 = r1.dot(axis_);
    float l2 = r2.dot(other.axis_);
    vec3f_t o1 = origin_ + l1 * axis_;
    vec3f_t o2 = other.origin_ + l2 * other.axis_;
    r1 -= l1 * axis_;
    r2 -= l2 * other.axis_;
    r1.normalize();
    r2.normalize();

    mat4f_t t = mat4f_t::Identity();
    t.block<3,1>(0,3) = -o1;

    quatf_t q1, q2;
    q1.setFromTwoVectors(axis_, other.axis_);
    r1 = q1._transformVector(r1);
    q2.setFromTwoVectors(r1, r2);
    q1 = q2*q1;

    mat4f_t rot = mat4f_t::Identity();
    rot.topLeftCorner<3,3>() = q1.toRotationMatrix();
    t = rot * t;
    t.block<3,1>(0,3) += o2;

    return t;
}

template <typename Point>
inline vec3f_t
cylinder<Point>::project(const vec3f_t& pos) const {
    vec3f_t local =  (proj_ * pos.homogeneous()).head(3);
    float u = atan2(local[1], local[0]);
    if (u < 0.f) {
        u += static_cast<float>(2.0 * M_PI);
    }
    u /= static_cast<float>(2.0 * M_PI);
    float v = local[2];
    float w = local.head(2).norm() - 1.f;
    return vec3f_t(u, v, w);
}

template <typename Point>
inline vec3f_t
cylinder<Point>::unproject(const vec2f_t& uv) const {
    vec4f_t local;
    float u = uv[0] * static_cast<float>(2.0 * M_PI);
    local[0] = std::cos(u);
    local[1] = std::sin(u);
    local[2] = uv[1];
    local[3] = 1.f;
    return (inv_ * local).head(3);
}

template <typename Point>
inline vec3f_t
cylinder<Point>::unproject(const vec3f_t& pos) const {
    vec4f_t local;
    float u = pos[0] * static_cast<float>(2.0 * M_PI);
    local[0] = (1.f + pos[2]) * std::cos(u);
    local[1] = (1.f + pos[2]) * std::sin(u);
    local[2] = pos[1];
    local[3] = 1.f;
    return (inv_ * local).head(3);
}

template <typename Point>
inline vec2f_t
cylinder<Point>::intrinsic_difference(const vec3f_t& pos0, const vec3f_t& pos1, bool debug) const {
    vec3f_t l0 =  (proj_ * pos0.homogeneous()).head(3);
    vec3f_t l1 =  (proj_ * pos1.homogeneous()).head(3);
    float v = std::abs(l1[2] - l0[2]);
    if (debug) pdebug("   l0: {},   l1: {}", l0.transpose(), l1.transpose());
    if (debug) pdebug("   v: {}", v);
    l0[2] = 0.f;
    l1[2] = 0.f;
    l0.normalize();
    l1.normalize();
    float u = radius_ * std::acos(l0.dot(l1)) / M_PI;
    return vec2f_t(u, v);
}

template <typename Point>
inline vec3f_t&
cylinder<Point>::origin() {
    return origin_;
}

template <typename Point>
inline const vec3f_t&
cylinder<Point>::origin() const {
    return origin_;
}

template <typename Point>
inline vec3f_t&
cylinder<Point>::axis() {
    return axis_;
}

template <typename Point>
inline const vec3f_t&
cylinder<Point>::axis() const {
    return axis_;
}

template <typename Point>
inline float&
cylinder<Point>::radius() {
    return radius_;
}

template <typename Point>
inline const float&
cylinder<Point>::radius() const {
    return radius_;
}

template <typename Point>
inline mat4f_t&
cylinder<Point>::projection_matrix() {
    return proj_;
}

template <typename Point>
inline const mat4f_t&
cylinder<Point>::projection_matrix() const {
    return proj_;
}

template <typename Point>
inline std::string
cylinder<Point>::opencl_path() {
    return "cylinder.cl";
}

} // triplet_match
