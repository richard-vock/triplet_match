namespace triplet_match {

template <typename Point>
struct scene<Point>::impl {
    impl(typename cloud_t::ConstPtr cloud);

    ~impl();

    template <typename PointModel>
    mat4f_t find(model<PointModel>& m, std::function<float (const mat4f_t&)> score_func, const sample_parameters& params, subset_t subset);

    typename cloud_t::ConstPtr cloud() const {
        return cloud_;
    }

    typename cloud_t::ConstPtr cloud_;
    pcl::search::KdTree<Point> kdtree_;
};

template <typename Point>
template <typename PointModel, typename ScoreFunctor>
inline mat4f_t
scene<Point>::find(model<PointModel>& m, ScoreFunctor&& score_func, const sample_parameters& sample_params) {
    return impl_->find(m, [&] (const mat4f_t& t) { return score_func(t); }, sample_params, subset_t());
}

template <typename Point>
template <typename PointModel, typename ScoreFunctor>
inline mat4f_t
scene<Point>::find(model<PointModel>& m, ScoreFunctor&& score_func, const sample_parameters& sample_params, const subset_t& subset) {
    return impl_->find(m, [&] (const mat4f_t& t) { return score_func(t); }, sample_params, subset);
}

} // triplet_match
