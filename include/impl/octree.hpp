#include <octree>
#include <bitset>

namespace triplet_match {

namespace detail {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

typedef std::bitset<3> octant_mask;

uint8_t
get_octant(const vec3f_t center, const vec3f_t pos) {
    std::bitset<3> gt;
    for (size_t i = 0; i < 3; ++i) {
        gt.set(i, pos[i] > center[i]);
    }
    return static_cast<uint8_t>(gt.to_ulong());
}

bbox3_t
get_octant_bbox(uint8_t octant, const bbox3_t& bbox) {
    vec3f_t lower = bbox.min();
    vec3f_t upper = bbox.max();
    vec3f_t center = 0.5f * (lower + upper);
    std::bitset<3> mask(octant);
    return bbox3_t(
        vec3f_t(
            mask.test(0) ? center[0] : lower[0],
            mask.test(1) ? center[1] : lower[1],
            mask.test(2) ? center[2] : lower[2]
        ),
        vec3f_t(
            mask.test(0) ? upper[0] : center[0],
            mask.test(1) ? upper[1] : center[1],
            mask.test(2) ? upper[2] : center[2]
        )
    );
}

template <typename Point>
node_ptr_t
subdivide(typename pcl::PointCloud<Point>::ConstPtr cloud,
          const subset_t& subset,
          const bbox3_t& bbox,
          subdivision_criterion_t crit,
          uint32_t max_depth,
          uint32_t depth = 0) {
    node_ptr_t this_node(new node());
    if (depth < max_depth && std::visit(overloaded{
            [&](max_voxel_size crit)  { return (bbox.max() - bbox.min()).maxCoeff() > crit.value; },
            [&](min_voxel_size crit)  { return (bbox.max() - bbox.min()).minCoeff() < 2.f * crit.value; },
            [&](max_point_count crit) { return subset.size() > crit.value; }
        }, crit)) {
        vec3f_t center = 0.5f * (bbox.min() + bbox.max());
        std::array<subset_t, 8> child_sets;
        for (const auto& idx : subset) {
            child_sets[get_octant(center, cloud->points[idx].getVector3fMap())].push_back(idx);
        }
        *this_node = branch_node{};
        for (uint8_t i = 0; i < 8; ++i) {
            std::get<branch_node>(*this_node).children[i] = subdivide<Point>(cloud, child_sets[i], get_octant_bbox(i, bbox), crit, depth + 1);
        }
    } else {
        *this_node = leaf_node{subset};
    }

    return std::move(this_node);
}

}  // namespace detail

template <typename Point>
inline typename octree<Point>::sptr_t
octree<Point>::from_pointcloud(typename cloud_t::ConstPtr cloud, uint32_t max_depth,
                               subdivision_criterion_t crit,
                               std::optional<subset_t> subset) {
    sptr_t tree(new octree());

    subset_t indices;
    if (subset) {
        indices.resize(subset->size());
        std::copy(subset->begin(), subset->end(), indices.begin());
    } else {
        indices.resize(cloud->size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    // bounding box
    bbox3_t bbox;
    for (uint32_t idx : indices) {
        bbox.extend(cloud->points[idx].getVector3fMap());
    }
    tree->bbox_ = bbox;

    tree->root_ = detail::subdivide<Point>(cloud, indices, bbox, crit, max_depth);

    return tree;
}

template <typename Point>
inline octree<Point>::~octree() {}

template <typename Point>
inline octree<Point>::octree() {}

}  // namespace triplet_match
