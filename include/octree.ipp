#include <deque>

namespace triplet_match {

template <typename Point>
class octree<Point>::depth_traverse
    : public ranges::view_facade<depth_traverse> {
    friend ranges::range_access;

public:
    depth_traverse() = default;
    explicit depth_traverse(typename octree<Point>::const_sptr_t octree)
        : octree_(octree) {
        stack_.push_front(octree->root_.get());
    }

    node const & read() const {
        return *stack_.front();
    };

    bool equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void next() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_front(child.get());
            }
        }
    };

protected:
    typename octree<Point>::const_sptr_t octree_;
    std::deque<node const*> stack_;
};

}  // namespace triplet_match
