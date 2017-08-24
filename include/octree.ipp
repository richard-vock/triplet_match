#include <deque>

namespace triplet_match {

template <typename Point>
struct octree<Point>::impl {
    impl();

    void create(typename cloud_t::ConstPtr cloud, uint32_t max_depth,
                subdivision_criterion_t crit, std::optional<subset_t> subset);

    static std::pair<uint32_t, node_ptr_t>
    subdivide(typename pcl::PointCloud<Point>::ConstPtr cloud,
              const subset_t& subset, const bbox3_t& bbox,
              subdivision_criterion_t crit, uint32_t max_depth,
              uint32_t depth = 0);

    node_ptr_t root_;
    bbox3_t bbox_;
    uint32_t depth_;
};

template <typename Point>
class octree<Point>::depth_traverse
    : public ranges::view_facade<depth_traverse> {
    friend ranges::range_access;

public:
    depth_traverse() = default;
    explicit depth_traverse(node const& root) {
        stack_.push_front(&root);
    }

    node const&
    read() const {
        return *stack_.front();
    };

    bool
    equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void
    next() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_front(child.get());
            }
        }
    };

protected:
    std::deque<node const*> stack_;
};

template <typename Point>
class octree<Point>::breadth_traverse
    : public ranges::view_facade<breadth_traverse> {
    friend ranges::range_access;

public:
    breadth_traverse() = default;
    explicit breadth_traverse(node const& root) {
        stack_.push_back(&root);
    }

    node const&
    read() const {
        return *stack_.front();
    };

    bool
    equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void
    next() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_back(child.get());
            }
        }
    };

protected:
    std::deque<node const*> stack_;
};

template <typename Point>
class octree<Point>::leaf_traverse : public ranges::view_facade<leaf_traverse> {
    friend ranges::range_access;

public:
    leaf_traverse() = default;
    explicit leaf_traverse(node const& root) {
        stack_.push_front(&root);
        if (!std::holds_alternative<leaf_node>(this->read())) {
            next();
        }
    }

    node const&
    read() const {
        return *stack_.front();
    };

    bool
    equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void
    next() {
        do {
            next_();
        } while (!stack_.empty() &&
                 !std::holds_alternative<leaf_node>(this->read()));
    }

protected:
    void
    next_() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_front(child.get());
            }
        }
    };

protected:
    std::deque<node const*> stack_;
};

template <typename Point>
class octree<Point>::branch_traverse
    : public ranges::view_facade<branch_traverse> {
    friend ranges::range_access;

public:
    branch_traverse() = default;
    explicit branch_traverse(node const& root) {
        stack_.push_front(&root);
        if (!std::holds_alternative<leaf_node>(this->read())) {
            next();
        }
    }

    node const&
    read() const {
        return *stack_.front();
    };

    bool
    equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void
    next() {
        do {
            next_();
        } while (!stack_.empty() &&
                 !std::holds_alternative<branch_node>(this->read()));
    }

protected:
    void
    next_() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_front(child.get());
            }
        }
    };

protected:
    std::deque<node const*> stack_;
};

template <typename Point>
class octree<Point>::level_traverse
    : public ranges::view_facade<level_traverse> {
    friend ranges::range_access;

public:
    level_traverse() = default;
    explicit level_traverse(node const& root, uint8_t level) : level_(level) {
        stack_.push_back(&root);
        if (as_base_node(root)->depth != level) {
            next();
        }
    }

    node const&
    read() const {
        return *stack_.front();
    };

    bool
    equal(ranges::default_sentinel) const {
        return stack_.empty();
    }

    void
    next() {
        do {
            next_();
        } while (!stack_.empty() &&
                 as_base_node(this->read())->depth != level_);
    }

protected:
    void
    next_() {
        node const* crt = stack_.front();
        stack_.pop_front();
        if (auto bnode = std::get_if<branch_node>(crt)) {
            for (const auto& child : bnode->children) {
                stack_.push_back(child.get());
            }
        }
    };

protected:
    std::deque<node const*> stack_;
    uint8_t level_;
};

}  // namespace triplet_match
