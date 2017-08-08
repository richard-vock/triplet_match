namespace triplet_match {

template <typename Iter>
inline
range<Iter>::range(const std::pair<Iter,Iter>& x) : std::pair<Iter,Iter>(x) {}

template <typename Iter>
inline
range<Iter>::~range() {}

template <typename Iter>
inline Iter
range<Iter>::begin() const {
    return this->first;
}

template <typename Iter>
inline Iter
range<Iter>::end() const {
    return this->second;
}

} // triplet_match
