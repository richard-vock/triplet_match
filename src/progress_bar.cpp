#include <progress_bar>

#include <cstdio>
#include <iostream>

namespace triplet_match {

namespace detail {

constexpr char clear_line[] = "\33[2K\r";

} // detail

progress_bar::progress_bar(std::string prefix, int width) : prefix_(std::move(prefix)), width_(width) {
}

void progress_bar::poll(float progress) {
	if (progress >= 1.f) {
        fmt::print("{}Done\n", detail::clear_line);
        return;
    }
    std::string bar(static_cast<int>(progress * width_), '=');
    std::string format = "{}{}|{:-<" + std::to_string(width_) + "}| {:3.2f}";
    fmt::print(format.c_str(), detail::clear_line, prefix_, bar, progress*100.f);
    std::cout.flush();
	//char* sB = new char[width_+1]; sB[0] = sB[width_-1] = '|'; sB[width_] = '\0';
	//for (int i=1; i< width_-1; ++i) 
		//sB[i] = (i<=static_cast<int>(progress*(width_-2))?'=':' ');
	//if (!print_percent_) return std::string(sB);
	//char* s = new char[width_+6];
	//int pI = static_cast<int>(progress * 100.f);
	//sprintf(s, "%s %3d%%", sB, pI);
	//delete [] sB;
	//delete [] s;
	//return std::string(s);
}

void progress_bar::poll(unsigned int done, unsigned int todo) {
	if (!todo) {
        throw std::domain_error("todo parameter cannot be 0");
    }
	float progress = static_cast<float>(done) / todo;
	poll(progress > 1.f ? 1.f : progress);
}

void progress_bar::finish() {
	poll(1.f);
}

} // triplet_match
