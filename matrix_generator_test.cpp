#include <iostream>
#include "matrix_generator.h"

int main() {
    using namespace mg;
    auto m = generate_matrix<float>(8, 8, 0.7, "checkerboard", 2, 42);
    for (size_t i = 0; i < m.size(); ++i) {
        for (size_t j = 0; j < m[i].size(); ++j) {
            std::cout << m[i][j] << ' ';
        }
        std::cout << '\n';
    }
    return 0;
}
