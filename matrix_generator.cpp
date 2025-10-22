// matrix_generator.cpp
#include "matrix_generator.h"
#include <type_traits>
#include <limits>
#include <random>
#include <algorithm>

namespace mg {

template<typename T>
Matrix<T> generate_dense_block(int rows, int cols, std::mt19937 &rng) {
    Matrix<T> block(rows, std::vector<T>(cols));
    if constexpr (std::is_integral_v<T>) {
        // avoid full int64 range for portability; clamp to -1000..1000 for ints
        std::uniform_int_distribution<long long> dist(-1000, 1000);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                block[i][j] = static_cast<T>(dist(rng));
    } else {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                block[i][j] = static_cast<T>(dist(rng));
    }
    return block;
}

template<typename T>
Matrix<T> generate_matrix(int num_rows,
                          int num_cols,
                          double sparsity,
                          const std::string &pattern,
                          int blocksize,
                          unsigned int seed) {
    Matrix<T> mat(num_rows, std::vector<T>(num_cols));
    std::mt19937 rng(seed);

    auto fill_block = [&](int bi, int bj) {
        int start_row = bi * blocksize;
        int start_col = bj * blocksize;
        int end_row = std::min(start_row + blocksize, num_rows);
        int end_col = std::min(start_col + blocksize, num_cols);
        auto block = generate_dense_block<T>(end_row - start_row, end_col - start_col, rng);
        for (int i = start_row; i < end_row; ++i)
            for (int j = start_col; j < end_col; ++j)
                mat[i][j] = block[i - start_row][j - start_col];
    };

    if (pattern == "random") {
        // Fill randomly with probability of being non-zero = 1 - sparsity
        std::bernoulli_distribution keep(1.0 - sparsity);
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                if (keep(rng)) {
                    if constexpr (std::is_integral_v<T>) {
                        std::uniform_int_distribution<int> d(-1000, 1000);
                        mat[i][j] = static_cast<T>(d(rng));
                    } else {
                        std::uniform_real_distribution<double> d(-1.0, 1.0);
                        mat[i][j] = static_cast<T>(d(rng));
                    }
                } else {
                    mat[i][j] = static_cast<T>(0);
                }
            }
        }

    } else if (pattern == "checkerboard") {
        int nbi = (num_rows + blocksize - 1) / blocksize;
        int nbj = (num_cols + blocksize - 1) / blocksize;
        for (int i = 0; i < nbi; ++i)
            for (int j = 0; j < nbj; ++j)
                if (((i % 2) ^ (j % 2)) == 0)
                    fill_block(i, j);

    } else if (pattern == "diagonal") {
        int minrc = std::min(num_rows, num_cols);
        for (int i = 0; i < minrc; ++i) {
            auto block = generate_dense_block<T>(1, 1, rng);
            mat[i][i] = block[0][0];
        }

    } else if (pattern == "blockdiagonal") {
        int n = std::min(num_rows, num_cols) / blocksize + 1;
        for (int i = 0; i < n; ++i) fill_block(i, i);

    } else if (pattern == "blockrandom") {
        int nbi = (num_rows + blocksize - 1) / blocksize;
        int nbj = (num_cols + blocksize - 1) / blocksize;
        std::bernoulli_distribution keep(1.0 - sparsity);
        for (int i = 0; i < nbi; ++i)
            for (int j = 0; j < nbj; ++j)
                if (keep(rng)) fill_block(i, j);

    } else {
        throw std::invalid_argument("Unknown pattern: " + pattern);
    }

    return mat;
}

// Explicit template instantiations for common types (to avoid linker issues)
template Matrix<float> generate_matrix<float>(int, int, double, const std::string&, int, unsigned int);
template Matrix<double> generate_matrix<double>(int, int, double, const std::string&, int, unsigned int);
template Matrix<int> generate_matrix<int>(int, int, double, const std::string&, int, unsigned int);
template Matrix<long long> generate_matrix<long long>(int, int, double, const std::string&, int, unsigned int);

template Matrix<float> generate_dense_block<float>(int, int, std::mt19937 &);
template Matrix<double> generate_dense_block<double>(int, int, std::mt19937 &);
template Matrix<int> generate_dense_block<int>(int, int, std::mt19937 &);
template Matrix<long long> generate_dense_block<long long>(int, int, std::mt19937 &);

} // namespace mg
