// matrix_generator.h
// Simple C++ port of matrix_generator.py functionality.
// Provides templated functions to generate 2D matrices (std::vector of vectors).
#pragma once

#include <vector>
#include <string>
#include <random>

namespace mg {

template<typename T>
using Matrix = std::vector<std::vector<T>>;

// Generate a dense block of shape (rows x cols) with random values of type T.
// For integer types, values are drawn from the full range of T (clamped to reasonable bounds).
// For floating point types, values are drawn uniformly in [-1, 1].
template<typename T>
Matrix<T> generate_dense_block(int rows, int cols, std::mt19937 &rng);

// Generate a full matrix with the given pattern.
// sparsity: fraction of zeros in the matrix (0.0..1.0). Default 0.7 like the python script.
// pattern: "random", "checkerboard", "diagonal", "blockdiagonal", "blockrandom"
// blocksize: used by block-based patterns
// seed: RNG seed
template<typename T>
Matrix<T> generate_matrix(int num_rows,
                          int num_cols,
                          double sparsity = 0.7,
                          const std::string &pattern = "random",
                          int blocksize = 16,
                          unsigned int seed = 123u);

} // namespace mg
