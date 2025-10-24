#ifndef MATRIX_CUH
#define MATRIX_CUH
#include <cuda_fp16.h>
#include <string>

using namespace std;

// Global project configuration constants
// Inline constexpr gives external linkage and avoids multiple-definition
// issues when included from many translation units.

class Matrix {
public:
    half *data = nullptr;
    int rows;
    int cols;
    int nonZeros;
    Matrix(int rows, int cols);
    explicit Matrix(const string &filename);
    ~Matrix();

    void print() const;
};


#endif //MATRIX_CUH