#include "Matrix.cuh"

#include <fstream>
#include <iostream>

Matrix::Matrix(const int rows, const int cols) {
    this->rows = rows;
    this->cols = cols;
    this->nonZeros = 0;
    this->data = static_cast<half *>(malloc((rows * cols) * sizeof(half)));
    fill_n(this->data, rows * cols, 0);
}

Matrix::Matrix(const string &filename) {
    this->rows = this->cols = this->nonZeros = 0;

    float v;
    freopen(filename.c_str(), "r", stdin);

    std::cin >> rows >> cols;
    this->data = static_cast<half *>(malloc((rows * cols) * sizeof(half)));

    for (int i = 0; i < rows * cols; i++) {
        std::cin >> v;
        if (v != 0.0f) this->nonZeros++;
        this->data[i] = __float2half(v);
    }
}

Matrix::~Matrix() {
    free(this->data);
}

void Matrix::print() const {
    std::cout << rows << ' ' << cols << "\n===\n";
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j++) {
            cout << __half2float(data[i * cols + j]) << " ";
        }
        cout << "\n";
    }
};

