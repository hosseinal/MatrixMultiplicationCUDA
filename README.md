# CUDA Sparse Matrix Multiplication with Tensor Cores

A high-performance CUDA implementation of sparse matrix multiplication using NVIDIA Tensor Cores (WMMA API) and various optimization techniques. This project benchmarks different sparse matrix multiplication algorithms using Block Compressed Sparse Row (BCSR) format with configurable block sizes.

## 🚀 Features

- **Multiple CUDA Kernels**: Dense and sparse matrix multiplication implementations
- **Tensor Core Acceleration**: Leverages NVIDIA Tensor Cores via WMMA API for FP16 operations
- **Flexible BCSR Format**: Support for variable block sizes (16x16, 64x16, etc.)
- **Multiple Sparse Patterns**: Random, checkerboard, diagonal, block-diagonal, and large-random patterns
- **Comprehensive Benchmarking**: Built-in NVBench integration for performance analysis
- **Matrix Generation**: Synthetic matrix generation with configurable sparsity levels
- **Performance Comparison**: Compare against cuBLAS GEMM and cuSPARSE

## 📋 Requirements

- **CUDA Toolkit**: Version 11.0 or higher
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (Tensor Core support)
- **CMake**: Version 3.30 or higher
- **C++ Compiler**: C++14 compatible compiler
- **Dependencies**: cuBLAS, cuSPARSE, cuRAND

## 🛠️ Building

### Basic Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Advanced Build Options
```bash
# Specify target GPU architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

# Enable matrix generator test
cmake .. -DBUILD_MG_TEST=ON

# Custom NVBench location
cmake .. -DNVBENCH_INCLUDE_DIR=/path/to/nvbench/include
```

## 🔧 Usage

### Basic Matrix Multiplication
```bash
# Run main executable with default matrices
./matrix_multiplication

# Run with custom matrix paths
./matrix_multiplication path/to/matrixA.mat path/to/matrixB.mat
```

### Benchmarking with NVBench
```bash
# Run comprehensive benchmarks
./matrix_multiplication_nvbench

# Run specific benchmark patterns
./matrix_multiplication_nvbench --benchmark-filter="sparseMatrixMulTensor.*"
```

### Matrix Generation
```bash
# Generate test matrices with different patterns
./generate_all_matrices.sh

# Run tests on all generated matrices
./run_all_matrices.sh
```

## 📊 Supported Matrix Patterns

| Pattern | Description | Block Size | Use Case |
|---------|-------------|------------|----------|
| `random` | Randomly distributed non-zeros | 16x16 | General sparse matrices |
| `checkerboard` | Alternating dense/sparse blocks | 16x16 | Structured sparsity |
| `diagonal` | Diagonal band structure | 16x16 | Banded matrices |
| `blockdiagonal` | Block diagonal structure | 16x16 | Domain decomposition |
| `blockrandom` | Random blocks with internal density | 16x16 | Clustered sparsity |
| `largerandom` | Large rectangular blocks | 64x16 | Wide sparse blocks |

## 🏗️ Architecture

### Core Components

- **Matrix Classes**: Dense (`Matrix`), CSR (`CSRMatrix`), BCSR (`BCSRMatrix`)
- **CUDA Kernels**: Multiple optimized implementations in `src/cuda_kernels.cu`
- **Matrix Generator**: Synthetic matrix creation with various sparsity patterns
- **Benchmarking**: NVBench integration for performance measurement

### Key Optimizations

1. **Tensor Core Utilization**: WMMA API for 16x16x16 half-precision operations
2. **Memory Coalescing**: Optimized memory access patterns
3. **Shared Memory**: Strategic use for data reuse
4. **Block-Level Parallelism**: Efficient BCSR block processing
5. **Load Balancing**: Work distribution across thread blocks

## 📈 Performance Kernels

### Dense Matrix Multiplication
- `denseMatrixMul`: Basic dense implementation
- `denseMatrixMulTensor`: Tensor Core accelerated version

### Sparse Matrix Multiplication
- `sparseMatrixMulTensor`: Standard BCSR with 16x16 blocks
- `sparseMatrixMulTensor_v2`: Optimized with 32-column tiles
- `sparseMatrixMulTensor_v3`: Shared memory optimization
- `sparseMatrixMulTensorlargeRandom`: Specialized for 64x16 blocks
- `sparseMatrixMulTensorlargeRandom_old`: Original implementation for comparison

### Advanced Kernels
- `sparseMatrixMulTensor_option2_ldmatrix_sm80`: PTX `ldmatrix` optimization for SM 8.0+
- `sparseMatrixMulTensor_v1_improved`: Enhanced shared memory usage

## 🧪 Testing and Validation

The project includes comprehensive correctness checking:

```cpp
// CPU reference implementation for validation
void cpu_matmul_ref(const Matrix *A, const Matrix *B, std::vector<float> &C);

// Automatic correctness checking with configurable tolerance
const float eps = 1e-2f;
```

## 📁 Project Structure

```
MatrixMultiplicationCUDA/
├── src/
│   ├── cuda_kernels.cu      # CUDA kernel implementations
│   └── cuda_kernels.cuh     # Kernel declarations
├── matrices/                # Generated test matrices
├── plots/                   # Benchmarking and visualization
├── nvbench/                 # NVBench integration
├── main.cu                  # Main application
├── nvbench_main.cu         # Benchmarking harness
├── Matrix.{cu,cuh}         # Dense matrix class
├── CSRMatrix.{cu,cuh}      # CSR sparse matrix class
├── BCSRMatrix.{cu,cuh}     # BCSR sparse matrix class
├── matrix_generator.{cpp,h} # Matrix generation utilities
└── CMakeLists.txt          # Build configuration
```

## 🎯 Benchmarking Results

The project generates detailed performance reports comparing:

- **Throughput**: Operations per second
- **Memory Bandwidth**: Effective bandwidth utilization  
- **Tensor Core Efficiency**: WMMA instruction utilization
- **Sparsity Impact**: Performance vs. sparsity level
- **Block Size Effects**: Optimal block dimensions

Results are exported to CSV format and can be visualized using the included plotting scripts.

## 🔬 Research Applications

This codebase is designed for:

- **Sparse Linear Algebra Research**: Novel sparse matrix algorithms
- **GPU Architecture Studies**: Tensor Core utilization analysis
- **Performance Optimization**: CUDA kernel optimization techniques
- **Comparative Analysis**: Benchmarking against established libraries

## 📝 Publications and References

If you use this code in research, please consider citing relevant GPU computing and sparse matrix literature. The implementation draws inspiration from:

- NVIDIA Tensor Core programming guides
- cuSPARSE optimization techniques
- WMMA API best practices

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional sparse matrix formats (ELL, COO, etc.)
- More sparsity patterns
- Advanced GPU architectures support
- Performance optimizations
- Documentation improvements

## 📄 License

This project is available under standard academic/research license terms. Please check with the repository owner for specific licensing information.

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

*Built with CUDA, optimized for performance, designed for research.*
