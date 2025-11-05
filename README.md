# CUDA Sparse Matrix Multiplication with Tensor Cores

A high-performance CUDA implementation of sparse matrix multiplication using NVIDIA Tensor Cores (WMMA API) and various optimization techniques. This project benchmarks different sparse matrix multiplication algorithms using Block Compressed Sparse Row (BCSR) format with configurable block sizes.

## 🚀 Features

- **Multiple CUDA Kernels**: Dense and sparse matrix multiplication implementations with various optimization strategies
- **Tensor Core Acceleration**: Leverages NVIDIA Tensor Cores via WMMA API for FP16 operations
- **Flexible BCSR Format**: Support for variable block sizes (16x16, 32x16, 64x16) with automatic validation
- **Multiple Sparse Patterns**: Random, checkerboard, diagonal, block-diagonal, large-random, medium-random, and specialized rectangular patterns
- **Advanced Kernel Variants**: v1, v2, v3 implementations with different memory hierarchies and thread organizations
- **Comprehensive Benchmarking**: Built-in NVBench integration with extensive performance analysis across all kernel variants
- **Matrix Generation**: Synthetic matrix generation with configurable sparsity levels and specialized block patterns
- **Performance Comparison**: Compare against cuBLAS GEMM (standard and Tensor Core enabled)

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
| `random` | Randomly distributed non-zeros | Configurable | General sparse matrices |
| `checkerboard` | Alternating dense/sparse blocks | Configurable | Structured sparsity |
| `diagonal` | Diagonal band structure | Configurable | Banded matrices |
| `blockdiagonal` | Block diagonal structure | Configurable | Domain decomposition |
| `blockrandom` | Random blocks with internal density | Configurable | Clustered sparsity |
| `largerandom` | Large rectangular blocks (legacy alias) | 64x16 | Wide sparse blocks |
| `mediumrandom` | Medium rectangular blocks | 32x16 | Medium-width sparse blocks |
| `pattern64by16` | Specialized 64x16 block pattern | 64x16 | Large rectangular sparsity |
| `pattern32by16` | Specialized 32x16 block pattern | 32x16 | Medium rectangular sparsity |

## 🏗️ Architecture

### Core Components

- **Matrix Classes**: Dense (`Matrix`), CSR (`CSRMatrix`), BCSR (`BCSRMatrix`) with configurable block sizes
- **CUDA Kernels**: Multiple optimized implementations in `src/cuda_kernels.cu` with different architectural approaches
- **Matrix Generator**: Synthetic matrix creation with various sparsity patterns and specialized rectangular block patterns
- **Benchmarking**: Comprehensive NVBench integration with extensive kernel coverage and automatic correctness validation
- **Block Size Support**: Runtime configurable block sizes (16x16, 32x16, 64x16) with validation

### Key Optimizations

1. **Tensor Core Utilization**: WMMA API for 16x16x16 half-precision operations across all block sizes
2. **Memory Coalescing**: Optimized memory access patterns with different warp organizations
3. **Shared Memory Staging**: Strategic use for A matrix data reuse (v3 kernels)
4. **Block-Level Parallelism**: Efficient BCSR block processing with size-specific optimizations
5. **Load Balancing**: Advanced work distribution with dual-warp designs (v2 kernels)
6. **Architectural Variants**: Multiple kernel designs for different performance characteristics

## 📈 Performance Kernels

### Dense Matrix Multiplication
- `denseMatrixMul`: Basic dense implementation (square and rectangular variants)
- `denseMatrixMulCo`: Coalesced memory access version
- `denseMatrixMulTensor`: Tensor Core accelerated version

### Sparse Matrix Multiplication (16x16 blocks)
- `sparseMatrixMulTensor`: Standard BCSR with 16x16 blocks
- `sparseMatrixMulTensor_v2`: Two-warp design with 32-column tiles (64 threads)
- `sparseMatrixMulTensor_v3`: Shared memory staging with cooperative loading
- `sparseMatrixMulTensor_v1_improved`: Enhanced shared memory with unified warp paths

### Block-Size Specific Kernels
- `sparseMatrixMulTensor32x16`: Optimized for 32x16 BCSR blocks (4 accumulators)
- `sparseMatrixMulTensor32x16_v2`: v2-style implementation for 32x16 blocks
- `sparseMatrixMulTensor64x16`: Optimized for 64x16 BCSR blocks (8 accumulators)
- `sparseMatrixMulTensor64x16_v2`: v2-style implementation for 64x16 blocks

### Legacy and Specialized Kernels
- `sparseMatrixMulTensorlargeRandom_old`: Original 64x16 implementation for comparison
- `sparseMatrixMulTensor_option2_ldmatrix_sm80`: PTX `ldmatrix` optimization for SM 8.0+

### Kernel Architecture Variants

**v1 Style**: Single warp (32 threads), 16x16 tiles
**v2 Style**: Dual warp (64 threads), each warp handles 16 columns, 32 columns total per block
**v3 Style**: Dual warp with shared memory staging for A matrix data

## 🧪 Testing and Validation

The project includes comprehensive correctness checking and validation:

```cpp
// CPU reference implementation for validation
void cpu_matmul_ref(const Matrix *A, const Matrix *B, std::vector<float> &C);

// Automatic correctness checking with configurable tolerance
const float eps = 1e-2f;

// Block size validation in BCSR constructor
if (blockSizeRow < 16 || blockSizeCol < 16 || 
    blockSizeRow % 16 != 0 || blockSizeCol % 16 != 0) {
    throw std::invalid_argument("Block sizes must be >= 16 and multiples of 16");
}
```

### Validation Features
- **Automatic Correctness Checking**: All benchmarks validate results against CPU reference
- **Block Size Validation**: Runtime validation of BCSR block size constraints
- **Pattern Validation**: Matrix generator validates pattern parameters
- **Memory Safety**: Comprehensive bounds checking and error handling
- **Cross-Kernel Validation**: Results consistency across different kernel implementations

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

- **Throughput**: Operations per second across all kernel variants
- **Memory Bandwidth**: Effective bandwidth utilization analysis
- **Tensor Core Efficiency**: WMMA instruction utilization and occupancy
- **Sparsity Impact**: Performance vs. sparsity level (50-90% sparse)
- **Block Size Effects**: Optimal block dimensions (16x16 vs 32x16 vs 64x16)
- **Kernel Architecture Comparison**: v1, v2, v3 architectural performance trade-offs
- **Pattern-Specific Performance**: Specialized patterns (largerandom, mediumrandom, etc.)

### Benchmark Coverage
- **Matrix Sizes**: 256x256 to 2048x1024 configurations  
- **Sparsity Levels**: 50%, 60%, 70%, 80%, 90%
- **Multiple Patterns**: All supported sparsity patterns
- **Column Variations**: N={32, 64, 128} for memory access pattern analysis
- **Cross-Validation**: All kernels validated against CPU reference

Results are exported to CSV format and can be visualized using the included plotting scripts in the `plots/` directory.

## 🔬 Research Applications

This codebase is designed for:

- **Sparse Linear Algebra Research**: Novel sparse matrix algorithms with configurable block structures
- **GPU Architecture Studies**: Comprehensive Tensor Core utilization analysis across different kernel designs
- **Performance Optimization**: Advanced CUDA kernel optimization techniques with architectural variants
- **Comparative Analysis**: Extensive benchmarking against cuBLAS GEMM implementations
- **Block Size Impact Studies**: Analysis of different BCSR block dimensions on performance
- **Memory Hierarchy Research**: Comparison of different shared memory usage patterns (v2/v3 kernels)
- **Sparsity Pattern Analysis**: Performance characterization across various sparse matrix patterns
- **WMMA API Optimization**: Deep dive into Tensor Core programming techniques

## 📝 Publications and References

If you use this code in research, please consider citing relevant GPU computing and sparse matrix literature. The implementation draws inspiration from:

- NVIDIA Tensor Core programming guides
- cuSPARSE optimization techniques
- WMMA API best practices

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional Sparse Formats**: ELL, COO, hybrid formats
- **More Sparsity Patterns**: Application-specific patterns, structured sparsity
- **Advanced GPU Support**: Hopper architecture optimizations, newer WMMA features  
- **Performance Optimizations**: PTX-level optimizations, async copy patterns
- **Block Size Extensions**: Support for larger block sizes (128x16, etc.)
- **Kernel Variants**: Additional architectural approaches (v4+ designs)
- **Documentation**: Enhanced algorithm descriptions, performance guides
- **Visualization**: Improved plotting and analysis tools

## 📄 License

This project is available under standard academic/research license terms. Please check with the repository owner for specific licensing information.

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

*Built with CUDA, optimized for performance, designed for research.*
