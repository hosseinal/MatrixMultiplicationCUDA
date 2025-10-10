/**
 * Compressed Sparce Row (CSR) matrix multiplication
 */

#include <iostream>

#include "Matrix.cuh"

// Matrix size (N*N)
#define N 3

struct CSRMatrix
{
        int *hdr;
        int *idx;
        float *data;

        CSRMatrix(float *M)
        {
                hdr = new int[N + 1];
                hdr[0] = 0;

                for (int i = 0; i < N; i++)
                {
                        hdr[i + 1] = hdr[i];
                        for (int j = 0; j < N; j++)
                        {
                                if (M[i * N + j])
                                {
                                        hdr[i + 1]++;
                                }
                        }
                }

                idx = new int[hdr[N]];
                data = new float[hdr[N]];

                for (int i = 0, j = 0; i < N * N; i++)
                {
                        if (M[i])
                        {
                                idx[j] = i % N;
                                data[j] = M[i];
                                j++;
                        }
                }
        }

        void print()
        {
                std::cout << "Header:\n";
                for (int i = 0; i < N + 1; i++)
                {
                        std::cout << hdr[i] << ' ';
                }
                std::cout << "\nIndexes:\n";
                for (int i = 0; i < hdr[N]; i++)
                {
                        std::cout << idx[i] << ' ';
                }
                std::cout << "\nData:\n";
                for (int i = 0; i < hdr[N]; i++)
                {
                        std::cout << data[i] << ' ';
                }
                std::cout << '\n';
        }
};

/**
 * generate a random sparce matrix with the specified sparcity percentage
 */
void generate_sparce_matrix(float *M, int sparcity_pctg)
{
        for (int i = 0; i < N * N; i++)
        {
                if ((rand() % 100) > sparcity_pctg)
                {
                        M[i] = rand() % 100;
                }
        }
}

/**
 * Multiply a CSR matrix x a dense matrix
 */
float *spmm1(const CSRMatrix *A, const float *B)
{
        float *C = new float[N * N];
        std::fill(C, C + N * N, 0.0);
        for (int i = 0; i < N; i++) // row
        {
                for (int j = 0; j < N; j++) // each col in B
                {
                        for (int k = A->hdr[i]; k < A->hdr[i + 1]; k++) // each col with non 0 in A
                        {
                                // I don't like this
                                C[i * N + j] += A->data[k] * B[A->idx[k] * N + j];
                        }
                }
        }
        return C;
}

/**
 * Multiply a CSR matrix x a dense matrix
 */
float *spmm2(const CSRMatrix *A, const float *B)
{
        float *C = new float[N * N];
        std::fill(C, C + N * N, 0.0);
        int k = 0;
        for (int i = 0; i < N; i++) // row
        {
                for (; k < A->hdr[i + 1]; k++) // each col with non 0 in A
                {
                        for (int j = 0; j < N; j++) // each col in B
                        {
                                // I don't like this
                                C[i * N + j] += A->data[k] * B[A->idx[k] * N + j];
                        }
                }
        }
        return C;
}

/**
 * Dense matrix multiplication
 */
float *mm(const float *A, const float *B)
{
        float *C = new float[N * N];
        std::fill(C, C + N * N, 0.0);
        for (int i = 0; i < N; i++)
        {
                for (int j = 0; j < N; j++)
                {
                        for (int k = 0; k < N; k++)
                        {
                                C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                }
        }
        return C;
}

int main(void)
{
        std::cout << "\n=== MATRIX A ===\n\n";
        float *M1 = new float[N * N];
        generate_sparce_matrix(M1, 80);

        for (int i = 0; i < N * N; i++)
        {
                std::cout << M1[i] << ' ';
                if ((i + 1) % N == 0)
                        std::cout << '\n';
        }

        CSRMatrix *A = new CSRMatrix(M1);
        A->print();

        std::cout << "\n=== MATRIX B ===\n\n";
        float *M2 = new float[N * N];
        generate_sparce_matrix(M2, 0);

        for (int i = 0; i < N * N; i++)
        {
                std::cout << M2[i] << ' ';
                if ((i + 1) % N == 0)
                        std::cout << '\n';
        }

        CSRMatrix *B = new CSRMatrix(M2);
        B->print();
        std::cout << "\n=== MATRIX C=AB using SpMM1 ===\n";
        float *C1 = spmm1(A, M2);
        for (int i = 0; i < N * N; i++)
        {
                std::cout << C1[i] << ' ';
                if ((i + 1) % N == 0)
                        std::cout << '\n';
        }

        std::cout << "\n=== MATRIX C=AB using SpMM2 ===\n";
        float *C2 = spmm2(A, M2);
        for (int i = 0; i < N * N; i++)
        {
                std::cout << C2[i] << ' ';
                if ((i + 1) % N == 0)
                        std::cout << '\n';
        }

        std::cout << "\n=== MATRIX C=AB using normal MM ===\n";
        float *C3 = mm(M1, M2);
        for (int i = 0; i < N * N; i++)
        {
                std::cout << C3[i] << ' ';
                if ((i + 1) % N == 0)
                        std::cout << '\n';
        }

        // Compare results
        for (int i = 0; i < N * N; i++)
        {
                if (C1[i] != C2[i] || C1[i] != C3[i]) {
                        std::cout << "THE RESULTS ARE NOT EQUAL\n";
                        return 1;
                }
        }
        std::cout << "Both results match\n";

        return 0;
}

// vim: ts=8 sw=8
