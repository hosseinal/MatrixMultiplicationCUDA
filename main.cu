#include <iostream>
#include <chrono>

int N = 10000;

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;



//generate random data for the matrix
void generateMatrix(float *M){
    for(int i = 0 ; i < N*N ; i++){
        M[i] = rand()%100;
    }
}


//this function runs at GPU and it multiply 2 matrixes.
__global__ void matrixMul(float* d_A,float* d_B,float* d_C , int n){

    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < n; k++) {
        // Accumulate results for a single element
        d_C[rowIdx * n + colIdx] += d_A[rowIdx * n + k] * d_B[k * n + colIdx];
    }
}



int main() {
    // size of the matrixes
    size_t bytes = N * N * sizeof(float);
    //allocate data for the hosr
    float *h_A;
    float *h_B;
    float *h_C;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);

    //generate random matrix
    generateMatrix(h_A);
    generateMatrix(h_B);

    // allocate data at GPU ram
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A , bytes);
    cudaMalloc((void**)&d_B , bytes);
    cudaError e = cudaMalloc((void**)&d_C , bytes);
    if (e != cudaSuccess){
        printf("%s \n" , cudaGetErrorString(e));
    }

    //copy data from RAM to GPU RAM
    cudaMemcpy(d_A,h_A , bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B , bytes,cudaMemcpyHostToDevice);

    //define the grid size 
    dim3 gridSize;
    dim3 blockSize;
    gridSize.x=N/32 ; blockSize.x=32;
    gridSize.y=N/32 ; blockSize.y=32;


    //run the code and calculate the execution time
    auto t1 = high_resolution_clock::now();
    matrixMul<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C,d_C , bytes , cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();

    //calculate duration time of the serial code.
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);
    cout<<"gpu : "<<ms_int.count()<<endl;


    //free the allocated ram
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
