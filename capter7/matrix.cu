
#include "common/error.cuh"
#include <iostream>

#ifdef USE_DP
    typedef double real;  
    const real EPSILON = 1.0e-15;
#else
    typedef float real;   
    const real EPSILON = 1.0e-6f;
#endif

// using namespace std;  // 不能使用std，会导致 `copy()` 不能使用（命名冲突）。


__constant__ int TILE_DIM = 32;  // 设备内存中线程块中矩阵维度（线程块大小，最大1024）。

__global__ void copy(const real *src, real *dst, const int N);
__global__ void transpose1(const real *src, real *dst, const int N);
__global__ void transpose2(const real *src, real *dst, const int N);


int main()
{
    const int N = 10000; 
    const int M = N * N * sizeof(real);

    int SIZE = 0;
    CHECK(cudaMemcpyFromSymbol(&SIZE, TILE_DIM, sizeof(int)));   

    const int grid_size_x = (N + SIZE - 1)/SIZE; // 获取网格大小。
    const int grid_size_y = grid_size_x;

    const dim3 block_size(SIZE, SIZE);
    const dim3 grid_size(grid_size_x, grid_size_y);

    real *h_matrix_org, *h_matrix_res;
    h_matrix_org = new real[N*N];
    h_matrix_res = new real[N*N];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_matrix_org[j] = i;
        }   
    }

    float elapsed_time = 0;
    float curr_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    real *d_matrix_org, *d_matrix_res;
    CHECK(cudaMalloc(&d_matrix_org, M));
    CHECK(cudaMalloc(&d_matrix_res, M));
    CHECK(cudaMemcpy(d_matrix_org, h_matrix_org, M, cudaMemcpyDefault));

    copy<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix copy time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    transpose1<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose1 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    transpose2<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose2 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    delete[] h_matrix_res;
    delete[] h_matrix_org;
    CHECK(cudaFree(d_matrix_org));
    CHECK(cudaFree(d_matrix_res));

    return 0;
}


__global__ void copy(const real *src, real *dst, const int N)
{
    // TILE_DIM = blockDim.x = blockDim.y
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x; // 矩阵列索引。
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y; // 矩阵行索引。
    const int index = ny * N + nx;

    if (nx >= N || ny >= N)
    {
        return;
    }

    dst[index] = src[index];  // 全局内存中数组也是线性存放的。
}

__global__ void transpose1(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;

    if (nx < N && ny < N)
    {
        // 矩阵转置（合并读取、非合并写入）。
        dst[nx*N + ny] = src[ny*N + nx];
    }
}

__global__ void transpose2(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;

    if (nx < N && ny < N)
    {
        // 矩阵转置（非合并读取、合并写入）。
        dst[ny*N + nx] = __ldg(&src[nx*N + ny]);   // 显示调用 `__ldg()` 函数缓存全局内存。 
    }
}