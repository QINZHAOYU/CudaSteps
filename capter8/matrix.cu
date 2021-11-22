#include "../common/error.cuh"
#include "../common/floats.hpp"
#include <iomanip>
#include <string>
#include <fstream>

#define TILE_DIM  32

__constant__ int c_TILE_DIM = 32;  // 设备内存中线程块中矩阵维度（线程块大小，最大1024）。

void show(const real *matrix, const int N, std::string outfile, std::string title);
__global__ void transpose1(const real *src, real *dst, const int N);
__global__ void transpose2(const real *src, real *dst, const int N);
__global__ void transpose3(const real *src, real *dst, const int N);
__global__ void transpose4(const real *src, real *dst, const int N);



int main()
{
    // 由于显存 2 GB，float 为 4 字节，double 为 8 字节，所以在 transpose3, transpose4中：
    // float 矩阵维度不能超过 726；
    // double 矩阵维度不能超过 512；
    const int N = 500;  
    const int M = N * N * sizeof(real);

    int SIZE = 0;
    CHECK(cudaMemcpyFromSymbol(&SIZE, c_TILE_DIM, sizeof(int)));   

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
            h_matrix_org[i * N + j] = i*1.0e-2;
        }   
    }
    // show(h_matrix_org, N, "result.txt", "origin matrix");
    
    real *d_matrix_org, *d_matrix_res;
    CHECK(cudaMalloc(&d_matrix_org, M));
    CHECK(cudaMalloc(&d_matrix_res, M));
    CHECK(cudaMemcpy(d_matrix_org, h_matrix_org, M, cudaMemcpyDefault));    

    float elapsed_time = 0;
    float curr_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    // 矩阵转置（全局内存合并读取、非合并写入）。
    transpose1<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "result.txt", "transpose1");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose1 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（全局内存非合并读取、合并写入）。
    transpose2<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "matrix.txt", "transpose2");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose2 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（通过共享内存全局内存合并读写）。
    transpose3<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "result.txt", "transpose3");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose3 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（通过共享内存、bank处理，实现全局内存合并读写）。
    transpose4<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);     
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "result.txt", "transpose3");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));    
    printf("matrix transpose4 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;    

    delete[] h_matrix_res;
    delete[] h_matrix_org;
    CHECK(cudaFree(d_matrix_org));
    CHECK(cudaFree(d_matrix_res));

    return 0;
}


void show(const real *x, const int N, std::string outfile, std::string title)
{
    std::fstream out(outfile, std::ios::app);
    if (!out.is_open())
    {
        std::cerr << "invalid output file: " << outfile << endl;
        return;
    }

    out << "\n\n----------------" << title << endl;

    for (int i = 0; i < N; ++i)
    {
        out << endl;
        for (int j = 0; j < N; ++j)
        {
            out << std::setw(6) << x[i * N + j];
        }
    }
}

__global__ void transpose1(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * c_TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * c_TILE_DIM;

    if (nx < N && ny < N)
    {
        // 矩阵转置（合并读取、非合并写入）。
        dst[nx*N + ny] = src[ny*N + nx];
    }
}

__global__ void transpose2(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * c_TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * c_TILE_DIM;

    if (nx < N && ny < N)
    {
        // 矩阵转置（非合并读取、合并写入）。
        dst[ny*N + nx] = __ldg(&src[nx*N + ny]);   // 显示调用 `__ldg()` 函数缓存全局内存。 
    }
}

__global__ void transpose3(const real *src, real *dst, const int N)
{
    // 正常的做法中，全局内存的读写必有一个是非合并访问。
    // 现在通过将非合并访问转移到共享内存，利用共享内存的高性能（100倍全局内存），提高计算速度：  
    // 1. 首先将全局内存拷贝到线程块的共享内存；
    // 2. 然后从共享内存非合并访问，读取数据，合并写入全局内存。

    __shared__ real s_mat[TILE_DIM][TILE_DIM];  //二维静态共享内存，存储线程块内的一片矩阵。

    int bx = blockIdx.x * blockDim.x;  // 当前线程块首线程在网格中列索引。
    int by = blockIdx.y * blockDim.y;  // 当前线程块首线程在网格中行索引。

    int tx = threadIdx.x + bx;  // 当前线程在网格中列索引。
    int ty = threadIdx.y + by;  // 当前线程在网格中行索引。

    if (tx < N && ty < N)
    {
        // 全局内存合并访问，共享内存合并访问。
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx]; // 全局内存中二维矩阵一维存储。
    }
    __syncthreads();
    
    // 全局内存合并访问。
    if (tx < N && ty < N)
    {
        // 局部矩阵转置和全局内存合并写入。
        int x = by + threadIdx.x;
        int y = bx + threadIdx.y;
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y]; 
    }
}

__global__ void transpose4(const real *src, real *dst, const int N)
{
    // 通过修改数组行大小，错开数组元素在共享内存bank中的分布，
    // 避免线程束的 32路bank冲突。
    __shared__ real s_mat[TILE_DIM][TILE_DIM + 1];  

    int bx = blockIdx.x * blockDim.x; 
    int by = blockIdx.y * blockDim.y; 

    int tx = threadIdx.x + bx;  
    int ty = threadIdx.y + by;  

    if (tx < N && ty < N)
    {
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx]; 
    }
    __syncthreads();
    
    if (tx < N && ty < N)
    {
        int x = by + threadIdx.x;
        int y = bx + threadIdx.y;
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y]; 
    }
}