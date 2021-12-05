#include "../common/error.cuh"
#include "../common/floats.hpp"
#include "../common/clock.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__constant__ unsigned FULL_MASK = 0xffffffff;
#define __gSize  10240
__device__ real static_y[__gSize];

__global__ void reduce_syncthreads(real *x, real *y, const int N);
__global__ void reduce_syncwarp(real *x, real *y, const int N);
__global__ void reduce_shfl_down(real *x, real *y, const int N);
__global__ void reduce_cp(real *x, real *y, const int N);
__global__ void reduce_cp_grid(const real *x, real *y, const int N);
real reduce_wrap(const real *x, const int N, const int gSize, const int bSize);
real reduce_wrap_static(const real *x, const int N, const int gSize, const int bSize);



int main()
{
    int N = 1e8;
    int M = N * sizeof(real);

    int bSize = 32;
    int gSize = (N + bSize - 1)/bSize;

    cout << FLOAT_PREC << endl;

    real *h_x, *h_x2, *h_y, *h_y2, *h_res;
    h_x = new real[N];
    h_x2 = new real[N];
    h_y = new real[gSize];
    h_y2 = new real[gSize];
    h_res = new real(0.0);
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = 1.23;
        h_x2[i] = 1.23;
    }
    real initRes = 0.0;
    for (int i = 0; i < gSize ; ++i)
    {
        h_y2[i] = 0.0;
    }

    cudaClockStart

    real *d_x, *d_y, *d_res;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, gSize*sizeof(real)));
    CHECK(cudaMalloc(&d_res, sizeof(real)));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));

    cudaClockCurr

    reduce_syncthreads<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, gSize*sizeof(real), cudaMemcpyDefault));
    real res = 0;
    for(int i = 0; i < gSize; ++i)
    {
        res += h_y[i];
    }
    cout << "reduce_syncthreads result: " << res << endl;
    cudaClockCurr
    
    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_syncwarp<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_syncwrap result: " << *h_res << endl;
    cudaClockCurr

    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_shfl_down<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_shfl_down result: " << *h_res << endl;
    cudaClockCurr

    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_cp<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_cp result: " << *h_res << endl;
    cudaClockCurr

    reduce_cp_grid<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_y, N);
    CHECK(cudaMemcpy(h_y, d_y, gSize*sizeof(real), cudaMemcpyDefault));
    res = 0.0;
    for(int i = 0; i < gSize; ++i)
    {
        res += h_y[i];
    }    
    cout << "reduce_cp_grid result: " << res << endl;
    cudaClockCurr  

    res =  reduce_wrap(d_x, N, 10240, 128);
    cout << "reduce_wrap result: " << res << endl;
    cudaClockCurr  

    res =  reduce_wrap_static(d_x, N, 10240, 128);
    cout << "reduce_wrap_static result: " << res << endl;
    cudaClockCurr         

    delete[] h_x;
    delete[] h_y;
    delete h_res;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_res));

    return 0;
}

__global__ void reduce_syncthreads(real *x, real *y, const int N)
{
    int tid = threadIdx.x;  // 线程块中线程在x方向的id。
    int ind = tid + blockIdx.x * blockDim.x; // 一维线程块中线程在GPU中的id。

    extern __shared__ real block_x[]; // 线程块共享内存。
    block_x[tid] = (ind < N)? x[ind] : 0;
    __syncthreads();  // 同步共享内存的拷贝操作，确保共享内存的数据已准备好。

    for(int offset = blockDim.x/2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            block_x[tid] += block_x[tid + offset];
        }
        __syncthreads(); // 同步线程块内线程。

    }    

    if (tid == 0)
    {
        y[blockIdx.x] = block_x[0];
    }

}

__global__ void reduce_syncwarp(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    // 线程束之间的二分求和。
    for (int offset = blockDim.x/2; offset >= 32; offset /=2)  
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads(); // 同步线程块内的线程。
    }

    // 线程束内的二分求和。
    for (int offset = 16; offset > 0; offset /=2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncwarp();  // 同步线程束内的线程。
    }

    if (tid == 0)
    {
        atomicAdd(y, block_arr[0]);  // 原子函数求和。
    }
}

__global__ void reduce_shfl_down(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }

        __syncthreads();
    }

    // 在线程寄存器上定义一个变量y。
    real curr_y = block_arr[tid];

    for (int offset = 16; offset > 0; offset /= 2)
    {
        // 通过线程束洗牌函数，从FULL_MASK出发，
        // 将高线程号（数组索引）中的curr_y值平移到低线程号，通过设置偏移值为 offset，等价实现了线程束内的折半归约。
        curr_y += __shfl_down_sync(FULL_MASK, curr_y, offset);
    }  

    if (tid == 0)
    {
        atomicAdd(y, curr_y);
    }  
}

__global__ void reduce_cp(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }

        __syncthreads();
    }

    real curr_y = block_arr[tid];   

    // 创建线程块片。
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());

    for (int offset = 16; offset > 0; offset /= 2)
    {
        // 线程块片的等价线程束内函数。
        curr_y += g32.shfl_down(curr_y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(y, curr_y);
    }
}

__global__ void reduce_cp_grid(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real block_arr[];

    real curr_y = 0.0;

    // 在归约前处理计算。
    // ???
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        curr_y += x[n];
    }

    block_arr[tid] = curr_y;
    __syncthreads();

    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }

        __syncthreads();
    }

    curr_y = block_arr[tid];
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // 线程块片的等价线程束内函数。
        curr_y += g32.shfl_down(curr_y, offset);
    }

    if (tid == 0)
    {
        y[bid] = curr_y;
    }    
}


real reduce_wrap(const real *x, const int N, const int gSize, const int bSize)
{
    const int ymem = gSize * sizeof(real);
    const int smem = bSize * sizeof(real);

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    // 使用两个核函数时，将数组 d_y 归约到最终结果的计算也是折半归约，  
    // 这比直接累加（使用原子函数或复制到主机再累加）要稳健（单精度下精度更高）。
    // 设备全局内存变量 d_x, d_y 对于每个线程块都是可见的，对于两个核函数是相同的。
    reduce_cp_grid<<<gSize, bSize, smem>>>(x, d_y, N);
    reduce_cp_grid<<<1, 1024, 1024*sizeof(real)>>>(d_y, d_y, gSize);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDefault));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

real reduce_wrap_static(const real *x, const int N, const int gSize, const int bSize)
{
    real *d_y;
    CHECK(cudaGetSymbolAddress((void**)&d_y, static_y));   // 获取设备静态全局内存或常量内存的地址（指针）。

    reduce_cp_grid<<<gSize, bSize, bSize * sizeof(real)>>>(x, d_y, N);
    reduce_cp_grid<<<1, 1024, 1024*sizeof(real)>>>(d_y, d_y, gSize);

    real h_y[1] = {0};  
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDefault));
    // CHECK(cudaFree(d_y));  // 全局内存由系统否则释放。

    return h_y[0];
}

