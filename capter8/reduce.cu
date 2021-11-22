#include "../common/error.cuh"
#include "../common/floats.hpp"
#include <chrono>

using namespace std::chrono;

__constant__ int BLOCK_DIM = 128;


real reduce_cpu(const real *x, const int N)
{
    real sum = 0.0;
    for (int i = 0; i < N ; ++i)
    {
        sum += x[i];
    }

    return sum;
}

__global__ void reduce(real *x, real *y)
{
    // 这里执行迭代折半归约计算时，实际上的线程执行过程：
    // 1. 线程 0-127，offset = N/2, 迭代第一次；
    // 2. 线程 0-127，offset = N/4, 迭代第二次；
    // ...
    // 即，核函数中循环的每一轮都会被拆解、分配到线程块内的所有线程上执行，而不是一个  
    // 线程连续执行一次完整循环。
    const int tid = threadIdx.x;
    real *curr_x = x + blockIdx.x * blockDim.x;  // 当前线程块中处理的内存首地址。

    for (int offset = blockDim.x >> 1; offset > 0; offset >>=1) // 迭代折半归约。
    {
        // 由于条件筛选，实际导致每轮有效的线程数量减半，即 “线程束的分化”。
        // 要求数组大小为线程块大小的整数倍。
        if (tid < offset) 
        {
            // 核函数中代码是 “单指令多线程” ，代码真正的执行顺序与出现顺序可能不同。
            // 所以 线程 0、1、... 127之间实际上并行的。
            curr_x[tid] += curr_x[tid + offset];
        }

        // 保证一个线程块中所有线程在执行该语句后面的语句之前，都完全执行了前面的语句。
        // 实现一个线程块中所有线程按照代码出现的顺序执行指令，即线程1等待线程0，如此。
        // 但是不同线程块之间依然是独立、异步的。
        __syncthreads();
    }

    if (tid == 0)
    {
        // 通过线程块内同步，线程块 0 中的归约顺序：
        // 第一轮，curr_x[0] += curr_x[0+64], ... curr_x[63] += curr_x[63+64]；
        // 第二轮，curr_x[0] += curr_x[0+32], ... curr_x[31] += curr_x[31+32]；
        // 第三轮，curr_x[0] += curr_x[0+16], ... curr_x[15] += curr_x[15+16]； 
        // 第四轮，curr_x[0] += curr_x[0+ 8], ... curr_x[7 ] += curr_x[7 + 8]；  
        // 第五轮，curr_x[0] += curr_x[0+ 4], ... curr_x[3 ] += curr_x[3 + 4]； 
        // 第六轮，curr_x[0] += curr_x[0+ 2], curr_x[1 ] += curr_x[1 + 2]； 
        // 第七轮，curr_x[0] += curr_x[0+ 1]；           
        y[blockIdx.x] = curr_x[0];  
    }
}

__global__ void reduce_shared(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    __shared__ real s_x[128];  // 定义线程块静态共享内存变量。 
    s_x[tid] = (ind < N) ? x[ind] : 0.0;  // 拷贝全局内存变量到线程块内的共享内存数据副本。
    __syncthreads();   // 同步线程块的数据拷贝操作，保证各线程块中数据对于块内线程都准备好。

    for (int offset = blockDim.x>>1; offset > 0; offset>>=1)
    {
        if (ind < offset)
        {
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads(); // 线程块内线程同步。
    }

    if (tid == 0)
    {
        y[bid] = s_x[0]; // 保存各个线程块中共享内存的0元素到全局内存。
    }
}

__global__ void reduce_shared2(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real s_x[];  // 定义线程块动态共享内存变量，内存大小由主机调用核函数时定义。 
    s_x[tid] = (ind < N) ? x[ind] : 0.0;  // 拷贝全局内存变量到线程块内的共享内存数据副本。
    __syncthreads();   // 同步线程块的数据拷贝操作，保证各线程块中数据对于块内线程都准备好。

    for (int offset = blockDim.x>>1; offset > 0; offset>>=1)
    {
        if (ind < offset)
        {
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads(); // 线程块内线程同步。
    }

    if (tid == 0)
    {
        y[bid] = s_x[0]; // 保存各个线程块中共享内存的0元素到全局内存。
    }
}


int main()
{
    int N = 1e8;  // 单精度将发生 “大数吃小数” 的现象，导致结果完全错误；双精度没有问题。
    int M = N * sizeof(real);

    int block_size = 0;
    CHECK(cudaMemcpyFromSymbol(&block_size, BLOCK_DIM, sizeof(real)));
    int grid_size = (N + block_size - 1)/block_size; 

    real *h_x = new real[N];
    real *h_y = new real[grid_size]; 
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = 1.23;
    }

    cout << FLOAT_PREC << endl;

    auto t1 = system_clock::now();

    // cpu归约，单精度下计算错误，大数吃小数。
    cout << "cpu reduce:  " << reduce_cpu(h_x, N) << endl;

    auto t2 = system_clock::now();
    double time = duration<double, std::milli>(t2 - t1).count();
    cout << "cpu reduce time cost: " << time << " ms" << endl;

    real *d_x, *d_y;
    int size = grid_size*sizeof(real);
    CHECK(cudaMalloc(&d_x, M)); 
    CHECK(cudaMalloc(&d_y, size)); // 数据分片后个线程块的归约结果数组。
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault)); 
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    // gpu归约，单精度下也能控制误差，稳健性更强。
    reduce<<<grid_size, block_size>>>(d_x, d_y); 
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());

    float elap_time=0, curr_time=0;
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;

    // gpu归约，采用静态共享内存的加速。
    reduce_shared<<<grid_size, block_size>>>(d_x, d_y, N); 
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu shared reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu shared reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;    

    // gpu归约，采用动态共享内存的加速。
    // <<<grid_size, block_size, sharedMemSize>>>，第三个参数指定动态共享内存大小。
    int sharedMemSize = block_size * sizeof(real);  // 核函数中每个线程块的动态共享内存大小。
    reduce_shared2<<<grid_size, block_size, sharedMemSize>>>(d_x, d_y, N); 
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu shared2 reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu shared2 reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time; 

    delete[] h_x;
    delete[] h_y;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    return 0;
}

