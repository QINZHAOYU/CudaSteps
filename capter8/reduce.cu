#include "../common/error.cuh"
#include "../common/floats.hpp"
#include <chrono>

using namespace std::chrono;


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
    const int tid = threadIdx.x;
    real *curr_x = x + blockIdx.x * blockDim.x;  // 当前线程块中处理的内存首地址。

    for (int offset = blockDim.x >> 1; offset > 0; offset >>=1)  // 折半归约。
    {
        if (tid < offset)
        {
            curr_x[tid] += curr_x[tid + offset];
        }

        // 保证一个线程块中所有线程在执行该语句后面的语句之前，都完全执行了前面的语句。
        // 实现一个线程块中线程按照代码出现的顺序执行指令。
        // 但是不同线程块之间依然是独立、异步的。
        __syncthreads();
    }

    if (tid == 0)
    {
        y[blockIdx.x] = curr_x[0];
    }
}


int main()
{
    int N = 1e8;  // 单精度将发生 “大数吃小数” 的现象，导致结果完全错误；双精度没有问题。
    int M = N * sizeof(real);

    int block_size = 128;
    int grid_size = (N + block_size - 1)/block_size; 

    real *h_x = new real[N];
    real *h_y = new real[grid_size]; 
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = 1.23;
    }

    cout << FLOAT_PREC << endl;

    auto t1 = system_clock::now();

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

    reduce<<<grid_size, block_size>>>(d_x, d_y);
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());

    float elap_time=0, curr_time=0;
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("gpu reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;

    delete[] h_x;
    delete[] h_y;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    return 0;
}

