#include <stdlib.h>
#include <stdio.h>
#include "add.cuh"
#include "clock.cuh"

const real a = 1.23;
const real b = 2.34;


void cuda_clock()
{
    const int N = 1e6;
    const int M = sizeof(real) * N;

    // cuda 计时。
    float elapsed_time = 0;
    float curr_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); // 创建cuda 事件对象。
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));  // 记录代表开始的事件。
    cudaEventQuery(start);  // 强制刷新 cuda 执行流。

    // --------------------------------------------------
    real *h_x, *h_y, *h_z;
    h_x = new real[N];
    h_y = new real[N];
    h_z = new real[N];
    if (!h_x || !h_y || !h_z)
    {
        printf("host memory malloc failed!\n");
        return;
    }
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;
        h_y[i] = b;
    }
    // --------------------------------------------------

    // 主机申请及初始化内存的耗时。
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); // 强制同步，让主机等待cuda事件执行完毕。
    CHECK(cudaEventElapsedTime(&curr_time, start, stop)); // 计算 start 和stop间的时间差（ms）。
    printf("host memory malloc and copy: %f ms.\n", curr_time - elapsed_time);    
    elapsed_time = curr_time;


    // --------------------------------------------------
    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyDefault));
    // --------------------------------------------------

    // 设备内存申请和拷贝耗时。
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); 
    CHECK(cudaEventElapsedTime(&curr_time, start, stop)); 
    printf("device memory malloc: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // --------------------------------------------------
    const int block_size = 128;
    const int grid_size = N/block_size + 1;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // --------------------------------------------------

    // 核函数运行耗时。
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); 
    CHECK(cudaEventElapsedTime(&curr_time, start, stop)); 
    printf("kernel function : %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;   

    // --------------------------------------------------
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault)); 
    check(h_z, N);
    // --------------------------------------------------

    // 数据拷贝耗时。
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); 
    CHECK(cudaEventElapsedTime(&curr_time, start, stop)); 
    printf("copy from device to host: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;  

    if (h_x) delete[] h_x;
    if (h_y) delete[] h_y;
    if (h_z) delete[] h_z;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
}