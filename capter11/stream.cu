#include "../common/error.cuh"
#include "../common/floats.hpp"
#include <math.h>
#include <stdio.h>


const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 30;
const int N2 = N1 * MAX_NUM_STREAMS;
const int M2 = sizeof(real) * N2;
cudaStream_t streams[MAX_NUM_STREAMS];  // cuda流数组，全局变量由系统负责销毁。


const int N = 100000000;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N - 1) / block_size + 1;


void timing(const real *h_x, const real *h_y, real *h_z,
    const real *d_x, const real *d_y, real *d_z,
    const int ratio, bool overlap);
void timing(const real *d_x, const real *d_y, real *d_z, 
    const int num);
void timing(const real *h_x, const real *h_y, real *h_z,
    real *d_x, real *d_y, real *d_z,
    const int num
);

int main(void)
{
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));


    // host and kernal overlap.
    printf("Without CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
    printf("With CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

    printf("Without CPU-GPU overlap (ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, false);
    printf("With CPU-GPU overlap (ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, true);

    printf("Without CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, false);
    printf("With CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, true);


    // kernal and kernal overlap.
    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        // 创建cuda流。
        CHECK(cudaStreamCreate(&(streams[n])));
    }

    for (int num = 1; num <= MAX_NUM_STREAMS; ++num)
    {
        timing(d_x, d_y, d_z, num);
    }

    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        // 销毁cuda流。
        CHECK(cudaStreamDestroy(streams[n]));
    }


    // kernal and data transfering overlap.
    real *h_x2, *h_y2, *h_z2;
    CHECK(cudaMallocHost(&h_x2, M));
    CHECK(cudaMallocHost(&h_y2, M));
    CHECK(cudaMallocHost(&h_z2, M));
    for (int n = 0; n < N; ++n)
    {
        h_x2[n] = 1.23;
        h_y2[n] = 2.34;
    }

    for (int i = 0; i < MAX_NUM_STREAMS; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    for (int num = 1; num <= MAX_NUM_STREAMS; num *= 2)
    {
        timing(h_x2, h_y2, h_z2, d_x, d_y, d_z, num);
    }

    for (int i = 0 ; i < MAX_NUM_STREAMS; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaFreeHost(h_x2));
    CHECK(cudaFreeHost(h_y2));
    CHECK(cudaFreeHost(h_z2));

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0;
}

void cpu_sum(const real *x, const real *y, real *z, const int N_host)
{
    for (int n = 0; n < N_host; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void __global__ gpu_sum(const real *x, const real *y, real *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void timing
(
    const real *h_x, const real *h_y, real *h_z,
    const real *d_x, const real *d_y, real *d_z,
    const int ratio, bool overlap
)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        if (!overlap)
        {
            cpu_sum(h_x, h_y, h_z, N / ratio);
        }

        gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);

        if (overlap)
        {
            // 主机函数与设备核函数重叠。
            cpu_sum(h_x, h_y, h_z, N / ratio);
        }
 
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

void __global__ add(const real *d_x, const real *d_y, real *d_z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N1)
    {
        for (int i = 0; i < 100000; ++i)
        {
            d_z[n] = d_x[n] + d_y[n];
        }
    }
}

void timing(const real *d_x, const real *d_y, real *d_z, const int num)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int n = 0; n < num; ++n)
        {
            int offset = n * N1;

            // 指定各个核函数的cuda流，实现核函数的并行。
            add<<<grid_size, block_size, 0, streams[n]>>>(d_x + offset, d_y + offset, d_z + offset);
        }
 
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%g\n", t_ave);
}

void __global__ add2(const real *x, const real *y, real *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        for (int i = 0; i < 40; ++i)
        {
            z[n] = x[n] + y[n];
        }
    }
}

void timing
(
    const real *h_x, const real *h_y, real *h_z,
    real *d_x, real *d_y, real *d_z,
    const int num
)
{
    int N1 = N / num;
    int M1 = M / num;

    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int i = 0; i < num; i++)
        {
            int offset = i * N1;

            // 划分主机不可分页内存，实现异步的数据传输。
            // 每个cuda流都有各自的数据传输操作。
            CHECK(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, 
                cudaMemcpyHostToDevice, streams[i]));
            CHECK(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, 
                cudaMemcpyHostToDevice, streams[i]));

            int block_size = 128;
            int grid_size = (N1 - 1) / block_size + 1;

            // 指定核函数的cuda流。
            add2<<<grid_size, block_size, 0, streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

            CHECK(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, 
                cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%d %g\n", num, t_ave);
}


