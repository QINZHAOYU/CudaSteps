#include <math.h>
//#include <stdlib.h>
#include <stdio.h>


const double EPSILON = 1.0e-10;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;


// 核函数。
__global__ void add(const double *x, const double *y, double *z, const int N);

// 重载设备函数。
__device__ double add_in_device(const double x, const double y);
__device__ void add_in_device(const double x, const double y, double &z);

// 主机函数。
void check(const double *z, const int N);


int main()
{
    const int N = 1e4;
    const int M = sizeof(double) * N;

    // 申请主机内存。
    // 支持使用 new-delete 方式创建和释放内存。
    //double *h_x = (double*) malloc(M);
    double *h_x = new double[N];
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化主机数据。
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    // 申请设备内存。
    // cudeError_t cudaMalloc(void **address, size_t size);
    double *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, M);
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);

    // 从主机复制数据到设备。
    // cudaError_t cudaMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind);
    // kind 可以简化使用 `cudaMemcpyDefault`，由系统自动判断拷贝方向（x64主机）。
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // 在设备中执行计算。
    const int block_size = 128;
    const int grid_size = N/128 + 1; // 线程数应该不少于计算数目。
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    // 从设备复制数据到主机。
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    // 释放主机内存。
    // free(h_x);
    if (h_x) delete[] h_x;
    free(h_y);
    free(h_z);

    // 释放设备内存。
    // cudaError_t cudaFree(void *address);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}


__global__ void add(const double *x, const double *y, double *z, const int N)
{
    // 在主机函数中需要依次对每个元素进行操作，需要使用一个循环。
    // 在设备函数中，因为采用“单指令-多线程”方式，所以可以去掉循环、只要将数组元素索引和线程索引一一对应即可。

    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n > N) return;

    if (n%5 == 0)
    {
        z[n] = add_in_device(x[n], y[n]);
    }
    else
    {
        add_in_device(x[n], y[n], z[n]);
    }
}

__device__ double add_in_device(const double x, const double y)
{
    return x + y;
}

__device__ void add_in_device(const double x, const double y, double &z)
{
    z = x + y;
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N ;++i)
    {
        if (fabs(z[i] - c) > EPSILON)
        {
            //printf("%d, %f, %f\n", i, z[i], c);
            has_error = true;
        }
    }

    printf("cuda; %s\n", has_error ? "has error" : "no error");
}
