#include "common/error.cuh"


// 静态全局内存变量(设备内存)。  
__device__ int d_x = 1;
__device__ int d_y[2] = {2, 3};


// 常量内存变量（设备内存）。
__constant__ double d_m = 23.33;
__constant__ double d_n[] = {12.2, 34.1, 14.3};


// 核函数。
__global__ void add_array()
{
    d_y[0] += d_x;
    d_y[1] += d_x;

    printf("d_y: {%d, %d}\n", d_y[0], d_y[1]);
}
__global__ void add_var()
{
    d_x += 2;

    printf("d_x: %d\n", d_x);
}
__global__ void display()
{
    printf("d_x: %d, d_y: {%d, %d}\n", d_x, d_y[0], d_y[1]);
}
__global__ void show()
{
    // 常量内存变量在核函数中不可更改（只读）。

    printf("d_m: %f, d_n: {%f, %f, %f}\n", d_m, d_n[0], d_n[1], d_n[2]);
}


int main()
{
    display<<<1, 1>>>();
    add_array<<<1, 1>>>();
    add_var<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    show<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    int h_y[2] = {10, 20};
    int h_x = 7;

    double h_m = 22.23;
    double h_n[3] = {1.1, 2.2, 3.3};

    // cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, 
    //     size_t offset=0, cudaMemcpyKind kind=cudaMemcpyHostToDevice);
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_x, &h_x, sizeof(int)));  // 对于全局内存变量，非数组对象不用取址。
    display<<<1, 1>>>();

    CHECK(cudaMemcpyToSymbol(d_m, &h_m, sizeof(double)));  // 对于常量内存变量，非数组对象不用取址。 
    CHECK(cudaMemcpyToSymbol(d_n, h_n, sizeof(double)*3));
    show<<<1, 1>>>();

    // cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, 
    //     size_t offset=0, cudaMemcpyKind kind=cudaMemcpyDeviceToHost);
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    CHECK(cudaMemcpyFromSymbol(&h_x, d_x, sizeof(int)));  // 对于全局内存变量，非数组对象不用取址。
    printf("host, h_y: %d, %d, h_x: %d\n", h_y[0], h_y[1], h_x);
    display<<<1, 1>>>();

    CHECK(cudaMemcpyFromSymbol(h_n, d_n, sizeof(double)*3));
    CHECK(cudaMemcpyFromSymbol(&h_m, d_m, sizeof(double)));  // 对于常量内存变量，非数组对象不用取址。  
    printf("host, h_n: %f, %f, h_m: %f\n", h_n[0], h_n[1], h_m);
    show<<<1, 1>>>();

    return 0;
}




