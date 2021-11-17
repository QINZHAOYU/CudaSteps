#include "error.cuh"


// 静态全局内存变量。  
__device__ int d_x = 1;
__device__ int d_y[2] = {2, 3};


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


int main()
{
    display<<<1, 1>>>();
    add_array<<<1, 1>>>();
    add_var<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
    int h_y[2] = {10, 20};

    // cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, 
    //     size_t offset=0, cudaMemcpyKind kind=cudaMemcpyHostToDevice);
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int)));
    display<<<1, 1>>>();

    // cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, 
    //     size_t offset=0, cudaMemcpyKind kind=cudaMemcpyDeviceToHost);
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("host, d_y: %d, %d\n", h_y[0], h_y[1]);
    display<<<1, 1>>>();

    return 0;
}




