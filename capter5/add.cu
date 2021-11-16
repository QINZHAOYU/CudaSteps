#include "add.cuh"

const real c = 3.57;


__global__ void add(const real *x, const real *y, real *z, const int N)
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

__device__ real add_in_device(const real x, const real y)
{
    return x + y;
}

__device__ void add_in_device(const real x, const real y, real &z)
{
    z = x + y;
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N ;++i)
    {
        if (fabs(z[i] - c) > EPSILON)
        {
            has_error = true;
        }
    }

    printf("cuda; %s\n", has_error ? "has error" : "no error");
}





