#include "common/error.cuh"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;  // 双精度
    const real EPSILON = 1.0e-15;
#else
    typedef float real;   // 单精度
    const real EPSILON = 1.0e-6f;
#endif


// 核函数。
__global__ void add(const real *x, const real *y, real *z, const int N);

// 重载设备函数。
__device__ real add_in_device(const real x, const real y);
__device__ void add_in_device(const real x, const real y, real &z);

// 主机函数。
void check(const real *z, const int N);
