#include "../common/error.cuh"
#include "../common/floats.hpp"
#include "../common/clock.cuh"


__global__ void reduce(real *x, real *y, const int N)
{
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;    

    for (int offset = blockDim.x/2 ; offset > 0 ; offset /= 2)
    {
        if (tid < offset)
        {
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        y[blockIdx.x] = curr_x[0];
    }
}

__global__ void reduce2(real *x, real *y, const int N)
{
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;    

    for (int offset = blockDim.x/2 ; offset > 0 ; offset /= 2)
    {
        if (tid < offset)
        {
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // 原子函数 atomicAdd(*address, val).
        atomicAdd(y, curr_x[0]);
    }
}



int main()
{
    int N = 1e8;
    int M = N * sizeof(real);

    int bSize = 32;
    int gSize = (N + bSize - 1)/bSize;

    cout << FLOAT_PREC << endl;

    real *h_x, *h_y;
    h_x = new real[N];
    h_y = new real[gSize];
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = 1.23;
    }

    cudaClockStart

    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, gSize*sizeof(real)));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));

    cudaClockCurr

    reduce<<<gSize, bSize, (bSize+1)*sizeof(real)>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, gSize*sizeof(real), cudaMemcpyDefault));
    real res = 0;
    for(int i = 0; i < gSize; ++i)
    {
        res += h_y[i];
    }
    cout << "reduce result: " << res << endl;

    cudaClockCurr

    reduce<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y, N);
    
    CHECK(cudaMemcpy(h_y, d_y, gSize*sizeof(real), cudaMemcpyDefault));
    res = 0.0;
    for(int i = 0; i < gSize; ++i)
    {
        res += h_y[i];
    }
    cout << "reduce result: " << res << endl;

    cudaClockCurr
    
    real *d_y2, *h_y2;
    h_y2 = new real(0.0);
    CHECK(cudaMalloc(&d_y2, sizeof(real)));

    // 采用原子函数、共享内存的核函数归约，
    // 由于减少了主机和设备间的数据传输，效率得以提高。
    reduce2<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y2, N);

    CHECK(cudaMemcpy(h_y2, d_y2, sizeof(real), cudaMemcpyDefault));
    cout << "reduce2 result: " << *h_y2 << endl;

    cudaClockCurr

    delete[] h_x;
    delete[] h_y;
    delete h_y2;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_y2));

    return 0;
}







