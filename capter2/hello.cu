#include <cstdio>
using namespace std;


__global__ void hell_from__gpu()
{
    // 核函数不支持 c++ 的 iostream。
    
    // 输出流的缓存顺序。
    // printf("gpu: hello world! ");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    printf("gpu: hello world! block(%d, %d, %d) -- thread(%d, %d, %d)\n", bx, by, bz, tx, ty, tz);
}


int main()
{
    printf("nvcc: hello world!\n");

    const dim3 block_size(2, 4);
    hell_from__gpu<<<1, block_size>>>();
    cudaDeviceSynchronize(); // 同步主机和设备，否则无法输出字符串。

    return 0;
}