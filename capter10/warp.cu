#include "../common/error.cuh"

const unsigned WIDTH = 8;
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff;

__global__ void test_warp_primitives(void)
{
    int tid = threadIdx.x;
    int laneId = tid % WIDTH;

    if (tid == 0)
    {
        printf("threadIdx.x: ");
    }
    printf("%2d  ", tid);
    if (tid == 0)
    {
        printf("\n");
    }

    if (tid == 0)
    {
        printf("laneId: ");
    }
    printf("%2d  ", laneId);
    if (tid == 0)
    {
        printf("\n");
    }

    // 从 FULL_MASK 出发， 计算 mask1（排除 0 号线程）掩码和mask2（仅保留 0 线程）掩码。
    unsigned mask1 = __ballot_sync(FULL_MASK, tid>0);
    unsigned mask2 = __ballot_sync(FULL_MASK, tid==0);

    if (tid == 0)
    {
        printf("FULL_MASK = %x\n", FULL_MASK);
    }
    if (tid == 1)
    {
        printf("mask1     = %x\n", mask1);
    }
    if (tid == 0)
    {
        printf("mask2     = %x\n", mask2);
    }

    // 从 FULL_MASK 出发计算线程束状态。
    // 因为不是所有线程的tid 都大于0，所以此处返回 0.
    int result = __all_sync(FULL_MASK, tid);
    if (tid == 0) 
    {
        printf("all_sync (FULL_MASK): %d\n", result);
    }
    // 从 mask1 出发计算线程束状态。
    // 因为mask1 中关闭了0号线程，所以剩下的所有线程tid > 0，此处返回 1.
    result = __all_sync(mask1, tid);
    if (tid == 1) 
    {
        printf("all_sync (mask1): %d\n", result);
    }

    // 从 FULL_MASK 出发计算线程束状态。
    // 因为存在线程的tid 都大于0，所以此处返回 1.
    result = __any_sync(FULL_MASK, tid);
    if (tid == 0)
    {
        printf("any_sync (FULL_MASK): %d\n", result);
    }
    // 从 mask2 出发计算线程束状态。
    // 因为mask2 中仅激活了 0 号线程，所以此处返回 0.
    result = __any_sync(mask2, tid);
    if (tid == 0)
    {
        printf("any_sync (mask2): %d\n", result);
    }

    // 从 FULL_MASK 出发，将每个线程束中 2号线程的tid广播到线程束内所有函数并作为返回值。
    // 所以在第一个线程束中，所有8个线程都将返回 laneId=2 线程（2 号线程）的tid值；
    // 在第二个线程束中，所有8个线程都也将返回 landId=2 线程（10 号线程）的tid值。
    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    if (tid == 0)
    {
        printf("shfl:    ");
    }
    printf("%2d  ", value);
    if (tid == 0)
    {
        printf("\n");
    }

    // 从FULL_MASK出发，将每个线程束内 1-7 号线程取 0-6号线程的tid值并作为返回值。
    // 所以在第一个线程束中，0号线程返回自己的tid，1号线程返回0号线程的tid，2号线程返回1号线程tid, ...
    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0)
    {
        printf("shfl_up:    ");
    }
    printf("%2d  ", value);
    if (tid == 0)
    {
        printf("\n");
    }

    // 从 FULL_MASK 出发，将每个线程束内 0-6号线程取 1-7 号线程的tid值并作为返回值。
    // 所以在第一个线程束中，0号线程返回1号线程的tid，2号线程返回3号线程的tid，..., 7号线程返回自己的tid。
    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0)
    {
        printf("shfl_down:    ");
    }
    printf("%2d  ", value);
    if (tid == 0)
    {
        printf("\n");
    }    

    // 从 FULL_MASK 出发，将线程束中相邻的线程的tid相互传递并作为返回值。
    // 所以在第一个线程束中，0号线程返回1号线程的tid、1号线程返回0号线程的tid，2号线程返回3号线程的tid、  
    // 3号线程返回2号线程的tid，...
    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0)
    {
        printf("shfl_xor:    ");
    }
    printf("%2d  ", value);
    if (tid == 0)
    {
        printf("\n");
    }    
}


int main()
{
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize());

    return 0;
}

