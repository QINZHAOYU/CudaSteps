# 获得 GPU 加速的关键

------ 

## CUDA 事件计时

C++ 的计时方法：

1. GCC 和 MSVC 都有的 `clock()`函数；
2. 原生的 <chrono> 时间库；
3. GCC 的 `gettimeofday()`计时；
4. MSVC 的 `QueryPerformanceCounter()` 和 `QueryPerformanceFrequency()` 计时。

CUDA 基于 CUDA 事件的计时方法：

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); // 创建cuda 事件对象。
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));  // 记录代表开始的事件。
    cudaEventQuery(start);  // 强制刷新 cuda 执行流。

    // run code.

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); // 强制同步，让主机等待cuda事件执行完毕。
    float elapsed_time = 0;
    CHECK(cudaEventElapsedTime(&curr_time, start, stop)); // 计算 start 和stop间的时间差（ms）。
    printf("host memory malloc and copy: %f ms.\n", curr_time - elapsed_time);  

由于 cuda 程序需要在主机和设备间传递数据，所以当计算强度较小时数据传输的性能对程序总耗时影响更大。
因此 cuda 的两种浮点数类型对程序性能的影响就较为明显。考虑提供编译选项，指定版本：
    
    #ifdef USE_DP
        typedef double real;  // 双精度
        const real EPSILON = 1.0e-15;
    #else
        typedef float real;   // 单精度
        const real EPSILON = 1.0e-6f;
    #endif

在编译时，除了指定 GPU 计算能力 `-arch=sm_50`，还可以指定 c++ 优化等级 `-O3`;同时，可以指定其他  
编译选项，如 `-DUSE_DP` 启用双精度版本。

    >> nvcc -O3 -arch=sm_50 -DUSE_DP -o ./bin/clock.exe add.cu clock.cu main.cpp
    ...
    >> ./bin/clock
    using double precision version
    host memory malloc and copy: 2.054112 ms.
    device memory malloc: 9.063583 ms.
    kernel function : 0.803360 ms.
    cuda; no error
    copy from device to host: 7.489505 ms.  

    >> nvcc -O3 -arch=sm_50 -o ./bin/clock.exe add.cu clock.cu main.cpp
    ...
    >> ./bin/clock     
    host memory malloc and copy: 0.950240 ms.
    device memory malloc: 5.298208 ms.
    kernel function : 0.620512 ms.
    cuda; no errors
    copy from device to host: 3.034208 ms.

可见双精度版本基本上比单精度版本耗时多一倍。

------

## nvprof 查看程序性能

    >> nvprof ./bin/clock

如果没有输出结果，需要将`nvprof`的目录包含到环境环境变量中（不支持7.5 以上计算能力的显卡）。  
推荐采用一代性能分析工具： [Nvidia Nsight Systems](https://developer.nvidia.com/zh-cn/nsight-systems).

------

## 影响 GPU 加速的关键因素

1. 要获得可观的 GPU 加速，就必须尽量缩减主机和设备间数据传输所花时间的占比。

有些计算即使在 GPU 中速度不高也要尽量放在 GPU 中实现，以避免过多数据经由 PCIe 传递。

2. 提高算术强度可以显著地提高 GPU 相对于 CPU 的加速比。

**算术强度**，是指一个计算问题中算术操作的工作量与必要的内存操作的工作量之比。  
对设备内存的访问速度取决于 GPU 的显存带宽。

3. 核函数的并行规模。

并行规模可以用 GPU 中的线程数目来衡量。  
一个 GPU 由多个流多处理器SM（streaming multiprocessor）构成，每个 SM 中有若干 CUDA 核心。  
每个 SM 是相对独立的，一个 SM 中最多驻留的线程数一般为 2048 或 1024（图灵架构）。

若要 GPU 满负荷工作，则核函数中定义的线程总数要不少于某值，一般与 GPU 能够驻留的线程总数相当。

------

## CUDA 的数学函数库

CUDA 提供的数学函数库提供了多种 **数学函数**，同时 CUDA 提供了一些高效率、低准确度的 **内建函数**。

CUDA 数学函数库的更多资料，详见：[CUDA math](https://docs.nvidia.com/cuda/cuda-math-api/index.html).

------
