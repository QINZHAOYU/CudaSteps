# CUDA 程序的错误检测

------

## 检测 CUDA 运行时错误的宏函数

定义检查 cuda 运行时 API 返回值 `cudaError_t` 的宏函数。

    #define CHECK(call)                                                     \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA ERROR: \n");                                       \
            printf("    FILE: %s\n", __FILE__);                             \
            printf("    LINE: %d\n", __LINE__);                             \
            printf("    ERROR CODE: %d\n", error_code);                     \
            printf("    ERROR TEXT: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    }while(0); 

因为核函数没有返回值，所以无法直接检查核函数错误。间接的方法是，在调用核函数后执行：
    
    CHECK(cudaGetLastError());  // 捕捉同步前的最后一个错误。
    CHECK(cudaDeviceSynchronize());  // 同步主机和设备。

核函数的调用是 **异步的**，即主机调用核函数后不会等待核函数执行完成、而是立刻执行之后的语句。  
同步操作较为耗时，一般尽量避免；同时，只要在核函数调用后还有对其他任何能返回错误值的 API   
函数进行同步调用，都会触发主机和设备的同步并捕捉到核函数中可能发生的错误。

此外，主机和设备之间的数据拷贝会隐式地同步主机和设备。一般要获得精确的出错位置，还是需要显式地  
同步，例如调用 `cudaDeviceSynchronize()`。

或者，通过设置环境变量 `CUDA_LAUNCH_BLOCKING` 为 1，这样所有核函数的调用都将不再是异步的，  
而是同步的。就是说，主机调用一个核函数之后必须等待其执行完，才能向下执行。  
一般仅用于程序调试。

------

## CUDA-MEMCHECK 检查内存错误

CUDA 提供了 CUDA-MEMCHECK 的工具集，包括 memcheck, racecheck, initcheck, synccheck.

    >> cuda-memcheck --tool memcheck [options] app-name [options]

对于 memcheck 工具，可以简化为：

    >> cuda-memcheck [options] app-name [options]

对于本例，可以通过如下方式检测错误： 

    >> cuda-memcheck ./bin/check.exe
    ========= CUDA-MEMCHECK
CUDA ERROR: 
    FILE: check.cu
    LINE: 56
    ERROR CODE: 9
    ERROR TEXT: invalid configuration argument
    ========= Program hit cudaErrorInvalidConfiguration (error 9) due to "invalid configuration argument" on CUDA API call to     cudaLaunchKernel.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x97c18) [0x2b8ca8]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x9a2da) [0x2bb36a]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     [0x7b52e]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x11ceaa) [0x33df3a]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x137532) [0x3585c2]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0x1679]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xd32b]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xd1a8]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xc6a1]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xcbf8]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xd944]
    =========     Host Frame:C:\Windows\System32\KERNEL32.DLL (BaseThreadInitThunk + 0x14) [0x17034]
    =========     Host Frame:C:\Windows\SYSTEM32\ntdll.dll (RtlUserThreadStart + 0x21) [0x52651]
    =========
    ========= Program hit cudaErrorInvalidConfiguration (error 9) due to "invalid configuration argument" on CUDA API call to     cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x97c18) [0x2b8ca8]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x9a2da) [0x2bb36a]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     [0x7b52e]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x11ceaa) [0x33df3a]
    =========     Host Frame:C:\Windows\system32\DriverStore\FileRepository\nvhq.inf_amd64_5550755be1247d27\nvcuda64.dll     (cuProfilerStop + 0x137532) [0x3585c2]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0x1461]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xcbfd]
    =========     Host Frame:D:\3_codes\CudaSteps\capter4\bin\check.exe [0xd944]
    =========     Host Frame:C:\Windows\System32\KERNEL32.DLL (BaseThreadInitThunk + 0x14) [0x17034]
    =========     Host Frame:C:\Windows\SYSTEM32\ntdll.dll (RtlUserThreadStart + 0x21) [0x52651]
    =========
    ========= ERROR SUMMARY: 2 errors

关于 CUDA-MEMCHECK 的更多内容，详见: [CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)。

------
