# GPU 硬件与 CUDA 程序开发工具

------

## GPU 硬件

在由 CPU 和 GPU 构成的异构计算平台中，通常将起控制作用的 CPU 称为 **主机（host）**，  
将起加速作用的 GPU 称为 **设备（device）**。 

主机和设备都有自己的 DRAM，之间一般由 PCIe 总线连接。

GPU 计算能力不等价于计算性能；表征计算性能的一个重要参数是 **浮点数运算峰值（FLOPS）**。  
浮点数运算峰值有单精度和双精度之分。对于 Tesla 系列的 GPU，双精度下 FLOPS 一般是单精度下的 1/2; 
对于 GeForce 系列的 GPU，双精度下 FLOPS 一般是单精度下的 1/32。

影响计算性能的另一个参数是 **GPU 内存带宽（显存）**。

------

## CUDA 程序开发工具

1. CUDA；
2. OpenCL，更为通用的各种异构平台编写并行程序的框架，AMD 的 GPU 程序开发工具；
3. OpenACC，由多公司共同开发的异构并行编程标准。

CUDA 提供两层 API，即 CUDA 驱动API 和 CUDA 运行时API。  
CUDA 开发环境中，程序应用程序是以主机（CPU）为出发点的；应用程序可以调用 CUDA 运行时 API、  
CUDA 驱动 API 和一些已有的 CUDA 库。

------

## CUDA 开发环境搭建

linux 操作系统：[linux下cuda环境搭建](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

windows10 操作系统：[windows10下cuda环境搭建](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

------

## nvidia-smi 检查与设置设备

    >> nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 462.30       Driver Version: 462.30       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce MX450      WDDM  | 00000000:2B:00.0 Off |                  N/A |
    | N/A   39C    P8    N/A /  N/A |    119MiB /  2048MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

1. **CUDA Version**， 11.2；
2. **GPU Name**，GeForce MX450，设备号为 0；如果系统中有多个 GPU 且只要使用其中某个特定的 GPU，  
可以通过设置环境变量 **CUDA_VISIBLE_DEVICES** 的值，从而可以在运行 CUDA 程序前选定 GPU;  
3. **TCC/WDDM**，WDDM（windows display driver model），其它包括 TCC（Tesla compute cluster）；  
可以通过命令行 `nvidia-smi -g GPU_ID -dm 0`，设置为 WDDM 模式（1 为 TCC 模式）；
4. **Compute mode**, Default，此时同一个 GPU 中允许存在多个进程；其他模式包括 E.Process，  
指的是独占进程模式，但不适用 WDDM 模式下的 GPU；  
可以通过命令行 `nvidia-smi -i GPU_ID -c 0`，设置为 Default 模式（1 为 E.Process 模式）;
5. **Perf**，p8（GPU 性能状态，最大p0~最小p12）；

更多关于 nvidia-smi 的资料：[nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)

------