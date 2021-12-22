# CUDA 中的线程组织

CUDA 虽然支持 C++ 但支持得并不充分，导致 C++ 代码中有很多 C 代码的风格。

CUDA 采用 nvcc 作为编译器，支持 C++ 代码；nvcc 在编译 CUDA 程序时，   
会将纯粹的 c++ 代码交给 c++ 编译器，自己负责编译剩下的 cu 代码。

------

## C++ 的 Hello World 程序

    >> g++ hello.cpp -o ./bin/hello.exe
    >> ./bin/hello
    msvc: hello world!

------

## CUDA 的 Hello World 程序

### 使用 nvcc 编译纯粹 c++ 代码

    >> nvcc -o ./bin/hello_cu.exe hello.cu 
    >> ./bin/hello_cu.exe
    nvcc: hello world!

在该程序中其实并未使用 GPU。

### 使用 核函数 的 CUDA 程序

一个利用了 GPU 的 CUDA 程序既有主机代码，又有设备代码（在设备中执行的代码）。  
主机对设备的调用是通过 **核函数（kernel function）** 实现的。

    int main()
    {
        主机代码
        核函数的调用
        主机代码

        return 0；
    }

核函数与 c++ 函数的区别：
1. 必须加 `__global__` 限定；
2. 返回类型必须是空类型 `void`。

    __global__ void hell_from__gpu()
    {
        // 核函数不支持 c++ 的 iostream。
        printf("gpu: hello world!\n");
    }

调用核函数的方式：

    hello_from_gpu<<<1, 1>>>

主机在调用一个核函数时，必须指明在设备中指派多少线程。核函数中的线程常组织为若干线程块： 
1. 三括号中第一个数字是线程块的个数（number of thread block）；
2. 三括号中第二个数字是每个线程块中的线程数（number of thread in per block）。

一个核函数的全部线程块构成一个网格（grid），线程块的个数称为网格大小（grid size）。  
每个线程块中含有相同数目的线程，该数目称为线程块大小（block size）。

所以，核函数的总的线程数即 网格大小*线程块大小:

    hello_from_gpu<<<grid size, block size>>>

调用核函数后，调用 CUDA 运行时 API 函数，同步主机和设备：

    cudaDeviceSynchronize();

核函数中调用输出函数，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。

------

## CUDA 的线程组织

核函数的总线程数必须至少等于计算核心数时才有可能充分利用 GPU 的全部计算资源。  

    hello_from_gpu<<<2, 4>>>

网格大小是2，线程块大小是4，总线程数即8。核函数中代码的执行方式是 “单指令-多线程”，  
即每个线程执行同一串代码。

从开普勒架构开始，最大允许的线程块大小是 2^10 (1024)，最大允许的网格大小是 2^31 - 1（一维网格）。

线程总数可以由两个参数确定：
1. gridDim.x, 即网格大小；
2. blockDim.x, 即线程块大小；

每个线程的身份可以由两个参数确定：
1. blockIdx.x, 即一个线程在一个网格中的线程块索引，[0, gridDm.x);
2. threadIdx.x, 即一个线程在一个线程块中的线程索引，[0, blockDim.x);

网格和线程块都可以拓展为三维结构（各轴默认为 1）：

1. 三维网格 grid_size(gridDim.x, gridDim.y, gridDim.z);
2. 三维线程块 block_size(blockDim.x, blockDim.y, blockDim.z);

相应的，每个线程的身份参数：

1. 线程块ID (blockIdx.x, blockIdx.y, blockIdx.z);
2. 线程ID (threadIdx.x, threadIdx.y, threadIdx.z);

多维网格线程在线程块上的 ID；
    
    tid = threadIdx.z * (blockDim.x * blockDim.y)  // 当前线程块上前面的所有线程数
        + threadIdx.y * (blockDim.x)               // 当前线程块上当前面上前面行的所有线程数
        + threadIdx.x                              // 当前线程块上当前面上当前行的线程数

多维网格线程块在网格上的 ID:

    bid = blockIdx.z * (gridDim.x * gridDim.y)
        + blockIdx.y * (gridDim.x)
        + blockIdx.x

一个线程块中的线程还可以细分为不同的 **线程束（thread warp）**，即同一个线程块中  
相邻的 warp_size 个线程（一般为 32）。

对于从开普勒架构到图灵架构的 GPU，网格大小在 x, y, z 方向的最大允许值为 （2^31 - 1, 2^16 - 1, 2^16 -1）；  
线程块大小在 x, y, z 方向的最大允许值为 （1024， 1024， 64），同时要求一个线程块最多有 1024 个线程。

------

## CUDA 的头文件

CUDA 头文件的后缀一般是 “.cuh”；同时，同时可以包含c/cpp 的头文件 “.h”、“.hpp”，采用 nvcc 编译器会自动包含必要的 cuda 头文件，  
如 <cuda.h>, <cuda_runtime.h>，同时前者也包含了c++头文件 <stdlib.h>。

------

## 使用 nvcc 编译 CUDA 程序

nvcc 会先将全部源代码分离为 主机代码 和 设备代码；主机代码完整的支持 c++ 语法，而设备代码只部分支持。

nvcc 会先将设备代码编译为 PTX（parrallel thread execution）伪汇编代码，再将其编译为二进制 cubin目标代码。  
在编译为 PTX 代码时，需要选项 `-arch=compute_XY` 指定一个虚拟架构的计算能力；在编译为 cubin 代码时，  
需要选项 `-code=sm_ZW` 指定一个真实架构的计算能力，以确定可执行文件能够使用的 GPU。


真实架构的计算能力必须大于等于虚拟架构的计算能力，例如： 

    -arch=compute_35  -code=sm_60  (right)
    -arch=compute_60  -code=sm_35  (wrong)

如果希望编译出来的文件能在更多的GPU上运行，则可以同时指定多组计算能力，例如：
    
    -gencode arch=compute_35, code=sm_35
    -gencode arch=compute_50, code=sm_50
    -gencode arch=compute_60, code=sm_60

此时，编译出来的可执行文件将包含3个二进制版本，称为 **胖二进制文件（fatbinary）**。

同时，nvcc 有一种称为 **实时编译（just-in-time compilation）**机制，可以在运行可执行文件时从其中保留的PTX  
代码中临时编译出一个 cubin 目标代码。因此， 需要通过选项 `-gencode arch=compute_XY, code=compute_XY`，  
指定所保留 PTX 代码的虚拟架构， 例如：

    -gencode arch=compute_35, code=sm_35
    -gencode arch=compute_50, code=sm_50
    -gencode arch=compute_60, code=sm_60  
    -gencode arch=compute_70, code=compute_70

于此同时，nvcc 编译有一个简化的编译选项 `-arch=sim_XY`，其等价于： 

    -gencode arch=compute_XY, code=sm_XY  
    -gencode arch=compute_XY, code=compute_XY

关于 nvcc 编译器的更多资料： [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

------

## 显卡架构和计算能力

1. 费米 Fermi（cuda 3.2~cuda 8）
SM20 or SM_20, compute_30 –
GeForce 400, 500, 600, GT-630.
CUDA 10 以后就完全不支持了。

2. 开普勒 Kepler（cuda 5~cuda 10）
SM30 or SM_30, compute_30 –
GeForce 700, GT-730
支持了统一内存模型编程

SM35 or SM_35, compute_35 –
Tesla K40.
支持动态并行化。

SM37 or SM_37, compute_37 –
Tesla K80.
增加了一些寄存器。

CUDA 11 以后就完全不支持了。

2. 麦克斯韦 Maxwell（CUDA 6~CUDA 11）
SM50 or SM_50, compute_50 –
Tesla/Quadro M 系列

SM52 or SM_52, compute_52 –
Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X

SM53 or SM_53, compute_53 –
Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano

cuda 11 以后彻底不支持

4. 帕斯卡 Pascal (CUDA 8 ~今)
SM60 or SM_60, compute_60 –
Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)

SM61 or SM_61, compute_61–
GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2

SM62 or SM_62, compute_62 –
Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

5. 伏特 Volta (CUDA 9 ~今)
SM70 or SM_70, compute_70 –
DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100

SM72 or SM_72, compute_72 –
Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX

6. 图灵Turing (CUDA 10 ~今)
SM75 or SM_75, compute_75 –
GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4

7. 安培Ampere (CUDA 11 ~今)
SM80 or SM_80, compute_80 –
NVIDIA A100 (不再用特斯拉（Tesla）做名字了 – GA100), NVIDIA DGX-A100

SM86 or SM_86, compute_86 – (from CUDA 11.1 onwards)
Tesla GA10x, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A6000, RTX A40

8. 哈珀Hopper (CUDA 12 计划中)
SM90 or SM_90, compute_90 –
NVIDIA H100 (GH100)
