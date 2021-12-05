# 线程束基本函数与协作组

线程束（warp），即一个线程块中连续32个线程。

------

## 单指令-多线程模式

一个GPU被分为若干个流多处理器（SM）。核函数中定义的线程块（block）在执行时  
将被分配到还没有完全占满的 SM。  
一个block不会被分配到不同的SM，同时一个 SM 中可以有多个 block。不同的block  
之间可以并发也可以顺序执行，一般不能同步。当某些block完成计算任务后，对应的  
SM 会部分或完全空闲，然后会有新的block被分配到空闲的SM。

一个 SM 以32个线程（warp）为单位产生、管理、调度、执行线程。  
一个 SM 可以处理多个block，一个block可以分为若干个warp。

在同一时刻，一个warp中的线程只能执行一个共同的指令或者闲置，即**单指令-多线程**执行模型，  
（single instruction multiple thread, SIMT）。

当一个线程束中线程顺序的执行判断语句中的不同分支时，称为发生了 **分支发散**（branch divergence）。

    if (condition)
    {
        A;
    }
    else
    {
        B;
    }

首先，满足 `condition` 的线程或执行语句A，其他的线程会闲置；  
然后，不满足条件的将会执行语句B，其他线程闲置。  
当语句A和B的指令数差不多时，整个warp的执行效率就比没有分支的情况 *低一半*。

一般应当在核函数中尽量避免分支发散，但有时这也是不可避免的。  
如数组计算中常用的判断语句：

    if(n < N)
    {
        // do something.
    }

该分支判断最多影响最后一个block中的某些warp发生分支发散，  
一般不会显著地影响性能。

有时能通过 **合并判断语句** 的方式减少分支发散；另外，如果两分支中有一个分支  
不包含指令，则即使发生分支发散也不会显著影响性能。

*注意不同架构中的线程调度机制*

------

## 线程束内的线程同步函数

`__syncwarp(·)`：当所涉及的线程都在一个线程束内时，可以将线程块同步函数 `__syncthreads()` 
换成一个更加廉价的线程束同步函数`__syncwarp(·)`，简称 **束内同步函数**。

函数参数是一个代表掩码的无符号整型数，默认值是全部32个二进制位都为1，代表  
线程束中的所有线程都参与同步。

关于 *掩码(mask)* 的简介文章：[思否小姐姐：“奇怪的知识——位掩码”](https://zhuanlan.zhihu.com/p/352025616)

------

## 更多线程束内的基本函数

**线程束表决函数**；  
+ `unsigned __ballot_sync(unsigned mask, int predicate)`，  
如果线程束内第n个线程参与计算（旧掩码）且predicate值非零，则返回的无符号整型数（新掩码）  
的第n个二进制位为1，否则为0。

+ `int __all_sync(unsigned mask, int predicate)`，   
线程束内所有参与线程的predicate值均非零，则返回1，否则返回0.

+ `int __any_sync(unsigned mask, int predicate)`，  
线程束内所有参与线程的predicate值存在非零，则返回1， 否则返回0.

**线程束洗牌函数**：  
+ `T __shfl_sync(unsigned mask, T v, int srcLane, int w = warpSize)`，  
参与线程返回标号为 srcLane 的线程中变量 v 的值。  
该函数将一个线程中的数据广播到所有线程。

+ `T __shfl_up_sync(unsigned mask, T v, unsigned d, int w=warpSize)`，  
标号为t的参与线程返回标号为 t-d 的线程中变量v的值，t-d<0的线程返回t线程的变量v。   
该函数是一种将数据向上平移的操作，即将低线程号的值平移到高线程号。  
例如当w=8、d=2时，2-7号线程将返回 0-5号线程中变量v的值；0-1号线程返回自己的 v。

+ `T __shfl_down_sync(unsigned mask, T v, unsigned d, int w=warpSize)`，  
标号为t的参与线程返回标号为 t+d 的线程中变量v的值，t+d>w的线程返回t线程的变量v。  
该函数是一种将数据向下平移的操作，即将高线程号的值平移到低线程号。  
例如当w=8、d=2时，0-5号线程将返回2-7号线程中变量v的值，6-7号线程将返回自己的 v。

+ `T __shfl__xor_sync(unsigned mask, T v, int laneMask, int w=warpSize)`,  
标号为t的参与线程返回标号为 t^laneMask 的线程中变量 v 的值。  
该函数让线程束内的线程两两交换数据。

每个线程束洗牌函数都有一个可选参数 `w`，默认是线程束大小（32），且只能取2、4，8、16、32。  
当 w 小于 32 时，相当于逻辑上的线程束大小是 w，其他规则不变。  
此时，可以定义一个 **束内索引**：(假设使用一维线程块)
   
    int laneId = threadIdx.x % w;  // 线程索引与束内索引的对应关系。

假设线程块大小为16，w 为 8：

    线程索引： 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
    束内索引： 0 1 2 3 4 5 6 7 0 1 2  3  4  5  6  7

参数中的 `mask` 称为掩码，是一个无符号整型，具有32位，一般用 *十六进制* 表示：  

    const unsigned FULL_MASK = 0xffffffff; // `0x`表示十六进制数；`0b`表示二进制数。

或者

    #define FULL_MASK 0xffffffff

以上所有线程束内函数都有 `_sync` 后缀，表示这些函数都具有 **隐式的同步功能**。

------

## 协作组

协作组（cooperative groups），可以看作是线程块和线程束同步机制的推广，  
提供包括线程块内部的同步与协作、线程块之间（网格级）的同步与协作、以及  
设备与设备之间的同步与协作。

使用协作组需要包含如下头文件：  

    #include <cooperative_groups.h>
    using namespace cooperative_groups;

**线程块级别的协作组**

协作组编程模型中最基本的类型是线程组 `thread_group`，其包含如下成员： 
+ `void sync()`，同步组内所有线程；
+ `unsigned size()`，返回组内总的线程数目，即组的大小；
+ `unsigned thread_rank()`，返回当前调用该函数的线程在组内的标号（从0计数）；
+ `bool is_valid()`，如果定义的组违反了任何cuda限制，返回 false，否则true；

线程组类型有一个导出类型，**线程块thread_block**，其中定义了额外的函数：
+ `dim3 group_index()`，返回当前调用该函数的线程的线程块指标，等价于 `blockIdx`；
+ `dim3 thread_index()`，返回当前调用该函数的线程的线程指标，等价于 `threadIdx`；

通过 `this_thread_block()` 初始化一个线程块对象： 

    thread_block g = this_thread_block();  // 相当于一个线程块类型的常量。

此时，  

    g.sync() <===> __syncthreads()
    g.group_index() <===> blockIdx
    g.thread_index() <===> threadIdx

通过 `tiled_partition()` 可以将一个线程块划分为若干片（tile），每一片构成一个新的线程组。  
目前，仅支持将片的大小设置为 2 的整数次方且不大于 32。  

    thread_group g32 = tiled_partition(this_thread_block(), 32); // 将线程块划分为线程束。

可以继续将线程组划分为更细的线程组：  

    thread_group g4 = tiled_partition(g32, 4);

采用模板、在编译期划分 **线程块片（thread block tile）**：  

    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    thread_block_tile<32> g4 = tiled_partition<4>(this_thread_block());

线程块片具有额外的函数（类似线程束内函数）：  
+ `unsigned ballot(int predicate)`;
+ `int all(int predicate)`;
+ `int any(int predicate)`;
+ `T shfl(T v, int srcLane)`;
+ `T shfl_up(T v, unsigned d)`;
+ `T shfl_down(T v, unsigned d)`;
+ `T shfl_xor(T v, unsigned d)`;

与一般的线程束不同，线程组内的所有线程都要参与代码运行计算；同时，线程组内函数不需要指定宽度，  
因为该宽度就是线程块片的大小。

------

## 数组归约程序的进一步优化

**提高线程利用率**

在当前的归约程序中，当 offset=64，只用了 1/2 的线程；当 offset=32，只用了 1/4 的线程；... 
最终，当 offset=1，只用了 1/128 的线程；  
归约过程一共用了 log2(128) = 7 步，平均线程利用率 (1/2 + 1/4 + ... + 1/128)/7 => 1/7。

而在归约前的数据拷贝中线程利用率为 100%，可以尽量把计算放在在归约前：让一个线程处理多个数据。

一个线程处理相邻若干个数据会导致全局内存的非合并访问。要保证全局内存的合并访问，这里需要  
保证相邻线程处理相邻数据，一个线程访问的数据需要有某种跨度。  
该跨度可以是线程块的大小，也可以是网格的大小；对于一维情况，分别是 blockDim.x 和 blockDim.x * gridDim.x。

**避免反复分配与释放设备内存**

设备内存的分配与释放是比较耗时的。  
通过采用静态全局内存替代动态全局内存，实现编译期的设备内存分配可以更加高效。

此外，应当尽量避免在较内存循环反复的分配和释放设备内存。

------

