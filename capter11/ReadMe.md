# CUDA 流

一个 CUDA 流一般是指由主机发出的、在设备中执行的cuda操作序列（即和cuda有关的操作，  
如主机--设备数据传输和核函数执行）。目前不考虑由设备段发出的流。

任何cuda操作都存在于某个cuda流，要么是 **默认流（default stream）**，也称为 **空流**；  
要么是明确指定的流。非默认的cuda流（非空流）都是在主机端产生与销毁。
一个cuda流由类型为 `cudaStream_t` 的变量表示，创建与销毁的方式：

```cuda
cudaSteam_t stream;
CHECK(cudaStreamCreate(&stream));
...
CHECK(cudaStreamDestroy(stream));
```

主机中可以产生多个相互独立的cuda流，并实现cuda流之间的并行。

为了检查一个cuda流中所有操作是否都已在设备中执行完毕：  

```cuda
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
```

`cudaStreamSynchronize` 会强制阻塞主机，直到其中的stream流执行完毕；  
`cudaStreamQuery` 不会阻塞主机，只是检查cuda流（stream）是否执行完毕，若是，则返回 `cudaSuccess`;  
否则，返回 `cudaErrorNotReady`。

------

## 在默认流中重叠主机和设备计算

**同一个cuda流在设备中都是顺序执行的。** 在数组相加的例子中：

```cuda
cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault);
cudaMemcpy(d_y, h_y, M, cudaMemcpyDefault);
add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault);
```

从设备的角度，以上4个cuda语句是按代码顺序执行的。

采用 `cudaMemcpy` 函数在主机与设备间拷贝数据，是具有隐式同步功能的。  
所以从主机的角度看，数据传输是同步的或者说阻塞的，即主机在发出命令：  

```cuda
cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault);
```

之后，会等待该命令执行完完毕，再接着往下走；数据传输时，主机是闲置的。
与此不同的是，核函数的启动是异步的或者说非阻塞的，即在主机发出命令：

```cuda
add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
```

之后，不会等待该命令执行完毕，而是立刻得到程序的控制权。紧接着发出：  

```cuda
cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault);
```

然而，该命令不会被立刻执行，因为其与核函数同处默认流，需要顺序执行。

所以，主机在发出核函数调用后会立刻发出下一个命令；如果下一个命令是  
主机中的某个计算任务，那么主机就会在设备执行核函数的同时执行计算。  
这样就可以实现主机和设备的重叠计算。

当主机和设备的计算量相当时，将主机函数放在设备核函数后可以达到主机函数  
与设备函数并发执行的效果，从而有效地隐藏主机函数的执行时间。

------

## 非默认 cuda 流重叠多个核函数

要实现多个核函数之间的并行必须使用多个非默认 cuda 流。

使用多个流相对于使用一个流有加速效果；当流的数目超过某个阈值时，加速比就趋于饱和。  
制约加速比的因素：  
+ GPU 计算资源，当核函数的线程总数超过某一值时，再增加流的数目就不会带来更高性能；
+ GPU 中能够并发执行的核函数的上限。

指定核函数的cuda流的方法：

```cuda
kernal_func<<<grid_size, block_size, 0, stream>>>(params);
```

在调用核函数时，如果不需要使用共享内存，则该项设为0；同时指定cuda流的id。

计算能力为7，5的GPU能执行的核函数上限值为128。

------

## 非默认 cuda 流重叠核函数与数据传递

要实现核函数执行与数据传输的并发（重叠），必须让这两个操作处于不同的非默认流；  
同时，数据传输需要使用 `cudaMemcpy` 的异步版本 `cudaMemcpyAsync`。

异步传输由GPU的DMA（direct memory access）实现，不需要主机的参与。

使用异步的数据传输函数时，需要将主机内存定义为不可分页内存或者固定内存，从而  
防止在程序执行期间物理地址被修改。如果将可分页内存传递给 `cudaMemcpyAsync`  
则会导致同步传输。

主机不可分页内存的分配与释放：  

```cuda
cudaError_t cudaMallocHost(void **ptr, size_t size);
或者
cudaError_t cudaHostAlloc(void **ptr, size_t size);

cudaError_t cudaFreeHost(void *ptr);
```

要利用多个流提升性能，一种方法是将数据和相应计算操作分为若干等分，  
然后在每个流中发布一个cuda操作序列。

如果核函数执行、主机与设备间的数据传输这3个cuda操作能完全并行执行，  
理论上最大加速比为 3。

------