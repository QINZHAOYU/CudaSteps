#include "../common/error.cuh"
#include "../common/floats.hpp"
#include "../common/clock.cuh"
#include <fstream>
#include <regex>
#include <string>
#include <vector>


void read_data(const std::string &fstr, std::vector<real> &x, std::vector<real> &y);
void write_data(const std::string &fstr, const int *NL, const int N, const int M);
void find_neighbor(int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real minDis);
__global__ void find_neighbor_gpu (int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real mindDis);
__global__ void find_neighbor_atomic(int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real minDis);


int main()
{
    cout << FLOAT_PREC << endl;

    std::string fstr = "xy.txt";
    std::string fout = "result.txt";
    std::vector<real> x, y;
    read_data(fstr, x, y);

    int N = x.size(), M = 10;
    real minDis = 1.9*1.9;

    int *NN = new int[N];
    int *NL = new int[N*M];
    for (int i = 0; i < N; ++i)
    {
        NN[i] = 0;
        for (int j = 0; j < M; ++j)
        {
            NL[i*M + j] = -1;
        }
    }

    int *d_NN, *d_NL;
    CHECK(cudaMalloc(&d_NN, N*sizeof(int)));
    CHECK(cudaMalloc(&d_NL, N*M*sizeof(int)));
    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, N*sizeof(real)));
    CHECK(cudaMalloc(&d_y, N*sizeof(real)));

    cppClockStart

    find_neighbor(NN, NL, x.data(), y.data(), N, M, minDis);
    // write_data(fout, NL, N, M);
    cppClockCurr

    cudaClockStart

    CHECK(cudaMemcpy(d_x, x.data(), N*sizeof(real), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_y, y.data(), N*sizeof(real), cudaMemcpyDefault));

    int block_size = 128;
    int grid_size = (N + block_size - 1)/block_size;
    find_neighbor_atomic<<<grid_size, block_size>>>(d_NN, d_NL, d_x, d_y, N, M, minDis);

    CHECK(cudaMemcpy(NN, d_NN, N*sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(NL, d_NL, N*M*sizeof(int), cudaMemcpyDefault));
    // write_data(fout, NL, N, M);

    cudaClockCurr

    CHECK(cudaMemcpy(d_x, x.data(), N*sizeof(real), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_y, y.data(), N*sizeof(real), cudaMemcpyDefault));
    find_neighbor_gpu<<<grid_size, block_size>>>(d_NN, d_NL, d_x, d_y, N, M, minDis);
    CHECK(cudaMemcpy(NN, d_NN, N*sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(NL, d_NL, N*M*sizeof(int), cudaMemcpyDefault));

    cudaClockCurr    

    write_data(fout, NL, N, M);

    delete[] NN;
    delete[] NL;
    CHECK(cudaFree(d_NN));
    CHECK(cudaFree(d_NL));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    return 0;
}


void find_neighbor(int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real minDis)
{
    for (int i = 0; i < N; ++i)
    {
        NN[i] = 0;
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy * dy;
            if (dis < minDis) // 比较平方，减少计算量。
            {
                NL[i*M + NN[i]] = j;  // 一维数组存放二维数据。
                NN[i] ++;
                NL[j*M + NN[j]] = i;  // 省去一般的判断。
                NN[j]++;
            }
        }
    }
}

__global__ void find_neighbor_gpu (int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real minDis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int count = 0;  // 寄存器变量，减少对全局变量NN的访问。
        for (int j = 0; j < N; ++j)  // 访问次数 N*N，性能降低。
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy * dy;

            if (dis < minDis && i != j) // 距离判断优先，提高“假”的命中率。
            {
                // 修改了全局内存NL的数据排列方式，实现合并访问（i 与 threadIdx.x的变化步调一致）。
                // ???
                NL[(count++) * N + i] = j;
            }
        }

        NN[i] = count;
    }
}

__global__ void find_neighbor_atomic(int *NN, int *NL, const real *x, const real *y, 
    const int N, const int M, 
    const real minDis)
{
    // 将 cpu 版本的第一层循环展开，一个线程对应一个原子操作。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        NN[i] = 0;

        for (int j = i + 1; j < N; ++j)
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy*dy;
            if (dis < minDis)
            {
                // 原子函数提高的性能，但是在NL中产生了一定的随机性，不便于后期调试。
                int old_i_num = atomicAdd(&NN[i], 1);  // 返回值为旧值，当前线程对应点的邻居数
                NL[i*M + old_i_num] = j;  // 当前线程对应点的新邻居
                int old_j_num = atomicAdd(&NN[j], 1);  // 返回值为旧值，当前邻居点的邻居数
                NL[j*M + old_j_num] = i;  // 当前邻居点的新邻居
            }
        }
    }
}

void read_data(const std::string &fstr, std::vector<real> &x, std::vector<real> &y)
{
    x.clear();
    y.clear();

    std::fstream reader(fstr, std::ios::in);
    if (!reader.is_open())
    {
        std::cerr << "data file open failed.\n";
        return;
    }

    std::regex re{"[\\s,]+"};
    std::string line;
    while(std::getline(reader, line))
    {
       std::vector<std::string> arr{std::sregex_token_iterator(line.begin(), line.end(), re, -1), 
           std::sregex_token_iterator()};

		if (arr.size() < 2 || arr[0].find("#") != std::string::npos)
		{
			continue;
		}

		x.push_back(stod(arr[0]));
		y.push_back(stod(arr[1]));        
    }
}

void write_data(const std::string &fstr, const int *NL, const int N, const int M)
{
    std::fstream writer(fstr, std::ios::out);
    if (!writer.is_open())
    {
        std::cerr << "result file open failed.\n";
        return;
    }

    for (int i = 0; i < N; ++i)
    {
        writer << i << "\t";
        for (int j = 0; j < M; ++j)
        {
            int ind = NL[i*M + j];
            if (ind >= 0) 
            {
                writer << ind << "\t";
            }
        }

        writer << endl;
    }
}








