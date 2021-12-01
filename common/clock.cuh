#include <chrono>

using namespace std::chrono;


#define cudaClockStart                                                             \
    float elapsed_time = 0;                                                        \
    float curr_time = 0;                                                           \
    cudaEvent_t start, stop;                                                       \
    CHECK(cudaEventCreate(&start));                                                \
    CHECK(cudaEventCreate(&stop));                                                 \
    CHECK(cudaEventRecord(start));                                                 \
    cudaEventQuery(start);                                                         \
   
   
#define cudaClockCurr                                                              \
    CHECK(cudaEventRecord(stop));                                                  \
    CHECK(cudaEventSynchronize(stop));                                             \
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));                          \
    printf("cuda time cost: %f ms.\n", curr_time - elapsed_time);                  \
    elapsed_time = curr_time;                                                      \


#define cppClockStart                                                              \
    double t = 0;                                                                  \
    auto tStart = system_clock::now();                                             \
    auto tStop  = system_clock::now();                                             \


#define cppClockCurr                                                               \
    tStop = system_clock::now();                                                   \
    t = duration<double, std::milli>(tStop - tStart).count();                      \
    printf("cpp time cost: %f ms.\n", t);                                          \
