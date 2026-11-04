#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <string>

struct GpuTimes {
    float h2dMs = 0;
    float kernelMs = 0;
    float d2hMs = 0;
    float totalMs = 0;
    double wallMs = 0;
    int blocks = 0;
};

class GpuConvolver {
public:
    GpuConvolver(size_t maxPixels, int maxK);
    ~GpuConvolver();
    GpuConvolver(const GpuConvolver&) = delete;
    GpuConvolver& operator=(const GpuConvolver&) = delete;

    GpuTimes run(const float* hostIn, float* hostOut, int w, int h, const float* hostMask, int k, int threadsPerBlock);

    void poisonOutput();

private:
    float* dIn_ = nullptr;
    float* dOut_ = nullptr;
    float* dMask_ = nullptr;
    size_t maxPixels_;
    int maxK_;
    cudaEvent_t ev_[4];
};

void gpuInit();
std::string gpuInfoJson();
