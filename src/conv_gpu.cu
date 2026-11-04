#include "conv_gpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            fprintf(stderr, "Error CUDA %s en %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                                  \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

__global__ void convolveKernel(const float* __restrict__ in, float* __restrict__ out, int w, int h,
                               const float* __restrict__ mask, int k) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)w * h;
    if (idx >= n) return;

    const int row = (int)(idx / w);
    const int col = (int)(idx - (long long)row * w);
    const int r = k / 2;

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int y = row + i - r;
        if (y < 0 || y >= h) continue;
        for (int j = 0; j < k; ++j) {
            const int x = col + j - r;
            if (x < 0 || x >= w) continue;
            acc += mask[i * k + j] * in[(size_t)y * w + x];
        }
    }
    out[idx] = acc;
}

GpuConvolver::GpuConvolver(size_t maxPixels, int maxK) : maxPixels_(maxPixels), maxK_(maxK) {
    CUDA_CHECK(cudaMalloc(&dIn_, maxPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut_, maxPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMask_, (size_t)maxK * maxK * sizeof(float)));
    for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));
}

GpuConvolver::~GpuConvolver() {
    for (auto& e : ev_) cudaEventDestroy(e);
    cudaFree(dIn_);
    cudaFree(dOut_);
    cudaFree(dMask_);
}

void GpuConvolver::poisonOutput() {
    CUDA_CHECK(cudaMemset(dOut_, 0xFF, maxPixels_ * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

GpuTimes GpuConvolver::run(const float* hostIn, float* hostOut, int w, int h, const float* hostMask, int k,
                           int threadsPerBlock) {
    const size_t n = (size_t)w * h;
    if (n > maxPixels_ || k > maxK_) {
        fprintf(stderr, "GpuConvolver: imagen o máscara mayores que la memoria reservada\n");
        std::exit(1);
    }
    const size_t bytes = n * sizeof(float);
    const size_t maskBytes = (size_t)k * k * sizeof(float);
    GpuTimes t;
    t.blocks = (int)((n + threadsPerBlock - 1) / threadsPerBlock);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto w0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(ev_[0]));
    CUDA_CHECK(cudaMemcpy(dIn_, hostIn, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMask_, hostMask, maskBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_[1]));
    convolveKernel<<<t.blocks, threadsPerBlock>>>(dIn_, dOut_, w, h, dMask_, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_[2]));
    CUDA_CHECK(cudaMemcpy(hostOut, dOut_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_[3]));
    CUDA_CHECK(cudaEventSynchronize(ev_[3]));
    auto w1 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventElapsedTime(&t.h2dMs, ev_[0], ev_[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, ev_[1], ev_[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2hMs, ev_[2], ev_[3]));
    CUDA_CHECK(cudaEventElapsedTime(&t.totalMs, ev_[0], ev_[3]));
    t.wallMs = std::chrono::duration<double, std::milli>(w1 - w0).count();
    return t;
}

void gpuInit() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
}

std::string gpuInfoJson() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"multiprocessors\": %d, "
             "\"global_memory_mb\": %zu, \"max_threads_per_block\": %d, \"max_threads_per_sm\": %d, "
             "\"warp_size\": %d, \"l2_cache_kb\": %d, \"memory_bus_bits\": %d, "
             "\"driver_cuda_version\": \"%d.%d\", \"runtime_cuda_version\": \"%d.%d\"}",
             p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / (1024 * 1024),
             p.maxThreadsPerBlock, p.maxThreadsPerMultiProcessor, p.warpSize, p.l2CacheSize / 1024,
             p.memoryBusWidth, drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    return buf;
}
