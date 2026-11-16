
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
