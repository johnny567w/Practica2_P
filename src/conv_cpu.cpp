#include "conv_cpu.h"

#include <cstddef>

void convolveCPU(const float* in, float* out, int w, int h, const float* mask, int k) {
    const int r = k / 2;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                const int y = row + i - r;
                if (y < 0 || y >= h) continue;  // fila fuera de la imagen no  aporta 
                for (int j = 0; j < k; ++j) {
                    const int x = col + j - r;
                    if (x < 0 || x >= w) continue;  // columna fuera de la imagen tampoco aporta
                    acc += mask[i * k + j] * in[(size_t)y * w + x];
                }
            }
            out[(size_t)row * w + col] = acc;
        }
    }
}
