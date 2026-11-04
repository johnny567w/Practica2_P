// Comparación numérica CPU vs GPU antes de convertir a [0, 255].
#pragma once
#include <cstddef>
#include <vector>

#include "filters.h"

struct ErrorStats {
    double maxAbs = 0;    // error absoluto máximo
    double meanAbs = 0;   // error absoluto medio
    size_t nonFinite = 0; 
};

ErrorStats compareOutputs(const float* ref, const float* test, size_t n);

double toleranceFor(const Mask& m, double maxInput = 255.0);

// Número de píxeles cuyo valor visualizado (8 bits) difiere entre las dos salidas.
size_t countDisplayDiffs(const float* ref, const float* test, size_t n, FilterType type);
