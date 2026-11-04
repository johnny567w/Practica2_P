#include "validation.h"

#include <cmath>

ErrorStats compareOutputs(const float* ref, const float* test, size_t n) {
    ErrorStats s;
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(test[i])) {
            ++s.nonFinite;
            continue;
        }
        double e = std::fabs((double)ref[i] - (double)test[i]);
        sum += e;
        if (e > s.maxAbs) s.maxAbs = e;
    }
    s.meanAbs = n ? sum / (double)n : 0;
    return s;
}

double toleranceFor(const Mask& m, double maxInput) {
    const double u = std::ldexp(1.0, -24);
    const double nk = (double)m.k * m.k;
    const double gamma = nk * u / (1.0 - nk * u);
    return 2.0 * gamma * maxInput * sumAbsCoef(m);
}

size_t countDisplayDiffs(const float* ref, const float* test, size_t n, FilterType type) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i)
        if (toDisplayValue(ref[i], type) != toDisplayValue(test[i], type)) ++d;
    return d;
}
