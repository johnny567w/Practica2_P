#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

inline double stddevSample(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double mean = 0;
    for (double x : v) mean += x;
    mean /= v.size();
    double s = 0;
    for (double x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / (v.size() - 1));
}
