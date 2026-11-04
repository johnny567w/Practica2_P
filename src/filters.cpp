#include "filters.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

Mask makeMask(FilterType type, int k) {
    Mask m;
    m.type = type;
    switch (type) {
        case FilterType::Average:
            if (k < 1 || k % 2 == 0) {
                fprintf(stderr, "El tamaño de la máscara de media debe ser impar (k=%d)\n", k);
                std::exit(1);
            }
            m.k = k;
            m.coef.assign((size_t)k * k, 1.0f / (float)(k * k));
            break;
        case FilterType::Sharpen:
            m.k = 3;
            m.coef = {0, -1, 0, -1, 5, -1, 0, -1, 0};
            break;
        case FilterType::Laplacian:
            m.k = 3;
            m.coef = {0, 1, 0, 1, -4, 1, 0, 1, 0};
            break;
    }
    return m;
}

Mask makeCustomMask(int k, const std::vector<float>& coef) {
    Mask m;
    m.type = FilterType::Sharpen;  
    m.k = k;
    m.coef = coef;
    return m;
}

const char* filterName(FilterType type) {
    switch (type) {
        case FilterType::Average: return "media";
        case FilterType::Sharpen: return "nitidez";
        case FilterType::Laplacian: return "laplaciano";
    }
    return "?";
}

bool parseFilter(const std::string& name, FilterType& type) {
    if (name == "media" || name == "average" || name == "mean") type = FilterType::Average;
    else if (name == "nitidez" || name == "sharpen") type = FilterType::Sharpen;
    else if (name == "laplaciano" || name == "laplacian") type = FilterType::Laplacian;
    else return false;
    return true;
}

double sumAbsCoef(const Mask& m) {
    double s = 0;
    for (float c : m.coef) s += std::fabs((double)c);
    return s;
}

unsigned char toDisplayValue(float v, FilterType type) {
    if (type == FilterType::Laplacian) v = std::fabs(v);
    float r = std::floor(v + 0.5f);
    if (r < 0.f) r = 0.f;
    if (r > 255.f) r = 255.f;
    return (unsigned char)r;
}

void toDisplay(const std::vector<float>& values, FilterType type, std::vector<unsigned char>& out) {
    out.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = toDisplayValue(values[i], type);
}
