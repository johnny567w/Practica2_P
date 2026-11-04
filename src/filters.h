// Máscaras de convolución y conversión al rango de visualización [0, 255].
#pragma once
#include <string>
#include <vector>

enum class FilterType { Average, Sharpen, Laplacian };

struct Mask {
    FilterType type;
    int k;                    // lado de la mascara
    std::vector<float> coef;  // k*k coeficientes en orden fila-mayor
};

// Media k x k: todos los coeficientes valen 1/(k*k).
// Nitidez 3x3:  
// Laplaciano 3x3:
Mask makeMask(FilterType type, int k = 3);
Mask makeCustomMask(int k, const std::vector<float>& coef);  // para pruebas (p. ej. la máscara de la Figura 1)

const char* filterName(FilterType type);
bool parseFilter(const std::string& name, FilterType& type);
double sumAbsCoef(const Mask& m);


unsigned char toDisplayValue(float v, FilterType type);
void toDisplay(const std::vector<float>& values, FilterType type, std::vector<unsigned char>& out);
