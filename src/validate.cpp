#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "conv_cpu.h"
#include "conv_gpu.h"
#include "filters.h"
#include "image_io.h"
#include "validation.h"

namespace {

FILE* logFile = nullptr;
FILE* csvCases = nullptr;
FILE* csvMatrix = nullptr;
int failures = 0;

void out(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vprintf(fmt, a);
    va_end(a);
    if (logFile) {
        va_start(a, fmt);
        vfprintf(logFile, fmt, a);
        va_end(a);
    }
}

void check(bool ok, const char* what) {
    out("    [%s] %s\n", ok ? "OK" : "FALLA", what);
    if (!ok) ++failures;
}

void printMatrix(const char* title, const std::vector<float>& m, int w, int h) {
    out("  %s\n", title);
    for (int y = 0; y < h; ++y) {
        out("    ");
        for (int x = 0; x < w; ++x) out("%10.4f", m[(size_t)y * w + x]);
        out("\n");
    }
}

bool recordCase(int test, const std::string& source, int w, int h, const Mask& m, const char* maskName, int tpb,
                int blocks, const ErrorStats& s, double tol) {
    const long long n = (long long)w * h;
    const bool ok = s.nonFinite == 0 && s.maxAbs <= tol;
    if (csvCases)
        fprintf(csvCases, "%d,%s,%dx%d,%lld,%s,%dx%d,%d,%d,%lld,%s,%.9g,%.9g,%.9g,%zu,%s\n", test, source.c_str(), w, h, n,
                maskName, m.k, m.k, tpb, blocks, n - (long long)(blocks - 1) * tpb, n % tpb ? "no" : "si", s.maxAbs,
                s.meanAbs, tol, s.nonFinite, ok ? "OK" : "FALLA");
    return ok;
}

struct Expected {
    int row, col;
    double value;
    const char* how;
};

struct ManualCase {
    const char* name;
    const char* title;
    Mask mask;
    std::vector<Expected> expected;
};

void runSmallMatrix(GpuConvolver& gpu, const std::string& source, const std::vector<float>& in, int w, int h,
                    const std::vector<ManualCase>& cases) {
    printMatrix("Entrada:", in, w, h);
    float maxIn = 1.f;
    for (float v : in) maxIn = std::max(maxIn, std::fabs(v));
    for (const ManualCase& c : cases) {
        out("\n  -- %s --\n", c.title);
        std::vector<float> cpu((size_t)w * h), gpuOut((size_t)w * h);
        convolveCPU(in.data(), cpu.data(), w, h, c.mask.coef.data(), c.mask.k);
        gpu.poisonOutput();
        GpuTimes t = gpu.run(in.data(), gpuOut.data(), w, h, c.mask.coef.data(), c.mask.k, 128);
        printMatrix("Salida CPU (antes de convertir a [0,255]):", cpu, w, h);
        printMatrix("Salida GPU (antes de convertir a [0,255]):", gpuOut, w, h);
        for (const Expected& e : c.expected) {
            float vc = cpu[e.row * w + e.col], vg = gpuOut[e.row * w + e.col];
            out("    (%d,%d): manual %.4f | CPU %.4f | GPU %.4f   <- %s\n", e.row, e.col, e.value, vc, vg, e.how);
            char what[128];
            snprintf(what, sizeof(what), "posición (%d,%d) coincide con el cálculo manual", e.row, e.col);
            check(std::fabs(vc - e.value) < 1e-4 && std::fabs(vg - e.value) < 1e-4, what);
        }
        ErrorStats s = compareOutputs(cpu.data(), gpuOut.data(), cpu.size());
        double tol = toleranceFor(c.mask, maxIn);
        out("    CPU vs GPU en toda la matriz: error máx %.3g, medio %.3g, no escritos %zu\n", s.maxAbs, s.meanAbs,
            s.nonFinite);
        check(recordCase(1, source, w, h, c.mask, c.name, 128, t.blocks, s, tol), "GPU = CPU dentro de la tolerancia");
        if (csvMatrix)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    fprintf(csvMatrix, "%s,%s,%d,%d,%.9g,%.9g,%.9g\n", source.c_str(), c.name, y, x, in[y * w + x],
                            cpu[y * w + x], gpuOut[y * w + x]);
    }
}

std::vector<ManualCase> manualCases(bool withExpected) {
    std::vector<ManualCase> cases = {
        {"figura1", "Máscara de la Figura 1 [-2 -1 0; -1 1 1; 0 1 2]",
         makeCustomMask(3, {-2, -1, 0, -1, 1, 1, 0, 1, 2}),
         {{1, 1, 78, "35(-2)+40(-1)+41(0)+40(-1)+40(1)+42(1)+42(0)+46(1)+50(2) = 78"},
          {1, 2, 87, "40(-2)+41(-1)+45(0)+40(-1)+42(1)+46(1)+46(0)+50(1)+55(2) = 87"},
          {0, 0, 195, "fila -1 = 0; 0(-1)+35(1)+40(1) + 0(0)+40(1)+40(2) = 195 (zero-padding)"},
          {3, 4, -164, "55(-2)+55(-1)+0(0) + 60(-1)+61(1)+0(1) + fila 4 = 0 -> -164"}}},
        {"media3x3", "Media 3x3 (1/9)", makeMask(FilterType::Average, 3),
         {{1, 1, 376.0 / 9.0, "(35+40+41+40+40+42+42+46+50)/9 = 376/9 = 41.778"},
          {0, 0, 155.0 / 9.0, "(35+40+40+40)/9 = 155/9 = 17.222 (zero-padding)"}}},
        {"nitidez", "Nitidez [0 -1 0; -1 5 -1; 0 -1 0]", makeMask(FilterType::Sharpen),
         {{1, 1, 32, "5(40) - (40+40+42+46) = 200 - 168 = 32"},
          {0, 0, 95, "5(35) - (0+0+40+40) = 175 - 80 = 95"}}},
        {"laplaciano", "Laplaciano [0 1 0; 1 -4 1; 0 1 0]", makeMask(FilterType::Laplacian),
         {{1, 1, 8, "(40+40+42+46) - 4(40) = 168 - 160 = 8"},
          {0, 0, -60, "(0+0+40+40) - 4(35) = 80 - 140 = -60"}}},
    };
    if (!withExpected)
        for (ManualCase& c : cases) c.expected.clear();
    return cases;
}

void manualTest(GpuConvolver& gpu) {
    out("== Prueba 1: cálculo manual (matriz 4x5, valores de la Figura 1) ==\n");

    const std::vector<float> in = {35, 40, 41, 45, 50,
                                   40, 40, 42, 46, 52,
                                   42, 46, 50, 55, 55,
                                   48, 52, 58, 60, 61};
    runSmallMatrix(gpu, "figura1", in, 5, 4, manualCases(true));
}

bool readMatrix(const std::string& path, std::vector<float>& m, int& w, int& h) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s\n", path.c_str());
        return false;
    }
    m.clear();
    w = h = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        const bool semicolon = line.find(';') != std::string::npos || line.find('\t') != std::string::npos;
        for (char& c : line) {
            if (semicolon) {
                if (c == ';' || c == '\t') c = ' ';
                else if (c == ',') c = '.';
            } else if (c == ',') {
                c = ' ';
            }
        }
        std::istringstream ss(line);
        std::vector<float> row;
        std::string tok;
        while (ss >> tok) {
            char* end = nullptr;
            float v = std::strtof(tok.c_str(), &end);
            if (*end != '\0') {
                fprintf(stderr, "%s: valor no numérico '%s' en la fila %d\n", path.c_str(), tok.c_str(), h + 1);
                return false;
            }
            row.push_back(v);
        }
        if (row.empty()) continue;
        if (w == 0) w = (int)row.size();
        if ((int)row.size() != w) {
            fprintf(stderr, "%s: la fila %d tiene %zu valores y la primera %d\n", path.c_str(), h + 1, row.size(), w);
            return false;
        }
        m.insert(m.end(), row.begin(), row.end());
        ++h;
    }
    if (w == 0) fprintf(stderr, "%s: no contiene valores\n", path.c_str());
    return w > 0;
}

void customMatrixTest(GpuConvolver& gpu, const std::string& path) {
    out("\n== Prueba 1b: matriz propia (%s) ==\n", path.c_str());
    std::vector<float> in;
    int w, h;
    if (!readMatrix(path, in, w, h)) {
        ++failures;
        return;
    }
    out("  %d filas x %d columnas\n", h, w);
    runSmallMatrix(gpu, "propia", in, w, h, manualCases(false));
}

void nonMultipleTest(GpuConvolver& gpu) {
    out("\n== Prueba 2: dimensiones que no son múltiplo de los hilos por bloque ==\n");
    struct Dim {
        int w, h;
    };
    const Dim dims[] = {{1001, 777}, {333, 1}, {1, 517}, {4097, 3}, {2000, 2001}};
    const int threads[] = {128, 256, 512};
    std::vector<Mask> masks = {makeMask(FilterType::Average, 3), makeMask(FilterType::Average, 9),
                               makeMask(FilterType::Average, 21), makeMask(FilterType::Sharpen),
                               makeMask(FilterType::Laplacian)};
    out("  %-11s %9s %-10s %-6s %6s %8s %12s %12s %12s %6s\n", "dimensión", "píxeles", "filtro", "másc.", "hilos",
        "bloques", "err. máx", "err. medio", "tolerancia", "res.");
    uint32_t seed = 12345;
    for (const Dim& d : dims) {
        const size_t n = (size_t)d.w * d.h;
        std::vector<float> in(n), cpu(n), gpuOut(n);
        for (auto& v : in) {
            seed = seed * 1664525u + 1013904223u;
            v = (float)(seed >> 24);
        }
        for (const Mask& m : masks) {
            convolveCPU(in.data(), cpu.data(), d.w, d.h, m.coef.data(), m.k);
            for (int tpb : threads) {
                gpu.poisonOutput();
                GpuTimes t = gpu.run(in.data(), gpuOut.data(), d.w, d.h, m.coef.data(), m.k, tpb);
                ErrorStats s = compareOutputs(cpu.data(), gpuOut.data(), n);
                double tol = toleranceFor(m);
                bool ok = recordCase(2, "aleatoria", d.w, d.h, m, filterName(m.type), tpb, t.blocks, s, tol);
                if (!ok) ++failures;
                char dim[32];
                snprintf(dim, sizeof(dim), "%dx%d", d.w, d.h);
                out("  %-11s %9zu %-10s %2dx%-3d %6d %8d %12.3g %12.3g %12.3g %6s%s\n", dim, n, filterName(m.type), m.k,
                    m.k, tpb, t.blocks, s.maxAbs, s.meanAbs, tol, ok ? "OK" : "FALLA",
                    n % tpb ? "" : "  (múltiplo)");
            }
        }
    }
}

void realImagesTest(const std::vector<std::string>& paths) {
    out("\n== Prueba 3: imágenes de los experimentos (2000x2000, 4000x4000, 6000x6000) ==\n");
    const int threads[] = {128, 256, 512};
    for (const std::string& path : paths) {
        GrayImage img;
        if (!readPGM(path, img)) {
            out("  %s no existe: se omite (ejecute scripts\\prepare_images.ps1)\n", path.c_str());
            continue;
        }
        const int w = img.width, h = img.height;
        const size_t n = (size_t)w * h;
        std::string name = path.substr(path.find_last_of("/\\") + 1);
        name = name.substr(0, name.find_last_of('.'));
        out("\n  %s: %dx%d = %zu píxeles\n", name.c_str(), w, h, n);
        for (int tpb : threads) {
            size_t blocks = (n + tpb - 1) / tpb;
            if (n % tpb == 0)
                out("    %3d hilos/bloque: %zu bloques, múltiplo exacto (todos los bloques completos)\n", tpb, blocks);
            else
                out("    %3d hilos/bloque: %zu bloques, NO es múltiplo: el último bloque usa %zu de %d hilos y %zu "
                    "salen por el if\n",
                    tpb, blocks, n % tpb, tpb, tpb - n % tpb);
        }
        std::vector<float> in(img.pixels.begin(), img.pixels.end()), cpu(n), gpuOut(n);
        img.pixels.clear();
        img.pixels.shrink_to_fit();
        GpuConvolver gpu(n, 3);
        out("  %-10s %-6s %6s %8s %12s %12s %12s %10s %6s\n", "filtro", "másc.", "hilos", "bloques", "err. máx",
            "err. medio", "tolerancia", "no escr.", "res.");
        for (FilterType type : {FilterType::Average, FilterType::Sharpen, FilterType::Laplacian}) {
            Mask m = makeMask(type, 3);
            convolveCPU(in.data(), cpu.data(), w, h, m.coef.data(), m.k);
            for (int tpb : threads) {
                gpu.poisonOutput();
                GpuTimes t = gpu.run(in.data(), gpuOut.data(), w, h, m.coef.data(), m.k, tpb);
                ErrorStats s = compareOutputs(cpu.data(), gpuOut.data(), n);
                double tol = toleranceFor(m);
                bool ok = recordCase(3, name, w, h, m, filterName(type), tpb, t.blocks, s, tol);
                if (!ok) ++failures;
                out("  %-10s %2dx%-3d %6d %8d %12.3g %12.3g %12.3g %10zu %6s%s\n", filterName(type), m.k, m.k, tpb,
                    t.blocks, s.maxAbs, s.meanAbs, tol, s.nonFinite, ok ? "OK" : "FALLA",
                    n % tpb ? "" : "  (múltiplo)");
            }
        }
    }
}

std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(tok);
    return v;
}

}

int main(int argc, char** argv) {
    std::string logPath = "results/validacion_pruebas.txt", matrixPath;
    std::vector<std::string> images = {"data/img_2000.pgm", "data/img_4000.pgm", "data/img_6000.pgm"};
    for (int i = 1; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if (a == "--log") logPath = argv[++i];
        else if (a == "--matriz") matrixPath = argv[++i];
        else if (a == "--imagenes") images = splitList(argv[++i]);
    }
    std::string dir = logPath.substr(0, logPath.find_last_of("/\\") + 1);
    logFile = fopen(logPath.c_str(), "w");
    if (!logFile) fprintf(stderr, "Aviso: no se pudo crear %s; solo se imprimirá en consola\n", logPath.c_str());
    csvCases = fopen((dir + "validacion_pruebas.csv").c_str(), "w");
    csvMatrix = fopen((dir + "matriz_salida.csv").c_str(), "w");
    if (csvCases)
        fprintf(csvCases,
                "prueba,origen,dimension,pixeles,filtro,mascara,hilos_por_bloque,bloques,hilos_ultimo_bloque,"
                "multiplo,error_abs_max,error_abs_medio,tolerancia,pixeles_no_escritos,resultado\n");
    if (csvMatrix) fprintf(csvMatrix, "matriz,mascara,fila,columna,entrada,cpu,gpu\n");

    gpuInit();
    {
        GpuConvolver gpu((size_t)2000 * 2001, 21);
        manualTest(gpu);
        if (!matrixPath.empty()) customMatrixTest(gpu, matrixPath);
        nonMultipleTest(gpu);
    }
    realImagesTest(images);

    out("\nResultado: %s (%d fallas)\n", failures ? "HAY FALLAS" : "TODAS LAS PRUEBAS PASARON", failures);
    if (logFile) fclose(logFile);
    if (csvCases) fclose(csvCases);
    if (csvMatrix) fclose(csvMatrix);
    return failures ? 1 : 0;
}
