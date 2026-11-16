#define _CRT_SECURE_NO_WARNINGS
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "conv_cpu.h"
#include "conv_gpu.h"
#include "filters.h"
#include "image_io.h"

int main(int argc, char** argv) {
    std::string input, output, impl = "cuda", filter = "media", raw, floatOut, coefList;
    int k = 3, threads = 256;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--input") input = v;
        else if (a == "--output") output = v;
        else if (a == "--filter") filter = v;
        else if (a == "--k") k = std::atoi(v.c_str());
        else if (a == "--impl") impl = v;
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--raw") raw = v;
        else if (a == "--float") floatOut = v;   // salida en float32 sin recortar (para comparar con Python)
        else if (a == "--coef") coefList = v;    // mascara propia: k*k valores separados por comas
    }
    FilterType type;
    if (input.empty() || (output.empty() && floatOut.empty()) || !parseFilter(filter, type) ||
        (impl != "cpu" && impl != "cuda")) {
        fprintf(stderr,
                "Uso: filter --input in.pgm --output out.png --filter media|nitidez|laplaciano [--k 3] "
                "[--impl cpu|cuda] [--threads 256] [--float salida.f32] [--coef a,b,c,...]\n");
        return 1;
    }
    GrayImage img;
    if (!readImage(input, img)) return 1;
    Mask m = makeMask(type, k);
    if (!coefList.empty()) {
        std::vector<float> c;
        size_t pos = 0;
        while (pos <= coefList.size()) {
            size_t comma = coefList.find(',', pos);
            if (comma == std::string::npos) comma = coefList.size();
            c.push_back((float)std::atof(coefList.substr(pos, comma - pos).c_str()));
            pos = comma + 1;
        }
        int ck = 1;
        while (ck * ck < (int)c.size()) ++ck;
        if (ck * ck != (int)c.size()) {
            fprintf(stderr, "--coef necesita k*k valores (recibidos %zu)\n", c.size());
            return 1;
        }
        m = makeCustomMask(ck, c);
    }
    std::vector<float> in(img.pixels.begin(), img.pixels.end()), out(in.size());

    if (impl == "cpu") {
        auto t0 = std::chrono::steady_clock::now();
        convolveCPU(in.data(), out.data(), img.width, img.height, m.coef.data(), m.k);
        auto t1 = std::chrono::steady_clock::now();
        printf("CPU: %.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
        gpuInit();
        GpuConvolver gpu(in.size(), m.k);
        GpuTimes t = gpu.run(in.data(), out.data(), img.width, img.height, m.coef.data(), m.k, threads);
        printf("CUDA (%d bloques x %d hilos): H2D %.3f ms, kernel %.3f ms, D2H %.3f ms, total %.3f ms\n", t.blocks,
               threads, t.h2dMs, t.kernelMs, t.d2hMs, t.totalMs);
    }

    bool ok = true;
    if (!floatOut.empty()) {
        FILE* f = fopen(floatOut.c_str(), "wb");
        ok = f && fwrite(out.data(), sizeof(float), out.size(), f) == out.size();
        if (f) fclose(f);
        if (output.empty()) return ok ? 0 : 1;
    }
    GrayImage res;
    res.width = img.width;
    res.height = img.height;
    toDisplay(out, type, res.pixels);
    ok = writePNG(output, res) && ok;
    if (!raw.empty()) ok = writePGM(raw, res) && ok;
    printf("%s -> %s (%s %dx%d)\n", input.c_str(), output.c_str(), filterName(type), m.k, m.k);
    return ok ? 0 : 1;
}
