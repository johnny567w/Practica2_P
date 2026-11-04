
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "image_io.h"

int main(int argc, char** argv) {
    std::string outDir = "data", source;
    std::vector<int> sizes = {6000, 4000, 2000};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (a == "--tamanos" && i + 1 < argc) {
            sizes.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) sizes.push_back(std::atoi(tok.c_str()));
        } else source = a;
    }
    if (source.empty()) {
        fprintf(stderr, "Uso: prepare_data [--out data] [--tamanos 6000,4000,2000] fuente.bmp\n");
        return 1;
    }
    GrayImage src;
    if (!readImage(source, src)) return 1;
    const int srcW = src.width, srcH = src.height, side = std::min(srcW, srcH);
    const int x0 = (src.width - side) / 2, y0 = (src.height - side) / 2;
    printf("Fuente %s: %dx%d -> recorte cuadrado centrado de %dx%d en (x=%d, y=%d)\n", source.c_str(), src.width,
           src.height, side, side, x0, y0);
    for (int s : sizes)
        if (s > side) {
            fprintf(stderr, "La foto es demasiado pequeña: el lado menor (%d) no alcanza para %dx%d sin ampliar\n", side, s, s);
            return 1;
        }
    const GrayImage square = cropImage(src, x0, y0, side, side);
    src = GrayImage();

    std::string json = "{\n  \"fuente\": \"" + source.substr(source.find_last_of("/\\") + 1) + "\",\n";
    char buf[256];
    snprintf(buf, sizeof(buf), "  \"fuente_ancho\": %d, \"fuente_alto\": %d,\n  \"recorte\": {\"x\": %d, \"y\": %d, \"lado\": %d},\n",
             srcW, srcH, x0, y0, side);
    json += buf;
    json += "  \"metodo\": \"promedio de area (resizeArea)\",\n  \"versiones\": [";
    for (size_t i = 0; i < sizes.size(); ++i) {
        const int s = sizes[i];
        GrayImage v = s == side ? square : resizeArea(square, s, s);
        const std::string name = "img_" + std::to_string(s);
        if (!writePGM(outDir + "/" + name + ".pgm", v)) return 1;
        writePNG(outDir + "/preview_" + name + ".png", downscaleBox(v, 800));
        const long long n = (long long)s * s;
        printf("  %s/%s.pgm: %dx%d = %lld píxeles (escala %.4f)\n", outDir.c_str(), name.c_str(), s, s, n, (double)s / side);
        for (int tpb : {128, 256, 512}) {
            long long rest = n % tpb;
            printf("      %3d hilos/bloque: %lld bloques%s\n", tpb, (n + tpb - 1) / tpb,
                   rest ? (", NO es múltiplo: el último bloque usa " + std::to_string(rest) + " de " + std::to_string(tpb) + " hilos").c_str()
                        : ", múltiplo exacto");
        }
        snprintf(buf, sizeof(buf), "%s\n    {\"nombre\": \"%s\", \"lado\": %d, \"escala\": %.6f}", i ? "," : "", name.c_str(), s,
                 (double)s / side);
        json += buf;
    }
    json += "\n  ]\n}\n";
    FILE* f = fopen((outDir + "/preparacion.json").c_str(), "w");
    if (!f) return 1;
    fputs(json.c_str(), f);
    fclose(f);
    return 0;
}
