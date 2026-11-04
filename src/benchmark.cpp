#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "conv_cpu.h"
#include "conv_gpu.h"
#include "filters.h"
#include "image_io.h"
#include "stats.h"
#include "validation.h"

namespace {

struct Options {
    std::string dataDir = "data";
    std::string outDir = "results";
    std::vector<std::string> images = {"img_2000", "img_4000", "img_6000"};
    std::string base = "img_4000";
    int reps = 10;
    double warmupMs = 2000;
    std::set<int> experiments = {1, 2, 3, 4};
    bool saveImages = true;
    bool fullImages = false;
    int detailX = -1, detailY = -1;
    int baseWidth = 0;
};

struct Input {
    std::string name;
    GrayImage img;
    std::vector<float> f;
    std::string dims() const { return std::to_string(img.width) + "x" + std::to_string(img.height); }
    size_t pixels() const { return (size_t)img.width * img.height; }
};

const int kDefaultThreads = 256;

Options opt;
FILE* csvTimes = nullptr;
FILE* csvValid = nullptr;
std::map<std::string, Input> inputs;
GpuConvolver* gpu = nullptr;

Input& loadInput(const std::string& name) {
    auto it = inputs.find(name);
    if (it != inputs.end()) return it->second;
    Input in;
    in.name = name;
    std::string path = opt.dataDir + "/" + name + ".pgm";
    if (!readPGM(path, in.img)) {
        fprintf(stderr, "No se encontró %s. Ejecute primero scripts\\prepare_images.ps1\n", path.c_str());
        std::exit(1);
    }
    in.f.assign(in.img.pixels.begin(), in.img.pixels.end());
    return inputs.emplace(name, std::move(in)).first->second;
}

std::string maskLabel(const Mask& m) { return std::to_string(m.k) + "x" + std::to_string(m.k); }

void writeTime(int exp, const char* impl, const Mask& m, const Input& in, int tpb, int blocks, int rep,
               const char* metric, double ms) {
    fprintf(csvTimes, "%d,%s,%s,%s,%s,%s,%d,%d,%d,%s,%.6f,ms\n", exp, impl, filterName(m.type), in.name.c_str(),
            in.dims().c_str(), maskLabel(m).c_str(), tpb, blocks, rep, metric, ms);
}

std::string powerJson(bool warn) {
#ifdef _WIN32
    SYSTEM_POWER_STATUS s;
    if (GetSystemPowerStatus(&s)) {
        const bool ac = s.ACLineStatus == 1;
        const int pct = s.BatteryLifePercent;
        if (warn && (!ac || (pct != 255 && pct < 20)))
            fprintf(stderr,
                    "ADVERTENCIA: %s, batería al %d %%. El portátil puede limitar la potencia y alterar los "
                    "tiempos; conviene medir conectado y con la batería por encima del 20 %%.\n",
                    ac ? "conectado a la red" : "funcionando con batería", pct);
        char buf[96];
        snprintf(buf, sizeof(buf), "{\"ac\": %s, \"battery_pct\": %d}", ac ? "true" : "false", pct == 255 ? -1 : pct);
        return buf;
    }
#endif
    (void)warn;
    return "null";
}

struct CropPos {
    int x0, y0, s;
};
std::map<std::string, CropPos> cropPos;

const CropPos& detailPos(const Input& in) {
    auto it = cropPos.find(in.name);
    if (it != cropPos.end()) return it->second;
    const GrayImage& img = in.img;
    const int s = std::min({512, img.width, img.height}), B = 64;
    if (opt.detailX >= 0 && opt.baseWidth > 0) {
        const double k = (double)img.width / opt.baseWidth;
        const int x = std::clamp((int)(opt.detailX * k + 256 * (k - 1)), 0, img.width - s);
        const int y = std::clamp((int)(opt.detailY * k + 256 * (k - 1)), 0, img.height - s);
        return cropPos.emplace(in.name, CropPos{x, y, s}).first->second;
    }
    const int bw = img.width / B, bh = img.height / B, nb = s / B;
    CropPos best{(img.width - s) / 2, (img.height - s) / 2, s};
    if (s == 512 && bw >= nb && bh >= nb) {
        std::vector<double> e((size_t)bw * bh, 0.0);
        for (int y = 0; y < bh * B - 1; ++y)
            for (int x = 0; x < bw * B - 1; ++x) {
                const unsigned char* p = &img.pixels[(size_t)y * img.width + x];
                e[(size_t)(y / B) * bw + x / B] += std::abs(p[1] - p[0]) + std::abs(p[img.width] - p[0]);
            }
        double bestSum = -1;
        for (int by = 0; by + nb <= bh; ++by)
            for (int bx = 0; bx + nb <= bw; ++bx) {
                double sum = 0;
                for (int j = 0; j < nb; ++j)
                    for (int i = 0; i < nb; ++i) sum += e[(size_t)(by + j) * bw + bx + i];
                if (sum > bestSum) {
                    bestSum = sum;
                    best = {bx * B, by * B, s};
                }
            }
    }
    return cropPos.emplace(in.name, best).first->second;
}

GrayImage detailCrop(const GrayImage& img, const CropPos& p) { return cropImage(img, p.x0, p.y0, p.s, p.s); }

void saveOutput(const std::string& base, const std::vector<float>& values, FilterType type, const Input& in) {
    if (!opt.saveImages) return;
    GrayImage img;
    img.width = in.img.width;
    img.height = in.img.height;
    toDisplay(values, type, img.pixels);
    std::string dir = opt.outDir + "/images/";
    if (opt.fullImages) writePNG(dir + base + "_full.png", img);
    writePNG(dir + base + "_crop.png", detailCrop(img, detailPos(in)));
    writePNG(dir + base + "_preview.png", downscaleBox(img, 800));
}

void saveInput(const Input& in) {
    if (!opt.saveImages) return;
    std::string dir = opt.outDir + "/images/entrada_" + in.name;
    writePNG(dir + "_crop.png", detailCrop(in.img, detailPos(in)));
    writePNG(dir + "_preview.png", downscaleBox(in.img, 800));
}

double runCPU(int exp, const Mask& m, const Input& in, std::vector<float>& ref) {
    ref.assign(in.pixels(), 0.f);
    std::vector<double> times;
    for (int rep = 0; rep <= opt.reps; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        convolveCPU(in.f.data(), ref.data(), in.img.width, in.img.height, m.coef.data(), m.k);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        writeTime(exp, "cpu", m, in, 0, 0, rep, "cpu_total", ms);
        if (rep > 0) times.push_back(ms);
    }
    fflush(csvTimes);
    double med = median(times);
    printf("  [exp %d] CPU  %-10s %-5s %-13s %-9s          mediana %10.2f ms  (de %.2f)\n", exp, filterName(m.type),
           maskLabel(m).c_str(), in.name.c_str(), in.dims().c_str(), med, stddevSample(times));
    return med;
}

void runGPU(int exp, const Mask& m, const Input& in, int tpb, const std::vector<float>& ref, std::vector<float>& out) {
    const size_t n = in.pixels();
    const int w = in.img.width, h = in.img.height;
    out.assign(n, 0.f);
    std::vector<double> kernel, total;
    ErrorStats worst;
    double meanSum = 0;
    int blocks = 0;

    GpuTimes t;
    int warmRuns = 0;
    auto w0 = std::chrono::steady_clock::now();
    do {
        t = gpu->run(in.f.data(), out.data(), w, h, m.coef.data(), m.k, tpb);
        ++warmRuns;
    } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < opt.warmupMs);
    writeTime(exp, "cuda", m, in, tpb, t.blocks, 0, "h2d", t.h2dMs);
    writeTime(exp, "cuda", m, in, tpb, t.blocks, 0, "kernel", t.kernelMs);
    writeTime(exp, "cuda", m, in, tpb, t.blocks, 0, "d2h", t.d2hMs);
    writeTime(exp, "cuda", m, in, tpb, t.blocks, 0, "total", t.totalMs);
    writeTime(exp, "cuda", m, in, tpb, t.blocks, 0, "total_wall", t.wallMs);

    for (int rep = 1; rep <= opt.reps; ++rep) {
        gpu->poisonOutput();
        t = gpu->run(in.f.data(), out.data(), w, h, m.coef.data(), m.k, tpb);
        blocks = t.blocks;
        writeTime(exp, "cuda", m, in, tpb, blocks, rep, "h2d", t.h2dMs);
        writeTime(exp, "cuda", m, in, tpb, blocks, rep, "kernel", t.kernelMs);
        writeTime(exp, "cuda", m, in, tpb, blocks, rep, "d2h", t.d2hMs);
        writeTime(exp, "cuda", m, in, tpb, blocks, rep, "total", t.totalMs);
        writeTime(exp, "cuda", m, in, tpb, blocks, rep, "total_wall", t.wallMs);
        ErrorStats e = compareOutputs(ref.data(), out.data(), n);
        if (e.maxAbs > worst.maxAbs) worst.maxAbs = e.maxAbs;
        if (e.nonFinite > worst.nonFinite) worst.nonFinite = e.nonFinite;
        meanSum += e.meanAbs;
        kernel.push_back(t.kernelMs);
        total.push_back(t.totalMs);
    }
    fflush(csvTimes);
    worst.meanAbs = meanSum / opt.reps;
    double tol = toleranceFor(m);
    size_t diff8 = countDisplayDiffs(ref.data(), out.data(), n, m.type);
    bool pass = worst.nonFinite == 0 && worst.maxAbs <= tol;
    fprintf(csvValid, "%d,%s,%s,%s,%s,%d,%d,%.9g,%.9g,%.9g,%zu,%zu,%s\n", exp, filterName(m.type), in.name.c_str(),
            in.dims().c_str(), maskLabel(m).c_str(), tpb, blocks, worst.maxAbs, worst.meanAbs, tol, worst.nonFinite,
            diff8, pass ? "OK" : "FALLA");
    fflush(csvValid);
    printf("  [exp %d] CUDA %-10s %-5s %-13s %-9s %3d hilos mediana kernel %8.3f ms (de %.3f), total %8.2f ms | "
           "err max %.3g (tol %.3g) %s [%d calent.]\n",
           exp, filterName(m.type), maskLabel(m).c_str(), in.name.c_str(), in.dims().c_str(), tpb, median(kernel),
           stddevSample(kernel), median(total), worst.maxAbs, tol, pass ? "OK" : "FALLA", warmRuns);
    if (!pass) fprintf(stderr, "  ADVERTENCIA: la validación numérica falló en esta configuración\n");
}

std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Falta valor para %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--data") opt.dataDir = next();
        else if (a == "--out") opt.outDir = next();
        else if (a == "--imagenes") opt.images = splitList(next());
        else if (a == "--base") opt.base = next();
        else if (a == "--reps") opt.reps = std::atoi(next().c_str());
        else if (a == "--warmup-ms") opt.warmupMs = std::atof(next().c_str());
        else if (a == "--no-images") opt.saveImages = false;
        else if (a == "--full-images") opt.fullImages = true;
        else if (a == "--detalle") {
            std::vector<std::string> xy = splitList(next());
            if (xy.size() == 2) opt.detailX = std::atoi(xy[0].c_str()), opt.detailY = std::atoi(xy[1].c_str());
        }
        else if (a == "--exp") {
            opt.experiments.clear();
            for (const std::string& e : splitList(next())) opt.experiments.insert(std::atoi(e.c_str()));
        } else {
            fprintf(stderr,
                    "Uso: benchmark [--data dir] [--out dir] [--imagenes img_2000,img_4000,img_6000] [--base img_4000] [--reps N] "
                    "[--warmup-ms ms] [--exp 1,2,3,4] [--detalle x,y] [--no-images] [--full-images]\n");
            std::exit(1);
        }
    }
    if (opt.reps < 1) opt.reps = 1;
}

}

int main(int argc, char** argv) {
    parseArgs(argc, argv);
    gpuInit();

    std::vector<Input*> imgs;
    for (const std::string& name : opt.images) imgs.push_back(&loadInput(name));
    std::sort(imgs.begin(), imgs.end(), [](const Input* a, const Input* b) { return a->pixels() < b->pixels(); });
    Input& base = loadInput(opt.base);
    opt.baseWidth = base.img.width;
    size_t maxPixels = base.pixels();
    for (const Input* in : imgs) maxPixels = std::max(maxPixels, in->pixels());
    const std::string powerStart = powerJson(true);
    std::time_t startTime = std::time(nullptr);

    std::string timesPath = opt.outDir + "/tiempos_crudos.csv";
    std::string validPath = opt.outDir + "/validacion.csv";
    csvTimes = fopen(timesPath.c_str(), "w");
    csvValid = fopen(validPath.c_str(), "w");
    if (!csvTimes || !csvValid) {
        fprintf(stderr, "No se pudieron crear los CSV en %s\n", opt.outDir.c_str());
        return 1;
    }
    fprintf(csvTimes,
            "experimento,implementacion,filtro,imagen,resolucion,mascara,hilos_por_bloque,bloques,repeticion,metrica,"
            "tiempo,unidad\n");
    fprintf(csvValid,
            "experimento,filtro,imagen,resolucion,mascara,hilos_por_bloque,bloques,error_abs_max,error_abs_medio,"
            "tolerancia,pixeles_no_finitos,pixeles_8bit_distintos,resultado\n");

    gpu = new GpuConvolver(maxPixels, 21);

    std::vector<float> ref, out;
    printf("Repeticiones medidas por configuración: %d (+ calentamiento: 1 en CPU, >= %.0f ms en GPU)\n", opt.reps,
           opt.warmupMs);
    printf("Imágenes (versiones de la misma foto):");
    for (const Input* in : imgs) printf(" %s %s;", in->name.c_str(), in->dims().c_str());
    printf(" base de los experimentos 1, 3 y 4: %s %s\n", base.name.c_str(), base.dims().c_str());

    if (opt.experiments.count(1)) {
        printf("Experimento 1: filtro (media, nitidez, laplaciano 3x3; %s; %d hilos)\n", base.dims().c_str(),
               kDefaultThreads);
        saveInput(base);
        for (FilterType t : {FilterType::Average, FilterType::Sharpen, FilterType::Laplacian}) {
            Mask m = makeMask(t, 3);
            runCPU(1, m, base, ref);
            runGPU(1, m, base, kDefaultThreads, ref, out);
            saveOutput(std::string("exp1_") + filterName(t) + "3x3_" + base.name + "_cuda", out, t, base);
            saveOutput(std::string("exp1_") + filterName(t) + "3x3_" + base.name + "_cpu", ref, t, base);
        }
    }
    if (opt.experiments.count(2)) {
        printf("Experimento 2: resolución (media 9x9; 2000, 4000 y 6000; %d hilos)\n", kDefaultThreads);
        Mask m = makeMask(FilterType::Average, 9);
        for (const Input* in : imgs) {
            saveInput(*in);
            runCPU(2, m, *in, ref);
            runGPU(2, m, *in, kDefaultThreads, ref, out);
            saveOutput("exp2_media9x9_" + in->name + "_cuda", out, m.type, *in);
        }
    }
    if (opt.experiments.count(3)) {
        printf("Experimento 3: máscara (media 3x3, 9x9, 21x21; %s; %d hilos)\n", base.dims().c_str(), kDefaultThreads);
        for (int k : {3, 9, 21}) {
            Mask m = makeMask(FilterType::Average, k);
            runCPU(3, m, base, ref);
            runGPU(3, m, base, kDefaultThreads, ref, out);
            saveOutput("exp3_media" + maskLabel(m) + "_" + base.name + "_cuda", out, m.type, base);
        }
    }
    if (opt.experiments.count(4)) {
        printf("Experimento 4: hilos por bloque (media 9x9; %s; 128, 256, 512)\n", base.dims().c_str());
        Mask m = makeMask(FilterType::Average, 9);
        runCPU(4, m, base, ref);
        for (int tpb : {128, 256, 512}) runGPU(4, m, base, tpb, ref, out);
    }
    delete gpu;
    fclose(csvTimes);
    fclose(csvValid);

    FILE* f = fopen((opt.outDir + "/run_info.json").c_str(), "w");
    if (!f) {
        fprintf(stderr, "No se pudo escribir %s/run_info.json\n", opt.outDir.c_str());
        return 1;
    }
    char date[64];
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", std::localtime(&startTime));
    fprintf(f, "{\n  \"date\": \"%s\",\n  \"repetitions\": %d,\n  \"cpu_warmup_runs\": 1,\n", date, opt.reps);
    fprintf(f, "  \"gpu_warmup_ms\": %.0f,\n  \"default_threads_per_block\": %d,\n", opt.warmupMs, kDefaultThreads);
    fprintf(f, "  \"power_start\": %s,\n  \"power_end\": %s,\n", powerStart.c_str(), powerJson(false).c_str());
    fprintf(f, "  \"base_image\": \"%s\",\n  \"images\": [", base.name.c_str());
    for (size_t i = 0; i < imgs.size(); ++i)
        fprintf(f, "%s\n    {\"name\": \"%s\", \"width\": %d, \"height\": %d, \"pixels\": %zu}", i ? "," : "",
                imgs[i]->name.c_str(), imgs[i]->img.width, imgs[i]->img.height, imgs[i]->pixels());
    fprintf(f, "\n  ],\n  \"detail_crops\": {");
    bool first = true;
    for (const auto& [name, p] : cropPos) {
        fprintf(f, "%s\n    \"%s\": {\"x\": %d, \"y\": %d, \"size\": %d}", first ? "" : ",", name.c_str(), p.x0, p.y0, p.s);
        first = false;
    }
    fprintf(f, "\n  },\n  \"gpu\": %s\n}\n", gpuInfoJson().c_str());
    fclose(f);
    printf("Tiempos crudos: %s\nValidación:     %s\n", timesPath.c_str(), validPath.c_str());
    return 0;
}
