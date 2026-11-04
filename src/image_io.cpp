#define _CRT_SECURE_NO_WARNINGS
#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

std::string lowerExt(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

// Lee un token del encabezado PGM saltando espacios y comentarios '#'.
bool readToken(FILE* f, std::string& tok) {
    int c;
    tok.clear();
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
            continue;
        }
        if (!std::isspace(c)) break;
    }
    if (c == EOF) return false;
    do {
        tok.push_back((char)c);
    } while ((c = fgetc(f)) != EOF && !std::isspace(c));
    return true;  // consume exactamente un separador tras el token
}

uint32_t le32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
uint16_t le16(const unsigned char* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

void put32be(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back((unsigned char)(x >> 24));
    v.push_back((unsigned char)(x >> 16));
    v.push_back((unsigned char)(x >> 8));
    v.push_back((unsigned char)x);
}

uint32_t crc32(const unsigned char* data, size_t n, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

bool writeChunk(FILE* f, const char* type, const std::vector<unsigned char>& data) {
    std::vector<unsigned char> buf;
    put32be(buf, (uint32_t)data.size());
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    uint32_t crc = crc32(buf.data() + 4, buf.size() - 4) ^ 0xFFFFFFFFu;
    put32be(buf, crc);
    return fwrite(buf.data(), 1, buf.size(), f) == buf.size();
}

}  // namespace

bool readImage(const std::string& path, GrayImage& img) {
    std::string ext = lowerExt(path);
    if (ext == "pgm") return readPGM(path, img);
    if (ext == "bmp") return readBMP(path, img);
    fprintf(stderr, "Formato no soportado: %s (use .pgm o .bmp)\n", path.c_str());
    return false;
}

bool readPGM(const std::string& path, GrayImage& img) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s\n", path.c_str());
        return false;
    }
    std::string magic, sw, sh, smax;
    bool ok = readToken(f, magic) && readToken(f, sw) && readToken(f, sh) && readToken(f, smax);
    if (!ok || (magic != "P5" && magic != "P2")) {
        fprintf(stderr, "%s no es un PGM válido\n", path.c_str());
        fclose(f);
        return false;
    }
    int w = std::atoi(sw.c_str()), h = std::atoi(sh.c_str()), maxv = std::atoi(smax.c_str());
    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 255) {
        fprintf(stderr, "%s: dimensiones o maxval no soportados\n", path.c_str());
        fclose(f);
        return false;
    }
    img.width = w;
    img.height = h;
    img.pixels.assign((size_t)w * h, 0);
    if (magic == "P5") {
        ok = fread(img.pixels.data(), 1, img.pixels.size(), f) == img.pixels.size();
    } else {
        std::string tok;
        for (size_t i = 0; i < img.pixels.size() && ok; ++i) {
            ok = readToken(f, tok);
            img.pixels[i] = (unsigned char)std::atoi(tok.c_str());
        }
    }
    fclose(f);
    if (ok && maxv != 255)
        for (auto& p : img.pixels) p = (unsigned char)std::lround(p * 255.0 / maxv);
    if (!ok) fprintf(stderr, "%s: datos incompletos\n", path.c_str());
    return ok;
}

bool readBMP(const std::string& path, GrayImage& img) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s\n", path.c_str());
        return false;
    }
    unsigned char fh[14], ih[40];
    if (fread(fh, 1, 14, f) != 14 || fh[0] != 'B' || fh[1] != 'M' || fread(ih, 1, 40, f) != 40) {
        fprintf(stderr, "%s no es un BMP válido\n", path.c_str());
        fclose(f);
        return false;
    }
    uint32_t offset = le32(fh + 10);
    uint32_t ihSize = le32(ih);
    int32_t w = (int32_t)le32(ih + 4);
    int32_t hRaw = (int32_t)le32(ih + 8);
    uint16_t bpp = le16(ih + 14);
    uint32_t compression = le32(ih + 16);
    uint32_t clrUsed = le32(ih + 32);
    bool topDown = hRaw < 0;
    int h = topDown ? -hRaw : hRaw;
    if (w <= 0 || h <= 0 || (bpp != 8 && bpp != 24 && bpp != 32) || (compression != 0 && compression != 3)) {
        fprintf(stderr, "%s: BMP no soportado (bpp=%d, compresión=%u)\n", path.c_str(), bpp, compression);
        fclose(f);
        return false;
    }
    unsigned char palette[256] = {0};
    if (bpp == 8) {
        uint32_t n = clrUsed ? clrUsed : 256;
        fseek(f, 14 + ihSize, SEEK_SET);
        for (uint32_t i = 0; i < n && i < 256; ++i) {
            unsigned char e[4];
            if (fread(e, 1, 4, f) != 4) break;
            palette[i] = (unsigned char)((299 * e[2] + 587 * e[1] + 114 * e[0] + 500) / 1000);
        }
    }
    size_t rowSize = ((size_t)bpp * w + 31) / 32 * 4;
    std::vector<unsigned char> row(rowSize);
    img.width = w;
    img.height = h;
    img.pixels.assign((size_t)w * h, 0);
    fseek(f, offset, SEEK_SET);
    for (int r = 0; r < h; ++r) {
        if (fread(row.data(), 1, rowSize, f) != rowSize) {
            fprintf(stderr, "%s: datos incompletos\n", path.c_str());
            fclose(f);
            return false;
        }
        int y = topDown ? r : h - 1 - r;
        unsigned char* dst = &img.pixels[(size_t)y * w];
        for (int x = 0; x < w; ++x) {
            if (bpp == 8) {
                dst[x] = palette[row[x]];
            } else {
                const unsigned char* p = &row[(size_t)x * (bpp / 8)];  // orden B, G, R
                dst[x] = (unsigned char)((299 * p[2] + 587 * p[1] + 114 * p[0] + 500) / 1000);
            }
        }
    }
    fclose(f);
    return true;
}

bool writePGM(const std::string& path, const GrayImage& img) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "No se pudo crear %s\n", path.c_str());
        return false;
    }
    fprintf(f, "P5\n%d %d\n255\n", img.width, img.height);
    bool ok = fwrite(img.pixels.data(), 1, img.pixels.size(), f) == img.pixels.size();
    fclose(f);
    return ok;
}

bool writePNG(const std::string& path, const GrayImage& img) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "No se pudo crear %s\n", path.c_str());
        return false;
    }
    static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    bool ok = fwrite(sig, 1, 8, f) == 8;

    std::vector<unsigned char> ihdr;
    put32be(ihdr, (uint32_t)img.width);
    put32be(ihdr, (uint32_t)img.height);
    ihdr.insert(ihdr.end(), {8, 0, 0, 0, 0});  // 8 bits, escala de grises, sin entrelazado
    ok = ok && writeChunk(f, "IHDR", ihdr);

    // Flujo zlib con bloques deflate sin comprimir; cada fila va precedida del filtro 0.
    std::vector<unsigned char> raw;
    raw.reserve((size_t)(img.width + 1) * img.height);
    for (int y = 0; y < img.height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), img.pixels.begin() + (size_t)y * img.width,
                   img.pixels.begin() + (size_t)(y + 1) * img.width);
    }
    std::vector<unsigned char> z = {0x78, 0x01};
    size_t pos = 0;
    do {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((unsigned char)(len & 0xFF));
        z.push_back((unsigned char)(len >> 8));
        z.push_back((unsigned char)(~len & 0xFF));
        z.push_back((unsigned char)((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
    } while (pos < raw.size());
    uint32_t a = 1, b = 0;
    for (unsigned char c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    put32be(z, (b << 16) | a);
    ok = ok && writeChunk(f, "IDAT", z);
    ok = ok && writeChunk(f, "IEND", {});
    fclose(f);
    return ok;
}

GrayImage cropImage(const GrayImage& img, int x0, int y0, int w, int h) {
    GrayImage out;
    out.width = w;
    out.height = h;
    out.pixels.resize((size_t)w * h);
    for (int y = 0; y < h; ++y)
        std::memcpy(&out.pixels[(size_t)y * w], &img.pixels[(size_t)(y0 + y) * img.width + x0], w);
    return out;
}

GrayImage downscaleBox(const GrayImage& img, int maxSide) {
    int factor = std::max(1, (std::max(img.width, img.height) + maxSide - 1) / maxSide);
    GrayImage out;
    out.width = img.width / factor;
    out.height = img.height / factor;
    out.pixels.resize((size_t)out.width * out.height);
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x) {
            unsigned sum = 0;
            for (int dy = 0; dy < factor; ++dy)
                for (int dx = 0; dx < factor; ++dx)
                    sum += img.pixels[(size_t)(y * factor + dy) * img.width + x * factor + dx];
            out.pixels[(size_t)y * out.width + x] = (unsigned char)((sum + factor * factor / 2) / (factor * factor));
        }
    return out;
}

// Redimensionado por área: cada píxel de destino es el promedio de los píxeles de origen que cubre,
// ponderado por la fracción de superficie cubierta. Es el método adecuado para reducir sin aliasing.
GrayImage resizeArea(const GrayImage& img, int newW, int newH) {
    struct Span {
        int first;
        std::vector<double> w;
    };
    auto spans = [](int n, int src) {
        const double s = (double)src / n;
        std::vector<Span> v(n);
        for (int i = 0; i < n; ++i) {
            const double a = i * s, b = (i + 1) * s;
            v[i].first = (int)a;
            for (int k = v[i].first; k < src && k < b; ++k) v[i].w.push_back(std::min(b, k + 1.0) - std::max(a, (double)k));
        }
        return v;
    };
    const std::vector<Span> cols = spans(newW, img.width), rows = spans(newH, img.height);
    const double area = ((double)img.width / newW) * ((double)img.height / newH);
    GrayImage out;
    out.width = newW;
    out.height = newH;
    out.pixels.resize((size_t)newW * newH);
    std::vector<double> acc(img.width);
    for (int y = 0; y < newH; ++y) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (size_t k = 0; k < rows[y].w.size(); ++k) {
            const unsigned char* p = &img.pixels[(size_t)(rows[y].first + k) * img.width];
            for (int x = 0; x < img.width; ++x) acc[x] += rows[y].w[k] * p[x];
        }
        for (int x = 0; x < newW; ++x) {
            double v = 0;
            for (size_t k = 0; k < cols[x].w.size(); ++k) v += cols[x].w[k] * acc[cols[x].first + k];
            out.pixels[(size_t)y * newW + x] = (unsigned char)std::lround(std::clamp(v / area, 0.0, 255.0));
        }
    }
    return out;
}
