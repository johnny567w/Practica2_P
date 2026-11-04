// Lectura/escritura de imágenes en escala de grises (8 bits por píxel).
// Formatos: PGM (P5/P2) y BMP (8/24/32 bits) para entrada; PGM y PNG para salida.
#pragma once
#include <string>
#include <vector>

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;  // orden fila-mayor: pixels[fila * width + columna]
};

bool readImage(const std::string& path, GrayImage& img);  // elige por extensión (.pgm / .bmp)
bool readPGM(const std::string& path, GrayImage& img);
bool readBMP(const std::string& path, GrayImage& img);    // convierte a gris con 0.299R + 0.587G + 0.114B
bool writePGM(const std::string& path, const GrayImage& img);
bool writePNG(const std::string& path, const GrayImage& img);  // PNG gris sin compresión (bloques "stored")

GrayImage cropImage(const GrayImage& img, int x0, int y0, int w, int h);
GrayImage downscaleBox(const GrayImage& img, int maxSide);  // reducción por promedio de bloques (solo para vistas previas)
GrayImage resizeArea(const GrayImage& img, int newW, int newH);  // redimensionado por promedio de área
