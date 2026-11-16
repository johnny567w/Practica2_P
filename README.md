# Convolución de imágenes: C++, CUDA C, Python y PyCUDA

Práctica de Computación Paralela (Carrera de Computación, UPS).
Integrantes: Johnny Segarra y Juan Alvares.

Se aplican tres filtros a imágenes en escala de grises (media k×k, nitidez y laplaciano) con cuatro versiones
y se comparan sus resultados y sus tiempos:

| Versión | Dónde corre | Archivos |
|---|---|---|
| C++ secuencial | CPU, 1 hilo | `src/conv_cpu.cpp` |
| CUDA C | GPU, un hilo por píxel | `src/conv_gpu.cu` |
| Python secuencial | CPU, 1 hilo | `python/convolucion.py` |
| PyCUDA | GPU, el mismo kernel lanzado desde Python | `python/kernel_convolucion.cu`, `python/conv_pycuda.py` |

## Requisitos

- Tarjeta NVIDIA con CUDA Toolkit 12.5
- Visual Studio 2022 con "Desarrollo para escritorio con C++"
- Para la parte de Python: Python 3.13 con NumPy y PyCUDA (`py -3.13 -m pip install numpy pycuda`)

## Cómo ejecutar

Desde la carpeta del proyecto (con la laptop conectada y sin otros programas abiertos):

```
run_all.bat          compila, prepara las imágenes, valida y mide C++ contra CUDA C
run_python.bat       valida y mide las cuatro versiones (tarda unos 40 minutos)
```

Para usar otra foto: `run_all.bat data\otra_foto.jpg` (jpg, png o bmp). Si es más pequeña que 6000 píxeles
de lado, se amplía con un aviso.

## Resultados

- `results/benchmark_log.txt`: tabla de tiempos CPU contra GPU y aceleración
- `results/tiempos_crudos.csv`: todas las repeticiones (C++ y CUDA C)
- `results/validacion_pruebas.txt`: detalle de la validación
- `results/python/`: validación y tiempos de las cuatro versiones (`tiempos_4_versiones.csv`)

## Versiones

- Práctica 01 (C++ y CUDA C): commit `420cf4c`
- Práctica complementaria: commits `d3cd70f` (C++, CUDA C y Python secuencial) y `810d562` (PyCUDA y experimentos)
