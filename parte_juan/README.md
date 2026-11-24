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
- Práctica 02, referencia para el servicio web: commit `1da731f`

## Servicio web con PyCUDA en Docker

Arquitectura: **navegador → servicio web Python (Flask + gunicorn) → PyCUDA → GPU → resultado → navegador**.

| Archivo | Qué hace |
|---|---|
| `web/procesamiento.py` | Procesamiento sin HTTP: valida y decodifica la imagen, la pasa a gris, ejecuta el kernel propio con PyCUDA y codifica el PNG. |
| `web/servicio.py` | Servicio HTTP e interfaz. Un único trabajador de GPU atiende en orden de llegada; cola de `COLA_MAX` solicitudes. |
| `web/templates/`, `web/static/` | Interfaz web: cargar imagen, elegir filtro y máscara, ver original y resultado, descargar y ver los tiempos. |
| `web/directo.py` | Ejecución directa (sin HTTP) de la misma función, en el anfitrión o dentro del contenedor. |
| `web/cliente.py` | Validación funcional y numérica, y experimentos a través del servicio. |
| `Dockerfile`, `web/requirements.txt` | Imagen del contenedor con versiones fijas. |

### Construir, comprobar la GPU e iniciar

Requiere Docker Desktop con WSL2 y una GPU NVIDIA (controlador 555 o superior en el anfitrión). El controlador
lo aporta el anfitrión; la imagen trae el CUDA Toolkit 12.5, Python 3.10, PyCUDA, Flask y gunicorn.

```
docker build -t convolucion-web:1.0 .
docker run --rm --gpus all convolucion-web:1.0 nvidia-smi
docker run --rm --name convolucion-web --gpus all -p 8000:8000 convolucion-web:1.0
```

(`run_web.bat` hace los tres pasos.) El servicio queda en `http://localhost:8000`, con gunicorn
(1 proceso, 8 hilos), sin depurador ni recarga automática.

### API

**POST `/api/procesar`** (`multipart/form-data`)

| Campo | Obligatorio | Valores |
|---|---|---|
| `imagen` | sí | archivo PNG, JPEG, BMP o PGM; máximo 60 MB y 6000×6000 píxeles |
| `filtro` | sí | `media`, `nitidez` o `laplaciano` |
| `k` | no (3) | `3`, `9` o `21` para la media; `3` para los otros filtros |
| `devolver_float` | no | `1` para obtener también la salida en float32 antes de pasar a 8 bits |

Respuesta correcta (200, JSON): `id` de la solicitud, `filtro`, `k`, `ancho`, `alto`, `hilos_por_bloque`,
`bloques`, `archivos` (URL del PNG y, si se pidió, del float32) y `tiempos` en milisegundos:
`espera_cola_ms`, `decodificacion_ms`, `preparacion_ms`, `reserva_ms`, `h2d_ms`, `kernel_ms`, `d2h_ms`
(los tres últimos con eventos CUDA, memoria paginable), `salida_ms` (8 bits + PNG), `escritura_ms`,
`procesamiento_ms` y `servidor_ms`.

Errores (JSON con `ok: false` y `error`): 400 parámetros o archivo no válidos, 413 archivo o imagen demasiado
grandes, 415 formato no permitido, 503 GPU no disponible o cola llena, 500 error al procesar.

**GET `/api/resultados/<id>/resultado.png`** descarga el resultado (`?descargar=1` para guardarlo).
**GET `/api/salud`** estado de la GPU, inicialización del contexto y compilación del kernel.

Cada solicitud tiene un identificador único y su propia carpeta; los resultados se borran a los 15 minutos.

### Pruebas y experimentos

```
py -3.13 web\cliente.py validar --url http://localhost:8000
py -3.13 web\directo.py --modo anfitrion
docker run --rm --gpus all -v %cd%\data:/app/data -v %cd%\results:/app/results convolucion-web:1.0 python3 directo.py --modo contenedor
py -3.13 web\cliente.py medir --url http://localhost:8000
py -3.13 web\frecuencia_gpu.py
```

`web\experimentos.ps1` ejecuta en orden las tres mediciones (directo en el anfitrión, directo en el contenedor
y servicio). `frecuencia_gpu.py` es una prueba de control: muestra que el kernel tarda más cuando hay pausas
entre lanzamientos (como entre solicitudes), porque la GPU de la laptop baja su frecuencia.

Resultados en `results/web/`.
