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

### Desplegar desde cero, paso a paso

**Requisitos del equipo** (Windows 10 u 11):

- GPU NVIDIA con controlador 555 o superior. En PowerShell, `nvidia-smi` debe mostrar la GPU.
- Docker Desktop con WSL2, abierto (en la barra de tareas debe decir *Engine running*).
- Git.

El controlador lo aporta el anfitrión; la imagen trae el CUDA Toolkit 12.5, Python 3.10, PyCUDA, Flask y
gunicorn, así que para el servicio no hace falta instalar Python ni CUDA en Windows.

**1. Clonar el repositorio**

```
git clone https://github.com/johnny567w/Practica2_P.git
cd Practica2_P
```

**2. Construir la imagen** (la primera vez descarga la imagen base de NVIDIA, unos GB, y tarda varios minutos)

```
docker build -t convolucion-web:1.0 .
```

**3. Comprobar que el contenedor ve la GPU** (debe mostrar el nombre de la GPU y la versión del controlador)

```
docker run --rm --gpus all convolucion-web:1.0 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Se usa la consulta en CSV porque la tabla completa de `nvidia-smi` puede fallar dentro de WSL2.

**4. Iniciar el servicio** (dejar esta terminal abierta; Ctrl+C lo detiene)

```
docker run --rm --name convolucion-web --gpus all -p 8000:8000 convolucion-web:1.0
```

Está listo cuando aparece `Listening at: http://0.0.0.0:8000`. `run_web.bat` hace los pasos 2, 3 y 4.
El servicio corre con gunicorn (1 proceso, 8 hilos), sin depurador ni recarga automática.

**5. Probarlo**

- Estado: abrir `http://localhost:8000/api/salud`; debe responder `"ok": true` con el nombre de la GPU.
- Interfaz: abrir `http://localhost:8000`, elegir una imagen (PNG, JPEG, BMP o PGM de hasta 6000 píxeles por
  lado), un filtro y *Procesar*. `data\img3.jpg` mide 6315×8416 y se rechaza a propósito por el límite.
- Desde otro equipo de la red: `http://<IP de este equipo>:8000` (hay que permitir el puerto 8000 en el
  firewall de Windows).

**6. Validar** (con el servicio encendido, en otra terminal)

La validación usa las imágenes `data\img_2000.pgm`, `img_4000.pgm` e `img_6000.pgm`, que no están en el
repositorio: se generan desde `data\img3.jpg`. Esto necesita Visual Studio 2022 (C++), CUDA Toolkit 12.5 y
Python 3.13 en Windows.

```
py -3.13 -m pip install numpy pycuda pillow
build.bat
powershell -ExecutionPolicy Bypass -File scripts\prepare_images.ps1
py -3.13 web\cliente.py validar --url http://localhost:8000
```

Debe terminar con `Resultado: TODAS LAS PRUEBAS PASARON`.

**7. Medir** (opcional; detiene y vuelve a iniciar el servicio por su cuenta, y reemplaza los archivos de
`results\web`)

```
powershell -ExecutionPolicy Bypass -File web\experimentos.ps1
```

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

### Pruebas y experimentos por separado

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
