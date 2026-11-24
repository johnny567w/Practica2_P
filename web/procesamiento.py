"""Procesamiento de imágenes con PyCUDA, separado de la parte HTTP.

El mismo código lo usan:
  - el servicio web (servicio.py), dentro de su trabajador de GPU;
  - la ejecución directa (directo.py), en el anfitrión o dentro del contenedor.

Reutiliza la práctica 02: las máscaras y la conversión a escala de grises de python/convolucion.py y el
kernel propio de python/kernel_convolucion.cu, compilado con PyCUDA (python/conv_pycuda.py).
Precisión: float32 en la entrada, la máscara, la acumulación y la salida, como en las prácticas anteriores.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "python"))
import convolucion as cv  # noqa: E402  (máscaras, escala de grises, conversión a 8 bits, tolerancia)
from conv_pycuda import PyCuda  # noqa: E402  (contexto CUDA y kernel compilado con SourceModule)

F32 = np.float32

# Límites de entrada (pensados para la GTX 1650 de 4 GB: una imagen de 6000x6000 en float32 ocupa 144 MB
# de entrada y 144 MB de salida en la GPU, más las copias en la memoria del servidor).
FORMATOS = {"PNG": "PNG", "JPEG": "JPEG", "BMP": "BMP", "PPM": "PGM/PPM"}  # nombre de Pillow -> nombre para el usuario
MAX_BYTES = 60 * 1024 * 1024          # tamaño máximo del archivo: 60 MB
MAX_LADO = 6000                       # lado máximo en píxeles
MIN_LADO = 1
MASCARAS_MEDIA = (3, 9, 21)           # tamaños de máscara permitidos para la media
HILOS_POR_BLOQUE = 256                # un píxel de salida por hilo, bloques e hilos en el eje X
Image.MAX_IMAGE_PIXELS = MAX_LADO * MAX_LADO  # Pillow avisa antes de decodificar imágenes más grandes


class ErrorEntrada(Exception):
    """Entrada no válida: se responde al cliente con un mensaje claro y el código HTTP indicado."""

    def __init__(self, mensaje, codigo=400):
        super().__init__(mensaje)
        self.mensaje, self.codigo = mensaje, codigo


def validar_parametros(filtro, k):
    """Comprueba el filtro y el tamaño de máscara. Devuelve (filtro, k) normalizados."""
    if filtro not in cv.FILTROS:
        raise ErrorEntrada(f"Filtro desconocido: '{filtro}'. Use media, nitidez o laplaciano.")
    try:
        k = int(k) if k not in (None, "") else 3
    except (TypeError, ValueError):
        raise ErrorEntrada(f"El tamaño de máscara debe ser un número entero (recibido: '{k}').")
    if filtro == "media" and k not in MASCARAS_MEDIA:
        raise ErrorEntrada(f"Tamaño de máscara no permitido para la media: {k}. Use 3, 9 o 21.")
    if filtro != "media" and k != 3:
        raise ErrorEntrada(f"El filtro {filtro} solo existe en 3x3 (recibido: {k}).")
    return filtro, k


def decodificar(datos):
    """Valida el archivo por su contenido real (no por la extensión) y lo devuelve en gris (uint8)."""
    if not datos:
        raise ErrorEntrada("No se envió ninguna imagen.")
    if len(datos) > MAX_BYTES:
        raise ErrorEntrada(f"El archivo pesa {len(datos) / 2**20:.1f} MB y el límite es {MAX_BYTES // 2**20} MB.", 413)
    try:
        with Image.open(io.BytesIO(datos)) as im:
            if im.format not in FORMATOS:
                raise ErrorEntrada(f"Formato {im.format} no permitido. Use PNG, JPEG, BMP o PGM.", 415)
            ancho, alto = im.size
            if ancho < MIN_LADO or alto < MIN_LADO or ancho > MAX_LADO or alto > MAX_LADO:
                raise ErrorEntrada(f"La imagen mide {ancho}x{alto}; se aceptan de 1x1 a {MAX_LADO}x{MAX_LADO} píxeles.", 413)
            formato = FORMATOS[im.format]
            im.load()
            if im.mode == "L":
                gris = np.asarray(im, dtype=np.uint8)
            elif im.mode in ("RGB", "RGBA", "P", "LA", "CMYK", "YCbCr"):
                gris = cv.a_grises(np.asarray(im.convert("RGB"), dtype=np.uint8))  # misma fórmula de luminancia
            else:
                raise ErrorEntrada(f"Modo de color {im.mode} no soportado (use imágenes de 8 bits).", 415)
    except ErrorEntrada:
        raise
    except Image.DecompressionBombError:
        raise ErrorEntrada(f"La imagen es demasiado grande (más de {MAX_LADO}x{MAX_LADO} píxeles).", 413)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ErrorEntrada("El archivo no es una imagen válida o está dañado.")
    return np.ascontiguousarray(gris), formato


def codificar_png(img_u8):
    """Codifica en PNG sin pérdida. compress_level=1 para que la codificación de imágenes grandes sea rápida."""
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode="L").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class MotorGPU:
    """Contexto CUDA + kernel propio. Debe crearse y usarse SIEMPRE en el mismo hilo: el contexto CUDA
    queda activo en el hilo que lo crea. En el servicio, ese hilo es el trabajador de GPU."""

    def __init__(self):
        t0 = time.perf_counter()
        self.pc = PyCuda()  # drv.init(), Device(0), make_context() y SourceModule(kernel_convolucion.cu)
        self.inicializacion = {
            "contexto_ms": self.pc.ms_contexto,
            "compilacion_kernel_ms": self.pc.ms_compilacion,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        }
        d = self.pc.dispositivo
        self.gpu = f"{d.name()} ({d.total_memory() // 2**20} MiB, capacidad {d.compute_capability()[0]}."
        self.gpu += f"{d.compute_capability()[1]})"
        self._buffers = None  # arreglos en la GPU, se reservan una vez y se reutilizan si el tamaño no cambia

    def cerrar(self):
        self._buffers = None
        self.pc.cerrar()

    def convolucion(self, img_u8, m, hilos=HILOS_POR_BLOQUE):
        """Convolución en la GPU midiendo por separado reserva, CPU->GPU, kernel y GPU->CPU.

        - Reserva: gpuarray.empty, solo cuando cambia el tamaño de la imagen (reloj del anfitrión).
        - Transferencias: memoria PAGINABLE del anfitrión (arreglos de NumPy normales, no fijada).
        - Tiempos de GPU: eventos CUDA; se sincroniza con el último evento antes de leerlos.
        """
        import pycuda.gpuarray as gpuarray
        drv = self.pc.drv
        alto, ancho = img_u8.shape
        n, k = alto * ancho, m.shape[0]
        entrada = np.ascontiguousarray(img_u8, dtype=F32).ravel()   # contigua y float32, como espera el kernel
        coef = np.ascontiguousarray(m, dtype=F32).ravel()
        t0 = time.perf_counter()
        if self._buffers is None or self._buffers[0].size != n or self._buffers[2].size != coef.size:
            self._buffers = None  # libera los anteriores antes de reservar los nuevos
            self._buffers = (gpuarray.empty(n, F32), gpuarray.empty(n, F32), gpuarray.empty(coef.size, F32))
        reserva_ms = (time.perf_counter() - t0) * 1000.0
        d_in, d_out, d_m = self._buffers
        salida = np.empty(n, dtype=F32)
        bloques = self.pc.bloques(n, hilos)  # se redondea hacia arriba: el kernel controla los hilos sobrantes
        ev = [drv.Event() for _ in range(4)]
        ev[0].record()
        d_in.set(entrada)
        d_m.set(coef)
        ev[1].record()
        self.pc.kernel(d_in.gpudata, d_out.gpudata, np.int32(ancho), np.int32(alto), d_m.gpudata, np.int32(k),
                       block=(hilos, 1, 1), grid=(bloques, 1))
        ev[2].record()
        d_out.get(salida)
        ev[3].record()
        ev[3].synchronize()
        tiempos = {"reserva_ms": reserva_ms, "h2d_ms": ev[1].time_since(ev[0]), "kernel_ms": ev[2].time_since(ev[1]),
                   "d2h_ms": ev[3].time_since(ev[2]), "gpu_total_ms": ev[3].time_since(ev[0])}
        return salida.reshape(alto, ancho), tiempos, bloques


def procesar(motor, datos, filtro, k, hilos=HILOS_POR_BLOQUE, devolver_float=False):
    """Todo el procesamiento de una solicitud, sin HTTP. Mide con un reloj monotónico (perf_counter):
    decodificación + validación + gris, preparación (máscara y float32), GPU (reserva, transferencias y
    kernel) y preparación de la salida (8 bits y codificación PNG). No incluye escribir el archivo."""
    t0 = time.perf_counter()
    filtro, k = validar_parametros(filtro, k)
    gris, formato = decodificar(datos)
    t1 = time.perf_counter()
    m = cv.mascara(filtro, k)
    t2 = time.perf_counter()
    valores, t_gpu, bloques = motor.convolucion(gris, m, hilos)
    t3 = time.perf_counter()
    png = codificar_png(cv.a_8bits(valores, filtro))
    t4 = time.perf_counter()
    tiempos = {"decodificacion_ms": (t1 - t0) * 1000.0, "preparacion_ms": (t2 - t1) * 1000.0,
               **t_gpu, "gpu_con_reserva_ms": (t3 - t2) * 1000.0, "salida_ms": (t4 - t3) * 1000.0,
               "procesamiento_ms": (t4 - t0) * 1000.0}
    return {"filtro": filtro, "k": k, "ancho": gris.shape[1], "alto": gris.shape[0], "formato": formato,
            "hilos": hilos, "bloques": bloques, "tiempos": tiempos, "png": png,
            "valores": valores if devolver_float else None}
