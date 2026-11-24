
import ctypes
import math
import os
import platform
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
BIN = RAIZ / "bin"
RESULTADOS = RAIZ / "results" / "python"
F32 = np.float32


# ----------------------------------------------------------------------------------------------
# Imagenes: lectura, escala de grises y escritura
# ----------------------------------------------------------------------------------------------
def leer_imagen(ruta):
    datos = Path(ruta).read_bytes()
    if datos[:2] in (b"P5", b"P6"):
        canales = 1 if datos[:2] == b"P5" else 3
        campos, pos = [], 2
        while len(campos) < 3:
            while datos[pos:pos + 1].isspace():
                pos += 1
            if datos[pos:pos + 1] == b"#":
                pos = datos.index(b"\n", pos)
                continue
            ini = pos
            while not datos[pos:pos + 1].isspace():
                pos += 1
            campos.append(int(datos[ini:pos]))
        ancho, alto, maximo = campos
        if maximo > 255:
            raise ValueError(f"{ruta}: solo se admiten imagenes de 8 bits")
        pixeles = np.frombuffer(datos, dtype=np.uint8, count=ancho * alto * canales, offset=pos + 1)
        forma = (alto, ancho) if canales == 1 else (alto, ancho, 3)
        return pixeles.reshape(forma).copy()
    if datos[:2] == b"BM":
        offset, = struct.unpack_from("<I", datos, 10)
        ancho, alto, _, bits = struct.unpack_from("<iiHH", datos, 18)
        if bits != 24:
            raise ValueError(f"{ruta}: solo BMP de 24 bits")
        fila = (ancho * 3 + 3) & ~3
        img = np.empty((abs(alto), ancho, 3), dtype=np.uint8)
        for y in range(abs(alto)):
            linea = np.frombuffer(datos, np.uint8, ancho * 3, offset + y * fila).reshape(ancho, 3)
            img[abs(alto) - 1 - y if alto > 0 else y] = linea[:, ::-1]  # BGR -> RGB, de abajo hacia arriba
        return img
    raise ValueError(f"{ruta}: formato no soportado")


def a_grises(img):
    if img.ndim == 2:
        return img
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(np.floor(y + 0.5), 0, 255).astype(np.uint8)


def escribir_pgm(ruta, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    Path(ruta).write_bytes(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]) + img.tobytes())


def escribir_png(ruta, img):
    """PNG en escala de grises de 8 bits."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    alto, ancho = img.shape
    crudo = b"".join(b"\x00" + img[y].tobytes() for y in range(alto))

    def bloque(tipo, contenido):
        return struct.pack(">I", len(contenido)) + tipo + contenido + struct.pack(">I", zlib.crc32(tipo + contenido))

    Path(ruta).write_bytes(b"\x89PNG\r\n\x1a\n" + bloque(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 0, 0, 0, 0))
                           + bloque(b"IDAT", zlib.compress(crudo, 6)) + bloque(b"IEND", b""))


def reducir(img, lado_max=800):
    """Reduccion por promedio de bloques, solo para las vistas previas."""
    f = max(1, math.ceil(max(img.shape) / lado_max))
    alto, ancho = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:alto, :ancho].reshape(alto // f, f, ancho // f, f).mean(axis=(1, 3)).round().astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Filtros, conversion a 8 bits, tolerancia y errores
# ----------------------------------------------------------------------------------------------
FILTROS = ("media", "nitidez", "laplaciano")


def mascara(filtro, k=3):
    """Mascaras de la practica 01 en float32."""
    if filtro == "media":
        if k < 1 or k % 2 == 0:
            raise ValueError("k debe ser impar")
        return np.full((k, k), F32(1.0) / F32(k * k), dtype=F32)
    if filtro == "nitidez":
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=F32)
    if filtro == "laplaciano":
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=F32)
    raise ValueError(f"filtro desconocido: {filtro}")


def a_8bits(valores, filtro):
    v = np.abs(valores) if filtro == "laplaciano" else valores
    return np.clip(np.floor(v.astype(F32) + F32(0.5)), 0, 255).astype(np.uint8)


def tolerancia(m, maximo_entrada=255.0):
    u, n = 2.0 ** -24, m.size
    gamma = n * u / (1.0 - n * u)
    return 2.0 * gamma * maximo_entrada * float(np.abs(m.astype(np.float64)).sum())


def errores(referencia, valores):
    """Error absoluto maximo y medio (antes de pasar a 8 bits) y pixeles no escritos (NaN/inf)."""
    no_finitos = int(np.count_nonzero(~np.isfinite(valores)))
    d = np.abs(valores.astype(np.float64) - referencia.astype(np.float64))
    d = d[np.isfinite(d)]
    return (float(d.max()) if d.size else 0.0), (float(d.mean()) if d.size else 0.0), no_finitos


# ----------------------------------------------------------------------------------------------
# Convolucion secuencial en Python: recorridos explicitos de pixeles y coeficientes
# ----------------------------------------------------------------------------------------------
class Interrumpida(Exception):

    def __init__(self, segundos, filas):
        super().__init__(f"interrumpida a los {segundos:.1f} s ({filas} filas)")
        self.segundos, self.filas = segundos, filas


def convolucion_python(img, m, limite_s=None):
    """Convolucion secuencial en Python, recorriendo cada pixel y cada coeficiente."""
    alto, ancho = img.shape
    k = m.shape[0]
    r = k // 2
    if img.dtype == np.uint8 or np.all(img == np.round(img)):
        src = img.astype(np.int64).ravel().tolist()  # enteros exactos; int * float32 da float32
    else:
        src = [F32(v) for v in img.ravel().tolist()]
    coef = [F32(c) for c in m.ravel().tolist()]
    cero = F32(0.0)
    salida = np.empty(alto * ancho, dtype=F32)
    t0 = time.perf_counter()
    for fila in range(alto):
        if limite_s is not None and time.perf_counter() - t0 > limite_s:
            raise Interrumpida(time.perf_counter() - t0, fila)
        i0, i1 = max(0, r - fila), min(k, alto - fila + r)  # filas de la mascara dentro de la imagen
        valores = []
        for col in range(ancho):
            j0, j1 = max(0, r - col), min(k, ancho - col + r)  # columnas de la mascara dentro de la imagen
            acc = cero
            for i in range(i0, i1):
                base = (fila + i - r) * ancho + col - r
                mi = i * k
                for j in range(j0, j1):
                    acc = acc + coef[mi + j] * src[base + j]
            valores.append(acc)
        salida[fila * ancho:(fila + 1) * ancho] = valores
    return salida.reshape(alto, ancho)


# ----------------------------------------------------------------------------------------------
# Version C/C++ y CUDA C de la practica 01 (para comparar)
# ----------------------------------------------------------------------------------------------
def ejecutar_c(version, ruta_pgm, filtro="media", k=3, hilos=256, coef=None):
    exe = BIN / "filter.exe"
    if not exe.exists():
        raise RuntimeError("Falta bin\\filter.exe: ejecute primero build.bat")
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tmp = RESULTADOS / "_salida_c.f32"
    cmd = [str(exe), "--input", str(ruta_pgm), "--filter", filtro, "--k", str(k), "--threads", str(hilos),
           "--impl", "cpu" if version == "c" else "cuda", "--float", str(tmp)]
    if coef is not None:
        cmd += ["--coef", ",".join(repr(float(c)) for c in np.ravel(coef))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"filter.exe fallo: {r.stderr or r.stdout}")
    img = leer_imagen(ruta_pgm)
    valores = np.fromfile(tmp, dtype=F32).reshape(img.shape[:2])
    tmp.unlink()
    return valores


# ----------------------------------------------------------------------------------------------
# Entorno de ejecucion
# ----------------------------------------------------------------------------------------------
def entorno(pc=None):
    info = {}
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as c:
            info["CPU"] = winreg.QueryValueEx(c, "ProcessorNameString")[0].strip()
    except Exception:
        info["CPU"] = platform.processor()
    try:
        nucleos = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).NumberOfCores"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        info["Nucleos"] = f"{nucleos} fisicos / {os.cpu_count()} hilos"
    except Exception:
        info["Nucleos"] = f"{os.cpu_count()} hilos"

    class _Mem(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        mem = _Mem()
        mem.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        info["RAM"] = f"{mem.ullTotalPhys / 2 ** 30:.1f} GB"
    except Exception:
        info["RAM"] = "?"
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        nombre, memoria, driver = [x.strip() for x in smi.strip().split(",")]
        info["GPU"] = nombre
        info["Memoria GPU"] = memoria
        info["Controlador NVIDIA"] = driver
    except Exception:
        if pc is not None:
            info["GPU"] = pc.dispositivo.name()
            info["Memoria GPU"] = f"{pc.dispositivo.total_memory() // 2 ** 20} MiB"
    if pc is not None:
        cc = pc.dispositivo.compute_capability()
        info["Capacidad de computo"] = f"{cc[0]}.{cc[1]}"
    info["Sistema operativo"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info["Python"] = platform.python_version()
    info["NumPy"] = np.__version__
    try:
        import pycuda
        info["PyCUDA"] = pycuda.VERSION_TEXT
    except Exception:
        info["PyCUDA"] = "no instalado"
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        info["CUDA Toolkit"] = [l for l in nvcc.splitlines() if "release" in l][0].split("release")[1].split(",")[0].strip()
    except Exception:
        info["CUDA Toolkit"] = "?"
    try:
        cl = subprocess.run(["cl"], capture_output=True, text=True).stderr.splitlines()[0]
        info["Compilador C++ (para nvcc)"] = cl.strip()
    except Exception:
        pass
    return info


def tamano(img):
    return f"{img.shape[1]}x{img.shape[0]}"


if __name__ == "__main__":
    sys.exit("Este archivo es una biblioteca: ejecute python\\main.py")
