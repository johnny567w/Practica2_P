"""Convolucion con PyCUDA: compila el kernel de la practica 01 (kernel_convolucion.cu) con SourceModule
y lo ejecuta desde Python, con drv.In / drv.Out o con arreglos de GPU para medir las copias por separado."""
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

F32 = np.float32
KERNEL = Path(__file__).resolve().parent / "kernel_convolucion.cu"


def preparar_compilador():
    """PyCUDA compila con nvcc, y nvcc necesita cl.exe (Visual Studio 2022) en el PATH."""
    cuda = os.environ.get("CUDA_PATH")
    if cuda and shutil.which("nvcc") is None:
        os.environ["PATH"] = str(Path(cuda) / "bin") + os.pathsep + os.environ["PATH"]
    if shutil.which("cl") is not None:
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.exists():
        rutas = subprocess.run([str(vswhere), "-version", "[17.0,18.0)", "-products", "*", "-requires",
                                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                                "installationPath"], capture_output=True, text=True).stdout.splitlines()
        for vs in rutas:
            for d in sorted((Path(vs) / "VC/Tools/MSVC").glob("*/bin/Hostx64/x64"), reverse=True):
                if (d / "cl.exe").exists():
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
                    return
    raise RuntimeError("No se encontro cl.exe. Instale Visual Studio 2022 con 'Desarrollo para escritorio con C++'.")


class PyCuda:
    """Contexto CUDA + kernel de la practica 01 compilado con SourceModule."""

    def __init__(self):
        preparar_compilador()
        t0 = time.perf_counter()
        import pycuda.driver as drv
        drv.init()
        self.dispositivo = drv.Device(0)
        self.contexto = self.dispositivo.make_context()
        self.ms_contexto = (time.perf_counter() - t0) * 1000.0

        from pycuda.compiler import SourceModule
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # nvcc imprime el nombre del archivo aunque compile bien
            # cache_dir=False: se compila de verdad cada vez, para medir la compilacion de forma honesta
            self.modulo = SourceModule(KERNEL.read_text(encoding="utf-8"), cache_dir=False)
        self.kernel = self.modulo.get_function("convolveKernel")
        self.ms_compilacion = (time.perf_counter() - t0) * 1000.0
        self.drv = drv
        self._buffers = None

    def cerrar(self):
        self._buffers = None
        self.contexto.pop()
        self.contexto.detach()

    @staticmethod
    def bloques(n, hilos):
        return (n + hilos - 1) // hilos  # se redondea hacia arriba: el ultimo bloque puede quedar incompleto

    @staticmethod
    def _contiguo(a):
        a = np.ascontiguousarray(a, dtype=F32).ravel()
        assert a.flags["C_CONTIGUOUS"] and a.dtype == F32
        return a

    def ejecutar(self, img, m, hilos=256, envenenar=False):
        """Ejecucion simple con drv.In / drv.Out: PyCUDA reserva la memoria de la GPU, copia la entrada,
        lanza el kernel, copia la salida y libera la memoria en la misma llamada.
        Con envenenar=True la salida se llena antes con NaN (drv.InOut): si un pixel no se calcula, queda NaN."""
        drv = self.drv
        alto, ancho = img.shape
        n, k = alto * ancho, m.shape[0]
        entrada, coef = self._contiguo(img), self._contiguo(m)
        salida = np.full(n, np.nan, dtype=F32) if envenenar else np.empty(n, dtype=F32)
        destino = drv.InOut(salida) if envenenar else drv.Out(salida)
        self.kernel(drv.In(entrada), destino, np.int32(ancho), np.int32(alto), drv.In(coef), np.int32(k),
                    block=(hilos, 1, 1), grid=(self.bloques(n, hilos), 1))
        return salida.reshape(alto, ancho)

    def medir_inout(self, img, m, hilos=256):
        """Tiempo total de la ejecucion con drv.In / drv.Out (incluye reservar y liberar memoria)."""
        drv = self.drv
        ini, fin = drv.Event(), drv.Event()
        ini.record()
        salida = self.ejecutar(img, m, hilos)
        fin.record()
        fin.synchronize()
        return fin.time_since(ini), salida

    def medir_separado(self, img, m, hilos=256):
        """Mide por separado CPU->GPU, kernel y GPU->CPU con eventos CUDA, igual que la version CUDA C.
        drv.In/Out hace las copias dentro de la misma llamada del kernel y no deja medirlas por separado,
        asi que aqui se usan arreglos de GPU (gpuarray) reservados antes de medir (la reserva no se cuenta)."""
        import pycuda.gpuarray as gpuarray
        drv = self.drv
        alto, ancho = img.shape
        n, k = alto * ancho, m.shape[0]
        entrada, coef = self._contiguo(img), self._contiguo(m)
        if self._buffers is None or self._buffers[0].size != n or self._buffers[2].size != coef.size:
            self._buffers = (gpuarray.empty(n, F32), gpuarray.empty(n, F32), gpuarray.empty(coef.size, F32))
        d_in, d_out, d_m = self._buffers
        salida = np.empty(n, dtype=F32)
        ev = [drv.Event() for _ in range(4)]
        ev[0].record()
        d_in.set(entrada)
        d_m.set(coef)
        ev[1].record()
        self.kernel(d_in.gpudata, d_out.gpudata, np.int32(ancho), np.int32(alto), d_m.gpudata, np.int32(k),
                    block=(hilos, 1, 1), grid=(self.bloques(n, hilos), 1))
        ev[2].record()
        d_out.get(salida)
        ev[3].record()
        ev[3].synchronize()
        tiempos = {"h2d": ev[1].time_since(ev[0]), "kernel": ev[2].time_since(ev[1]),
                   "d2h": ev[3].time_since(ev[2]), "total": ev[3].time_since(ev[0])}
        return tiempos, salida.reshape(alto, ancho)


