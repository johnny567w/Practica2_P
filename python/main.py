
import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import convolucion as cv
from conv_pycuda import PyCuda

RES = cv.RESULTADOS


class Registro:

    def __init__(self, nombre):
        RES.mkdir(parents=True, exist_ok=True)
        self.ruta = RES / nombre
        self.f = open(self.ruta, "w", encoding="utf-8")

    def say(self, texto=""):
        print(texto, flush=True)
        self.f.write(texto + "\n")
        self.f.flush()

    def detalle(self, texto=""):
        self.f.write(texto + "\n")

    def cerrar(self):
        self.f.close()


def cargar(lado_o_ruta):
    ruta = Path(lado_o_ruta) if not str(lado_o_ruta).isdigit() else cv.DATA / f"img_{lado_o_ruta}.pgm"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Ejecute primero run_all.bat (prepara las imagenes de data\\).")
    return ruta, cv.a_grises(cv.leer_imagen(ruta))


def etiqueta(filtro, k):
    return f"{filtro} {k}x{k}"


def mediana(v):
    return statistics.median(v) if v else None


def de(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def segundos(s):
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def aviso_bateria(log):
    import ctypes

    class _Estado(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    try:
        e = _Estado()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(e))
        if e.ACLineStatus != 1:
            log.say(f"  ADVERTENCIA: la laptop funciona con bateria ({e.BatteryLifePercent & 0xFF} %). "
                    "Conectela para que los tiempos sean confiables.")
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------
# entorno
# ----------------------------------------------------------------------------------------------
def cmd_entorno(_args):
    pc = PyCuda()
    info = cv.entorno(pc)
    log = Registro("entorno.txt")
    log.say("  EQUIPO Y HERRAMIENTAS")
    for clave, valor in info.items():
        log.say(f"  {clave:<28} {valor}")
    log.say(f"  {'Inicializar contexto CUDA':<28} {pc.ms_contexto:.1f} ms")
    log.say(f"  {'Compilar y cargar el kernel':<28} {pc.ms_compilacion:.1f} ms")
    log.cerrar()
    pc.cerrar()


# ----------------------------------------------------------------------------------------------
# filtrar: carga, escala de grises, mascara, filtro elegido, ejecucion y guardado
# ----------------------------------------------------------------------------------------------
def cmd_filtrar(a):
    ruta, img = cargar(a.imagen)
    m = cv.mascara(a.filtro, a.k)
    print(f"  {ruta.name} ({cv.tamano(img)}), {etiqueta(a.filtro, m.shape[0])}, version {a.version}")
    t0 = time.perf_counter()
    if a.version == "python":
        valores = cv.convolucion_python(img, m)
        print(f"  Python secuencial: {(time.perf_counter() - t0) * 1000:.1f} ms")
    else:
        pc = PyCuda()
        ms, valores = pc.medir_inout(img, m, a.hilos)
        print(f"  PyCUDA (drv.In/Out, {pc.bloques(img.size, a.hilos)} bloques x {a.hilos} hilos): {ms:.3f} ms")
        pc.cerrar()
    salida = Path(a.salida) if a.salida else RES / f"{ruta.stem}_{a.filtro}{m.shape[0]}x{m.shape[0]}_{a.version}.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv.escribir_png(salida, cv.a_8bits(valores, a.filtro))
    print(f"  Guardado: {salida}")


# ----------------------------------------------------------------------------------------------
# validar: las cuatro versiones (C++, CUDA C, Python, PyCUDA) dan el mismo resultado
# ----------------------------------------------------------------------------------------------
def cmd_validar(_a):
    log = Registro("validacion_python.txt")
    pc = PyCuda()
    tmp = RES / "_tmp"
    tmp.mkdir(exist_ok=True)
    resumen = {}  # version -> [comparaciones, dentro de tolerancia, error maximo]
    fallas = [0]

    def comparar(caso, m, ref, versiones):
        tol = cv.tolerancia(m)
        for version, valores in versiones.items():
            emax, emed, nf = cv.errores(ref, valores)
            ok = nf == 0 and emax <= tol
            r = resumen.setdefault(version, [0, 0, 0.0])
            r[0] += 1
            r[1] += ok
            r[2] = max(r[2], emax)
            if not ok:
                fallas[0] += 1
            log.detalle(f"  {caso:<34} {version:<18} err.max {emax:10.3g}  err.medio {emed:10.3g}  "
                        f"tol {tol:9.3g}  no escritos {nf:3d}  {'OK' if ok else 'FALLA'}")

    log.say("  Validacion de las cuatro versiones (referencia: C++ secuencial de la practica 01)")
    log.say("  Se comparan los valores en float32 ANTES de pasarlos a 8 bits.\n")

    # 1) Prueba pequena: la matriz de 4x5 de la guia, comprobada a mano
    log.detalle("== Prueba pequena: matriz 4x5 de la guia ==")
    matriz = np.array([[35, 40, 41, 45, 50], [40, 40, 42, 46, 52],
                       [42, 46, 50, 55, 55], [48, 52, 58, 60, 61]], dtype=np.uint8)
    pgm = tmp / "matriz.pgm"
    cv.escribir_pgm(pgm, matriz)
    figura1 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    manual_ok = True
    for nombre, filtro, m, coef in [("Figura 1", "media", figura1, figura1),
                                    ("media 3x3", "media", cv.mascara("media"), None),
                                    ("nitidez 3x3", "nitidez", cv.mascara("nitidez"), None),
                                    ("laplaciano 3x3", "laplaciano", cv.mascara("laplaciano"), None)]:
        ref = cv.ejecutar_c("c", pgm, filtro, 3, coef=coef)
        versiones = {"CUDA C": cv.ejecutar_c("cuda", pgm, filtro, 3, coef=coef),
                     "Python secuencial": cv.convolucion_python(matriz, m),
                     "PyCUDA": pc.ejecutar(matriz, m, 256, envenenar=True)}
        comparar(f"matriz 4x5, {nombre}", m, ref, versiones)
        if nombre == "Figura 1":
            for v, val in [("C++", ref)] + list(versiones.items()):
                okv = abs(val[1, 1] - 78) < 1e-4 and abs(val[1, 2] - 87) < 1e-4
                manual_ok &= okv
                log.detalle(f"    {v:<18} (1,1) = {val[1, 1]:.4f}  (1,2) = {val[1, 2]:.4f}  "
                            f"(esperado 78 y 87)  {'OK' if okv else 'FALLA'}")
    if not manual_ok:
        fallas[0] += 1
    log.say(f"  Prueba pequena (matriz 4x5, calculo a mano): (1,1) = 78 y (1,2) = 87 en las 4 versiones: "
            f"{'SI' if manual_ok else 'NO'}")

    # 2) Prueba de limites: imagen cuyo numero de pixeles no es multiplo de los hilos por bloque
    log.detalle("\n== Prueba de limites: 1001x777 (777777 pixeles, no es multiplo de 128, 256 ni 512) ==")
    rng = np.random.default_rng(12345)
    aleatoria = rng.integers(0, 256, size=(777, 1001), dtype=np.uint8)
    pgm = tmp / "aleatoria.pgm"
    cv.escribir_pgm(pgm, aleatoria)
    no_escritos = 0
    bordes = np.zeros(aleatoria.shape, dtype=bool)
    bordes[:4, :] = bordes[-4:, :] = bordes[:, :4] = bordes[:, -4:] = True
    emax_bordes = 0.0
    for filtro, k in [("media", 3), ("nitidez", 3), ("laplaciano", 3), ("media", 9)]:
        m = cv.mascara(filtro, k)
        ref = cv.ejecutar_c("c", pgm, filtro, k)
        versiones = {"CUDA C": cv.ejecutar_c("cuda", pgm, filtro, k)}
        if k == 3:
            versiones["Python secuencial"] = cv.convolucion_python(aleatoria, m)
        for hilos in (128, 256, 512):
            val = pc.ejecutar(aleatoria, m, hilos, envenenar=True)
            no_escritos += int(np.count_nonzero(np.isnan(val)))
            emax_bordes = max(emax_bordes, cv.errores(ref[bordes], val[bordes])[0])
            versiones[f"PyCUDA {hilos} hilos"] = val
        comparar(f"1001x777, {etiqueta(filtro, k)}", m, ref, versiones)
    log.say(f"  Prueba de limites (1001x777, no multiplo de los bloques): pixeles sin escribir = {no_escritos}, "
            f"error maximo en los bordes = {emax_bordes:.3g}")

    # 3) Comparacion numerica con las imagenes de la practica
    log.detalle("\n== Comparacion numerica con las imagenes de la practica ==")
    ruta500, img500 = cargar(500)
    for filtro, k in [("media", 3), ("nitidez", 3), ("laplaciano", 3), ("media", 9), ("media", 21)]:
        m = cv.mascara(filtro, k)
        ref = cv.ejecutar_c("c", ruta500, filtro, k)
        comparar(f"500x500, {etiqueta(filtro, k)}", m, ref, {
            "CUDA C": cv.ejecutar_c("cuda", ruta500, filtro, k),
            "Python secuencial": cv.convolucion_python(img500, m),
            "PyCUDA": pc.ejecutar(img500, m, 256)})
    ruta4k, img4k = cargar(4000)
    salidas = {}
    for filtro, k in [("media", 3), ("nitidez", 3), ("laplaciano", 3), ("media", 9), ("media", 21)]:
        m = cv.mascara(filtro, k)
        ref = cv.ejecutar_c("c", ruta4k, filtro, k)
        val = pc.ejecutar(img4k, m, 256)
        comparar(f"4000x4000, {etiqueta(filtro, k)}", m, ref, {"CUDA C": cv.ejecutar_c("cuda", ruta4k, filtro, k),
                                                              "PyCUDA": val})
        if k == 3:
            salidas[filtro] = val
    log.detalle("  (Python secuencial no se compara en 4000x4000: ver la viabilidad en 'medir')")
    log.say("  Comparacion numerica (500x500 con las 4 versiones; 4000x4000 con C++, CUDA C y PyCUDA):")
    for version in ("Python secuencial", "CUDA C", "PyCUDA", "PyCUDA 128 hilos", "PyCUDA 256 hilos",
                    "PyCUDA 512 hilos"):
        if version in resumen:
            n, ok, emax = resumen[version]
            log.say(f"      {version:<20} {ok:3d} de {n:3d} casos dentro de la tolerancia, error maximo {emax:.3g}")

    # 4) Evidencia visual: imagen original y los tres filtros (salida de PyCUDA)
    carpeta = RES / "images"
    carpeta.mkdir(exist_ok=True)
    x, y, s = 2700, 900, 512  # mismo recorte del ojo que en la practica 01
    cv.escribir_png(carpeta / "entrada_vista.png", cv.reducir(img4k))
    cv.escribir_png(carpeta / "entrada_recorte.png", img4k[y:y + s, x:x + s])
    for filtro, val in salidas.items():
        img8 = cv.a_8bits(val, filtro)
        cv.escribir_png(carpeta / f"{filtro}3x3_vista.png", cv.reducir(img8))
        cv.escribir_png(carpeta / f"{filtro}3x3_recorte.png", img8[y:y + s, x:x + s])
    log.say(f"  Imagenes de evidencia (original y 3 filtros, vista reducida y recorte): {carpeta}")

    log.say(f"\n  Resultado: {'las cuatro versiones dan el mismo resultado' if fallas[0] == 0 else 'HAY FALLAS'}"
            f"   (detalle en {log.ruta})")
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    log.cerrar()
    pc.cerrar()
    return 0 if fallas[0] == 0 else 1


# ----------------------------------------------------------------------------------------------
# medir: experimentos con las cuatro versiones
# ----------------------------------------------------------------------------------------------
EXPERIMENTOS = [
    # (experimento, filtro, k, lado de la imagen, hilos por bloque)
    (1, "media", 3, 4000, 256), (1, "nitidez", 3, 4000, 256), (1, "laplaciano", 3, 4000, 256),
    (2, "media", 9, 2000, 256), (2, "media", 9, 4000, 256), (2, "media", 9, 6000, 256),
    (3, "media", 3, 4000, 256), (3, "media", 9, 4000, 256), (3, "media", 21, 4000, 256),
    (4, "media", 9, 4000, 128), (4, "media", 9, 4000, 256), (4, "media", 9, 4000, 512),
]
# Comparacion de las cuatro versiones en una resolucion menor comun (Python si alcanza a terminar)
REDUCIDA = [(1, "media", 3, 500, 256), (1, "nitidez", 3, 500, 256), (1, "laplaciano", 3, 500, 256),
            (3, "media", 9, 500, 256), (3, "media", 21, 500, 256)]


def tiempos_c(ruta):
    """Lee los tiempos de C++ y CUDA C (tiempos_crudos.csv de bin/benchmark.exe), sin el calentamiento."""
    t = {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if int(fila["repeticion"]) == 0:
                continue
            lado = int(fila["resolucion"].split("x")[0])
            k = int(fila["mascara"].split("x")[0])
            hilos = int(fila["hilos_por_bloque"])
            clave = (int(fila["experimento"]), fila["implementacion"], fila["filtro"], k, lado, hilos)
            t.setdefault(clave, {}).setdefault(fila["metrica"], []).append(float(fila["tiempo"]))
            t.setdefault(("filas",), []).append(fila)
    return t


def correr_benchmark_c(args_extra, salida, log):
    exe = cv.BIN / "benchmark.exe"
    if not exe.exists():
        sys.exit("Falta bin\\benchmark.exe: ejecute primero build.bat")
    salida.mkdir(parents=True, exist_ok=True)
    log.say(f"  Ejecutando la version C/C++ y CUDA C (bin\\benchmark.exe {' '.join(args_extra)}) ...")
    r = subprocess.run([str(exe), "--no-images", "--out", str(salida)] + args_extra, cwd=cv.RAIZ,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"benchmark.exe fallo:\n{r.stdout}\n{r.stderr}")


def cmd_medir(a):
    if a.rapido:
        a.reps_pycuda, a.reps_python, a.limite = 3, 2, 20
    log = Registro("medicion_python.txt")
    aviso_bateria(log)
    # Evita que Windows suspenda el equipo mientras se mide (si se suspende, los tiempos no sirven).
    # No impide la suspension al cerrar la tapa de la laptop.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    pc = PyCuda()
    info = cv.entorno(pc)
    log.say("  EQUIPO: " + f"{info.get('CPU')} | {info.get('Nucleos')} | RAM {info.get('RAM')}")
    log.say("          " + f"{info.get('GPU')} ({info.get('Memoria GPU')}) | controlador {info.get('Controlador NVIDIA')}"
            f" | CUDA {info.get('CUDA Toolkit')}")
    log.say("          " + f"Python {info['Python']} | NumPy {info['NumPy']} | PyCUDA {info['PyCUDA']}")
    log.say(f"  PyCUDA: inicializar el contexto {pc.ms_contexto:.0f} ms, compilar y cargar el kernel "
            f"{pc.ms_compilacion:.0f} ms (se hace una sola vez, fuera de las mediciones)")
    log.say(f"  Repeticiones: PyCUDA {a.reps_pycuda} (calentamiento >= 2 s), Python {a.reps_python}; "
            f"limite por corrida de Python: {a.limite:.0f} s")

    filas_csv = []

    def anotar(version, exp, filtro, k, lado, hilos, rep, metrica, ms, estado="completada"):
        bloques = pc.bloques(lado * lado, hilos) if hilos else 0
        filas_csv.append([version, exp, filtro, f"img_{lado}", f"{lado}x{lado}", f"{k}x{k}", hilos, bloques, rep,
                          metrica, f"{ms:.6f}", "ms", estado])

    anotar("pycuda", 0, "-", 0, 0, 0, 0, "inicializacion_contexto", pc.ms_contexto)
    anotar("pycuda", 0, "-", 0, 0, 0, 0, "compilacion_kernel", pc.ms_compilacion)

    # --- C/C++ y CUDA C: se vuelven a medir en esta misma sesion para que las cuatro versiones tengan las
    # mismas condiciones del equipo. Con --reusar-c se usan los tiempos ya guardados de la practica 01.
    ruta_c = RES / "c" / "tiempos_crudos.csv"
    if not a.reusar_c or not ruta_c.exists():
        correr_benchmark_c([], RES / "c", log)
    ruta_c500 = RES / "c_500" / "tiempos_crudos.csv"
    if not a.reusar_c or not ruta_c500.exists():
        correr_benchmark_c(["--imagenes", "img_500", "--base", "img_500", "--exp", "1,3"], RES / "c_500", log)
    tc = tiempos_c(ruta_c)
    tc500 = tiempos_c(ruta_c500)

    imagenes = {}

    def imagen(lado):
        if lado not in imagenes:
            imagenes[lado] = cargar(lado)
        return imagenes[lado]

    referencias = {}

    def referencia(filtro, k, lado):
        clave = (filtro, k, lado)
        if clave not in referencias:
            referencias.clear()  # se guarda solo una para no llenar la memoria
            referencias[clave] = cv.ejecutar_c("c", imagen(lado)[0], filtro, k)
        return referencias[clave]

    # --- PyCUDA
    log.say("\n  [1/2] Midiendo PyCUDA ...")
    r_pycuda = {}
    for exp, filtro, k, lado, hilos in EXPERIMENTOS + REDUCIDA:
        clave = (filtro, k, lado, hilos)
        if clave in r_pycuda:  # misma configuracion en otro experimento: se comparte la medicion
            continue
        _, img = imagen(lado)
        m = cv.mascara(filtro, k)
        ref = referencia(filtro, k, lado)
        # Si la GPU cambia de velocidad a mitad de la medicion (la laptop baja el reloj), las repeticiones
        # varian mucho: en ese caso se repite la medicion de esta configuracion (hasta 3 intentos).
        for intento in range(1, 4):
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0:  # calentamiento de la GPU
                pc.medir_separado(img, m, hilos)
            med = {"h2d": [], "kernel": [], "d2h": [], "total": [], "total_inout": []}
            # La comparacion con C++ se hace DESPUES de cada serie y no entre repeticiones: compararla tarda
            # ~0.3 s en la CPU y, mientras tanto, la GPU queda sin trabajo y baja su velocidad.
            for rep in range(1, a.reps_pycuda + 1):  # copias y kernel medidos por separado
                t, val = pc.medir_separado(img, m, hilos)
                for metrica, ms in t.items():
                    med[metrica].append(ms)
            emax = cv.errores(ref, val)[0]
            for rep in range(1, a.reps_pycuda + 1):  # todo junto con drv.In / drv.Out
                ms, val = pc.medir_inout(img, m, hilos)
                med["total_inout"].append(ms)
            emax = max(emax, cv.errores(ref, val)[0])
            variacion = de(med["kernel"]) / mediana(med["kernel"])
            if variacion <= 0.10:
                break
            log.say(f"      {etiqueta(filtro, k):<15} {lado}x{lado} {hilos:3d} hilos: las repeticiones variaron "
                    f"{variacion:.0%} (la GPU cambio de velocidad); se repite la medicion")
        r_pycuda[clave] = (med, emax <= cv.tolerancia(m))
        for e, f_, k_, l_, h_ in EXPERIMENTOS + REDUCIDA:
            if (f_, k_, l_, h_) == clave:
                for metrica, lista in med.items():
                    for rep, ms in enumerate(lista, 1):
                        anotar("pycuda", e, filtro, k, lado, hilos, rep, metrica, ms)
        log.say(f"      {etiqueta(filtro, k):<15} {lado}x{lado} {hilos:3d} hilos: kernel {mediana(med['kernel']):8.3f} ms,"
                f" total {mediana(med['total']):8.2f} ms, In/Out {mediana(med['total_inout']):8.2f} ms")

    # --- Python secuencial
    log.say(f"\n  [2/2] Midiendo Python secuencial (es lento: hasta {a.limite:.0f} s por corrida) ...")
    r_python = {}
    for exp, filtro, k, lado, hilos in EXPERIMENTOS + REDUCIDA:
        clave = (filtro, k, lado)
        if exp == 4 or clave in r_python:
            continue
        _, img = imagen(lado)
        m = cv.mascara(filtro, k)
        # prueba pequena (tambien sirve de calentamiento) para estimar el tiempo de la imagen completa
        trozo = img[:96, :96]
        t0 = time.perf_counter()
        cv.convolucion_python(trozo, m)
        estimado = (time.perf_counter() - t0) * img.size / trozo.size
        tiempos, estado, transcurrido = [], "completada", 0.0
        print(f"      {etiqueta(filtro, k):<15} {lado}x{lado}: estimado {segundos(estimado)} por corrida ...",
              end="", flush=True)
        try:
            for rep in range(1, a.reps_python + 1):
                t0 = time.perf_counter()
                val = cv.convolucion_python(img, m, limite_s=a.limite if rep == 1 else None)
                tiempos.append((time.perf_counter() - t0) * 1000.0)
                print(f" {rep}", end="", flush=True)
        except cv.Interrumpida as e:
            estado, transcurrido = "no_completada", e.segundos
        print("\r" + " " * 100 + "\r", end="", flush=True)
        if estado == "completada":
            igual = cv.errores(referencia(filtro, k, lado), val)[0] <= cv.tolerancia(m)
            log.say(f"      {etiqueta(filtro, k):<15} {lado}x{lado}: {len(tiempos)} corridas, mediana "
                    f"{mediana(tiempos) / 1000:8.2f} s")
        else:
            igual = None
            log.say(f"      {etiqueta(filtro, k):<15} {lado}x{lado}: NO COMPLETADA (se detuvo a los {transcurrido:.0f} s;"
                    f" la prueba pequena estimaba unos {segundos(estimado)} por corrida)")
        r_python[clave] = (tiempos, estado, transcurrido, estimado, igual)
        for e, f_, k_, l_, h_ in EXPERIMENTOS + REDUCIDA:
            if (f_, k_, l_) == clave and e != 4:
                if estado == "completada":
                    for rep, ms in enumerate(tiempos, 1):
                        anotar("python", e, filtro, k, lado, 0, rep, "cpu_total", ms)
                else:
                    anotar("python", e, filtro, k, lado, 0, 1, "cpu_total", transcurrido * 1000.0, "no_completada")
    pc.cerrar()

    # --- C/C++ y CUDA C al CSV (los mismos datos de la practica 01, para tener las 4 versiones juntas)
    for tabla in (tc, tc500):
        for fila in tabla[("filas",)]:
            lado = int(fila["resolucion"].split("x")[0])
            k = int(fila["mascara"].split("x")[0])
            hilos = int(fila["hilos_por_bloque"])
            exp = int(fila["experimento"])
            if (exp, fila["filtro"], k, lado, hilos if hilos else 256) in [(e, f, kk, l, h) for e, f, kk, l, h in
                                                                            EXPERIMENTOS + REDUCIDA] or \
                    (fila["implementacion"] == "cpu" and any((e, f, kk, l) == (exp, fila["filtro"], k, lado)
                                                             for e, f, kk, l, _ in EXPERIMENTOS + REDUCIDA)):
                version = "c" if fila["implementacion"] == "cpu" else "cuda_c"
                if fila["metrica"] == "total_wall":
                    continue
                filas_csv.append([version, exp, fila["filtro"], fila["imagen"], fila["resolucion"], fila["mascara"],
                                  hilos, fila["bloques"], fila["repeticion"], fila["metrica"], fila["tiempo"], "ms",
                                  "completada"])
    ruta_csv = RES / "tiempos_4_versiones.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "experimento", "filtro", "imagen", "resolucion", "mascara", "hilos_por_bloque",
                    "bloques", "repeticion", "metrica", "tiempo", "unidad", "estado"])
        w.writerows(filas_csv)

    # --- Tablas
    def fila_tiempos(exp, filtro, k, lado, hilos, tabla_c):
        cpu = tabla_c.get((exp, "cpu", filtro, k, lado, 0), {}).get("cpu_total", [])
        cud = tabla_c.get((exp, "cuda", filtro, k, lado, hilos), {})
        med, igual_pyc = r_pycuda[(filtro, k, lado, hilos)]
        py = r_python.get((filtro, k, lado))
        return {"c": mediana(cpu), "cuda_k": mediana(cud.get("kernel", [])), "cuda_t": mediana(cud.get("total", [])),
                "pyc_k": mediana(med["kernel"]), "pyc_t": mediana(med["total"]), "pyc_io": mediana(med["total_inout"]),
                "pyc_cv": 100 * de(med["kernel"]) / mediana(med["kernel"]),
                "py": mediana(py[0]) if py and py[1] == "completada" else None,
                "igual_py": None if not py else py[4], "igual_pyc": igual_pyc}

    def txt(v, fmt):
        return format(v, fmt) if v is not None else "-"

    def veces(a_, b_):
        return f"{a_ / b_:.1f}x" if a_ is not None and b_ is not None else "-"

    def sino(v):
        return "-" if v is None else ("SI" if v else "NO")

    titulos = {1: "Tipo de filtro", 2: "Tamano de la imagen", 3: "Tamano de la mascara"}
    acel_k, acel_t = {"cuda": [], "pyc": []}, {"cuda": [], "pyc": []}
    for nombre, lista, tabla_c in [("", EXPERIMENTOS, tc), (" - RESOLUCION MENOR COMUN", REDUCIDA, tc500)]:
        for exp in (1, 2, 3):
            confs = [c for c in lista if c[0] == exp]
            if not confs:
                continue
            fijo = {1: f"imagen {confs[0][3]}x{confs[0][3]}, 256 hilos", 2: "media 9x9, 256 hilos",
                    3: f"imagen {confs[0][3]}x{confs[0][3]}, 256 hilos"}[exp]
            log.say(f"\n  EXPERIMENTO {exp} - {titulos[exp]}{nombre} ({fijo})")
            log.say(f"  {'':16} {'---- Secuencial ----':>22} {'--- GPU solo calculo ---':>26} "
                    f"{'---------- GPU con copias ----------':>38}")
            log.say(f"  {'Prueba':<16} {'C++ (ms)':>10} {'Python (ms)':>12} {'CUDA C':>12} {'PyCUDA':>12}   "
                    f"{'CUDA C':>10} {'PyCUDA':>10} {'In/Out':>10}   {'Igual Py/PyC':>12}")
            log.say("  " + "-" * 126)
            filas = []
            for _, filtro, k, lado, hilos in confs:
                r = fila_tiempos(exp, filtro, k, lado, hilos, tabla_c)
                filas.append((filtro, k, lado, r))
                prueba = f"{lado}x{lado}" if exp == 2 else etiqueta(filtro, k)
                py = txt(r["py"], "12.1f") if r["py"] is not None else f"{'NO COMPL.':>12}"
                log.say(f"  {prueba:<16} {txt(r['c'], '10.2f')} {py} {txt(r['cuda_k'], '12.3f')} "
                        f"{txt(r['pyc_k'], '12.3f')}   {txt(r['cuda_t'], '10.2f')} {txt(r['pyc_t'], '10.2f')} "
                        f"{txt(r['pyc_io'], '10.2f')}   {sino(r['igual_py']) + '/' + sino(r['igual_pyc']):>12}")
            log.say(f"  {'Aceleracion':<16} {'CUDA C vs C++':>22} {'PyCUDA vs Python':>24} "
                    f"{'Tiempo PyCUDA / CUDA C':>30}")
            log.say(f"  {'':16} {'calculo':>11} {'con copias':>11} {'calculo':>11} {'con copias':>12} "
                    f"{'calculo':>14} {'con copias':>15}")
            for filtro, k, lado, r in filas:
                prueba = f"{lado}x{lado}" if exp == 2 else etiqueta(filtro, k)
                log.say(f"  {prueba:<16} {veces(r['c'], r['cuda_k']):>11} {veces(r['c'], r['cuda_t']):>11} "
                        f"{veces(r['py'], r['pyc_k']):>11} {veces(r['py'], r['pyc_t']):>12} "
                        f"{txt(r['pyc_k'] / r['cuda_k'] if r['cuda_k'] else None, '14.2f')} "
                        f"{txt(r['pyc_t'] / r['cuda_t'] if r['cuda_t'] else None, '15.2f')}")
                if r["c"] and r["cuda_k"]:
                    acel_k["cuda"].append(r["c"] / r["cuda_k"])
                    acel_t["cuda"].append(r["c"] / r["cuda_t"])
                if r["py"]:
                    acel_k["pyc"].append(r["py"] / r["pyc_k"])
                    acel_t["pyc"].append(r["py"] / r["pyc_t"])

    log.say("\n  EXPERIMENTO 4 - Hilos por bloque (media 9x9, imagen 4000x4000; solo CUDA C y PyCUDA)")
    log.say(f"  {'Hilos/bloque':<16} {'CUDA C calc':>12} {'PyCUDA calc':>12} {'CV PyCUDA':>10}   "
            f"{'CUDA C total':>13} {'PyCUDA total':>13} {'In/Out':>10}")
    log.say("  " + "-" * 94)
    for _, filtro, k, lado, hilos in [c for c in EXPERIMENTOS if c[0] == 4]:
        r = fila_tiempos(4, filtro, k, lado, hilos, tc)
        log.say(f"  {str(hilos) + ' hilos':<16} {txt(r['cuda_k'], '12.3f')} {txt(r['pyc_k'], '12.3f')} "
                f"{r['pyc_cv']:9.2f}%   {txt(r['cuda_t'], '13.2f')} {txt(r['pyc_t'], '13.2f')} {txt(r['pyc_io'], '10.2f')}")

    log.say("\n  RESUMEN")
    if acel_k["cuda"]:
        log.say(f"  CUDA C fue entre {min(acel_k['cuda']):.1f} y {max(acel_k['cuda']):.1f} veces mas rapido que C++ "
                f"calculando, y entre {min(acel_t['cuda']):.1f} y {max(acel_t['cuda']):.1f} con las copias.")
    if acel_k["pyc"]:
        log.say(f"  PyCUDA fue entre {min(acel_k['pyc']):.0f} y {max(acel_k['pyc']):.0f} veces mas rapido que Python "
                f"calculando, y entre {min(acel_t['pyc']):.0f} y {max(acel_t['pyc']):.0f} con las copias.")
    no_comp = [(c, v) for c, v in r_python.items() if v[1] == "no_completada"]
    if no_comp:
        log.say("  Python secuencial NO COMPLETADO (sin aceleracion, no se estima):")
        for (filtro, k, lado), v in no_comp:
            log.say(f"      {etiqueta(filtro, k)} {lado}x{lado}: se detuvo a los {v[2]:.0f} s por superar el limite "
                    f"de {a.limite:.0f} s por corrida")
    log.say("  'Igual Py/PyC': Python y PyCUDA dan el mismo resultado que C++ (dentro de la tolerancia).")
    log.say(f"  'In/Out': PyCUDA con drv.In/drv.Out (incluye reservar y liberar la memoria de la GPU en cada llamada).")
    log.say(f"\n  Guardado en: {log.ruta} y {ruta_csv}")
    log.cerrar()
    return 0


def main():
    p = argparse.ArgumentParser(description="Convolucion en Python secuencial y PyCUDA")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("entorno")
    f = sub.add_parser("filtrar")
    f.add_argument("--imagen", default="4000", help="ruta de la imagen o lado (2000, 4000, 6000, 500)")
    f.add_argument("--filtro", choices=cv.FILTROS, default="media")
    f.add_argument("--k", type=int, default=3)
    f.add_argument("--version", choices=["pycuda", "python"], default="pycuda")
    f.add_argument("--hilos", type=int, default=256)
    f.add_argument("--salida")
    sub.add_parser("validar")
    m = sub.add_parser("medir")
    m.add_argument("--reps-pycuda", type=int, default=10)
    m.add_argument("--reps-python", type=int, default=5)
    m.add_argument("--limite", type=float, default=120.0, help="segundos maximos por corrida de Python")
    m.add_argument("--reusar-c", action="store_true",
                   help="no volver a medir C/C++ y CUDA C (usar los tiempos de la ultima medicion)")
    m.add_argument("--rapido", action="store_true", help="prueba corta (pocas repeticiones, limite de 20 s)")
    a = p.parse_args()
    if a.cmd in ("filtrar",) and a.filtro != "media":
        a.k = 3
    return {"entorno": cmd_entorno, "filtrar": cmd_filtrar, "validar": cmd_validar, "medir": cmd_medir}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
