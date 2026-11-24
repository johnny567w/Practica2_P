"""Ejecución DIRECTA del procesamiento (sin HTTP): la misma función procesar() que usa el servicio.

Sirve para el experimento de integración: PyCUDA directo en el anfitrión, PyCUDA directo dentro del
contenedor y solicitud al servicio contenerizado, con el mismo hardware, los mismos archivos y los mismos
hilos por bloque.

Uso:
  py -3.13 web\\directo.py --modo anfitrion                         (en Windows)
  docker run --rm --gpus all -v <data>:/app/data -v <results>:/app/results convolucion-web:1.0 \\
         python3 directo.py --modo contenedor                        (dentro del contenedor)
"""
import argparse
import csv
import datetime
import hashlib
import platform
import statistics
import time
from pathlib import Path

import procesamiento as pr

CAMPOS = ["id_solicitud", "modo", "filtro", "resolucion", "mascara", "hilos", "bloques", "repeticion", "concurrencia",
          "ronda", "cliente_ms", "servidor_ms", "espera_cola_ms", "decodificacion_ms", "preparacion_ms", "reserva_ms",
          "h2d_ms", "kernel_ms", "d2h_ms", "gpu_total_ms", "salida_ms", "escritura_ms", "procesamiento_ms",
          "ronda_ms", "solicitudes_por_s", "unidad", "estado", "codigo_http", "entorno", "fecha"]

# (experimento, filtro, k, lado). Las configuraciones repetidas se miden una vez y se anotan en cada experimento.
CONFIGURACIONES = [
    ("integracion", "media", 9, 4000),
    ("resolucion", "media", 9, 2000), ("resolucion", "media", 9, 4000), ("resolucion", "media", 9, 6000),
    ("filtros", "media", 3, 4000), ("filtros", "media", 9, 4000), ("filtros", "media", 21, 4000),
    ("filtros", "nitidez", 3, 4000), ("filtros", "laplaciano", 3, 4000),
]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def fila_base(modo, filtro, k, lado, hilos, bloques, rep, entorno, estado="completada"):
    f = dict.fromkeys(CAMPOS, "")
    f.update(modo=modo, filtro=filtro, resolucion=f"{lado}x{lado}", mascara=f"{k}x{k}", hilos=hilos, bloques=bloques,
             repeticion=rep, concurrencia=1, ronda="", unidad="ms", estado=estado, entorno=entorno,
             fecha=datetime.datetime.now().isoformat(timespec="seconds"))
    return f


def main():
    p = argparse.ArgumentParser(description="Procesamiento directo con PyCUDA (sin servicio)")
    p.add_argument("--modo", default="anfitrion", help="anfitrion o contenedor (se anota en el CSV)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--calentamiento-s", type=float, default=2.0)
    p.add_argument("--data", default=str(pr.RAIZ / "data"))
    p.add_argument("--salida", default=str(pr.RAIZ / "results" / "web"))
    p.add_argument("--solo-integracion", action="store_true", help="solo media 9x9 en 4000x4000")
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    entorno = f"{a.modo} | {platform.system()} {platform.release()} | Python {platform.python_version()}"

    t0 = time.perf_counter()
    motor = pr.MotorGPU()
    ini = motor.inicializacion
    print(f"  GPU: {motor.gpu}")
    print(f"  Arranque en frio: contexto {ini['contexto_ms']:.0f} ms, compilar y cargar el kernel "
          f"{ini['compilacion_kernel_ms']:.0f} ms (total {ini['total_ms']:.0f} ms, fuera de las mediciones)")
    filas = []
    f = fila_base(a.modo, "-", 0, 0, 0, 0, 0, entorno, "arranque_en_frio")
    f.update(procesamiento_ms=f"{ini['total_ms']:.4f}", preparacion_ms=f"{ini['compilacion_kernel_ms']:.4f}")
    filas.append(f)

    configs = CONFIGURACIONES[:1] if a.solo_integracion else CONFIGURACIONES
    medidas = {}
    archivos = {}
    for exp, filtro, k, lado in configs:
        ruta = Path(a.data) / f"img_{lado}.pgm"
        if lado not in archivos:
            archivos[lado] = ruta.read_bytes()
            print(f"  {ruta.name}: sha256 {sha256(ruta)[:16]}...")
        datos = archivos[lado]
        clave = (filtro, k, lado)
        if clave not in medidas:
            primera = pr.procesar(motor, datos, filtro, k)  # primera ejecucion de esta configuracion (se anota aparte)
            t = time.perf_counter()
            while time.perf_counter() - t < a.calentamiento_s:  # calentamiento de la GPU
                pr.procesar(motor, datos, filtro, k)
            reps = [pr.procesar(motor, datos, filtro, k) for _ in range(a.reps)]
            medidas[clave] = (primera, reps)
            ker = [r["tiempos"]["kernel_ms"] for r in reps]
            proc = [r["tiempos"]["procesamiento_ms"] for r in reps]
            print(f"  {filtro} {k}x{k} {lado}x{lado}: kernel {statistics.median(ker):8.3f} ms "
                  f"(DE {statistics.stdev(ker):.3f}), procesamiento {statistics.median(proc):8.1f} ms")
        primera, reps = medidas[clave]
        for rep, r in [(0, primera)] + list(enumerate(reps, 1)):
            fila = fila_base(a.modo, filtro, k, lado, r["hilos"], r["bloques"], rep, entorno,
                             "primera_ejecucion" if rep == 0 else "completada")
            fila["id_solicitud"] = f"{a.modo}-{exp}-{filtro}{k}-{lado}-{rep}"
            for c, v in r["tiempos"].items():
                if c in fila:
                    fila[c] = f"{v:.4f}"
            filas.append(fila)
    motor.cerrar()
    ruta_csv = salida / f"tiempos_directo_{a.modo}.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(f"  Guardado: {ruta_csv}  ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
