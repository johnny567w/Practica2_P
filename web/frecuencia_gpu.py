"""Prueba de control: tiempo del kernel con lanzamientos seguidos y con pausas entre lanzamientos.

En el servicio, entre un kernel y el siguiente hay cientos de milisegundos de trabajo en la CPU (decodificar,
pasar a 8 bits, codificar el PNG). Durante esas pausas la GTX 1650 de la laptop baja su frecuencia y el
kernel se ejecuta antes de que vuelva a subir. Esta prueba lo comprueba con la misma imagen y la misma
mascara (media 3x3, 4000x4000), sin HTTP.

Uso:  py -3.13 web\\frecuencia_gpu.py      -> results\\web\\frecuencia_gpu.txt
"""
import statistics
import time

import numpy as np

import procesamiento as pr
import convolucion as cv

REPS = 10


def main():
    motor = pr.MotorGPU()
    img = np.fromfile(pr.RAIZ / "data" / "img_4000.pgm", dtype=np.uint8)[-4000 * 4000:].reshape(4000, 4000)
    m = cv.mascara("media", 3)
    t = time.perf_counter()
    while time.perf_counter() - t < 3:  # calentamiento
        motor.convolucion(img, m)
    casos = {}
    casos["seguidos (sin pausa)"] = [motor.convolucion(img, m)[1]["kernel_ms"] for _ in range(REPS)]
    for pausa in (0.1, 0.5):
        r = []
        for _ in range(REPS):
            time.sleep(pausa)
            r.append(motor.convolucion(img, m)[1]["kernel_ms"])
        casos[f"pausa de {pausa} s antes de cada uno"] = r
    motor.cerrar()

    lineas = ["Tiempo del kernel (eventos CUDA, ms), media 3x3, 4000x4000, 256 hilos por bloque",
              f"GPU: {motor.gpu}", ""]
    for nombre, r in casos.items():
        lineas.append(f"{nombre:36s} mediana {statistics.median(r):7.2f}  min {min(r):6.2f}  max {max(r):6.2f}"
                      f"  | {' '.join(f'{x:.2f}' for x in r)}")
    lineas += ["", "Con lanzamientos seguidos la GPU mantiene su frecuencia alta; con pausas (como entre solicitudes",
               "del servicio) la baja y el mismo kernel tarda varias veces mas."]
    texto = "\n".join(lineas)
    print(texto)
    salida = pr.RAIZ / "results" / "web" / "frecuencia_gpu.txt"
    salida.write_text(texto + "\n", encoding="utf-8")
    print(f"\nGuardado: {salida}")


if __name__ == "__main__":
    main()
