"""Cliente de pruebas del servicio web: validación funcional y numérica, y experimentos.

Uso (desde Practica_2, con el servicio en marcha):
  py -3.13 web\\cliente.py validar --url http://localhost:8000
  py -3.13 web\\cliente.py medir   --url http://localhost:8000 --entorno "cliente en el anfitrion"

El tiempo del cliente va desde que se envía la solicitud hasta recibir la imagen resultado completa
(incluye la comunicación y la descarga del PNG). No se restan relojes de máquinas distintas: cada tiempo se
mide en una sola máquina (el cliente con su reloj y el servidor con el suyo).
"""
import argparse
import csv
import datetime
import io
import json
import urllib.parse
import statistics
import sys
import threading
import time
import uuid
from http.client import HTTPConnection
from pathlib import Path

import numpy as np

import directo
import procesamiento as pr

RES = pr.RAIZ / "results" / "web"


# ----------------------------------------------------------------------------------------------
# HTTP sin dependencias (urllib)
# ----------------------------------------------------------------------------------------------
def multipart(campos, archivo, nombre="imagen.pgm"):
    limite = uuid.uuid4().hex
    partes = []
    for k, v in campos.items():
        partes.append(f'--{limite}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    partes.append(f'--{limite}\r\nContent-Disposition: form-data; name="imagen"; filename="{nombre}"\r\n'
                  f'Content-Type: application/octet-stream\r\n\r\n'.encode() + archivo + b"\r\n")
    partes.append(f"--{limite}--\r\n".encode())
    return b"".join(partes), f"multipart/form-data; boundary={limite}"


def http(url, datos=None, tipo=None, timeout=900):
    """GET o POST. Si el servidor rechaza un archivo muy grande (413) antes de terminar de recibirlo, cierra la
    conexion a mitad del envio: en ese caso se deja de enviar y se lee igual la respuesta que mando."""
    u = urllib.parse.urlsplit(url)
    conn = HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    ruta = u.path + (f"?{u.query}" if u.query else "")
    try:
        conn.putrequest("POST" if datos is not None else "GET", ruta)
        if datos is not None:
            conn.putheader("Content-Type", tipo)
            conn.putheader("Content-Length", str(len(datos)))
        conn.endheaders()
        if datos is not None:
            try:
                for i in range(0, len(datos), 1 << 20):
                    conn.send(datos[i:i + (1 << 20)])
            except (ConnectionError, OSError):
                pass  # el servidor ya respondio y cerro: se lee su respuesta abajo
        try:
            r = conn.getresponse()
            return r.status, r.read()
        except (ConnectionError, OSError) as e:
            return 0, json.dumps({"ok": False, "error": f"conexion cerrada por el servidor ({e})"}).encode()
    finally:
        conn.close()


def solicitar(url, archivo, filtro, k, devolver_float=False, nombre="imagen.pgm"):
    """Envía una imagen y descarga el resultado. Devuelve (codigo, json, cliente_ms, png, valores)."""
    cuerpo, tipo = multipart({"filtro": filtro, "k": k, "devolver_float": "1" if devolver_float else "0"},
                             archivo, nombre)
    t0 = time.perf_counter()
    codigo, resp = http(f"{url}/api/procesar", cuerpo, tipo)
    try:
        d = json.loads(resp)
    except ValueError:
        d = {"ok": False, "error": resp[:200].decode(errors="replace")}
    png = None
    if codigo == 200:
        _, png = http(url + d["archivos"]["resultado"])
    cliente_ms = (time.perf_counter() - t0) * 1000.0  # hasta tener la imagen final completa
    valores = None
    if codigo == 200 and devolver_float:
        _, crudo = http(url + d["archivos"]["valores_float32"])
        valores = np.frombuffer(crudo, dtype=np.float32).reshape(d["alto"], d["ancho"])
    return codigo, d, cliente_ms, png, valores


class Registro:
    def __init__(self, nombre):
        RES.mkdir(parents=True, exist_ok=True)
        self.ruta = RES / nombre
        self.f = open(self.ruta, "w", encoding="utf-8")

    def say(self, t=""):
        print(t, flush=True)
        self.f.write(t + "\n")
        self.f.flush()


def pgm(img):
    return b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]) + np.ascontiguousarray(img, dtype=np.uint8).tobytes()


# ----------------------------------------------------------------------------------------------
# Validación funcional y numérica
# ----------------------------------------------------------------------------------------------
def validar(a):
    url = a.url.rstrip("/")
    log = Registro("validacion_web.txt")
    fallas = []

    def chequeo(ok, texto):
        log.say(f"    [{'OK' if ok else 'FALLA'}] {texto}")
        if not ok:
            fallas.append(texto)

    codigo, cuerpo = http(f"{url}/api/salud")
    salud = json.loads(cuerpo)
    log.say(f"  Servicio: {url}  |  GPU: {salud.get('gpu', salud.get('error'))}  |  cola maxima {salud.get('cola_max')}")
    chequeo(codigo == 200, "el servicio responde y tiene la GPU lista")

    log.say("  Referencia: PyCUDA directo validado (practica 02), ejecutado en este equipo con el mismo kernel.")
    motor = pr.MotorGPU()
    try:
        return _validar(a, url, log, fallas, chequeo, salud, motor)
    finally:
        motor.cerrar()  # libera el contexto CUDA aunque una prueba falle


def _validar(a, url, log, fallas, chequeo, salud, motor):

    def directo_valores(img, filtro, k):
        return motor.convolucion(img, pr.cv.mascara(filtro, k))[0]

    def comparar(caso, img, filtro, k, valores):
        ref = directo_valores(img, filtro, k)
        emax, emed, nf = pr.cv.errores(ref, valores)
        tol = pr.cv.tolerancia(pr.cv.mascara(filtro, k))
        chequeo(nf == 0 and emax <= tol, f"{caso:<34} err.max {emax:9.3g}  err.medio {emed:9.3g}  tol {tol:9.3g}  "
                                         f"no escritos {nf}")
        return emax

    # 1) Matriz pequeña y límites
    log.say("\n  1) Matriz pequeña (4x5 de la guia) y limites")
    matriz = np.array([[35, 40, 41, 45, 50], [40, 40, 42, 46, 52], [42, 46, 50, 55, 55], [48, 52, 58, 60, 61]],
                      dtype=np.uint8)
    esperado = {("media", 3): ((1, 1), 41.7778), ("nitidez", 3): ((1, 1), 32.0), ("laplaciano", 3): ((0, 0), -60.0)}
    for (filtro, k), (pos, valor) in esperado.items():
        c, d, _, _, val = solicitar(url, pgm(matriz), filtro, k, True)
        if c != 200:
            chequeo(False, f"matriz 4x5 {filtro}: HTTP {c} {d.get('error')}")
            continue
        chequeo(abs(val[pos] - valor) < 1e-3, f"matriz 4x5 {filtro} {k}x{k}: valor en {pos} = {val[pos]:.4f} "
                                              f"(calculo a mano {valor})")
        comparar(f"matriz 4x5 {filtro} {k}x{k}", matriz, filtro, k, val)
    rng = np.random.default_rng(12345)
    aleatoria = rng.integers(0, 256, size=(777, 1001), dtype=np.uint8)
    log.say(f"     Imagen 1001x777 = {aleatoria.size} pixeles: con 256 hilos por bloque el ultimo bloque usa "
            f"{aleatoria.size % 256} de 256 hilos")
    for filtro, k in [("media", 3), ("nitidez", 3), ("laplaciano", 3), ("media", 9)]:
        c, d, _, _, val = solicitar(url, pgm(aleatoria), filtro, k, True)
        if c == 200:
            comparar(f"1001x777 {filtro} {k}x{k} (bordes incl.)", aleatoria, filtro, k, val)
        else:
            chequeo(False, f"1001x777 {filtro}: HTTP {c}")

    # 2) Equivalencia directa - servicio con las imágenes de la práctica
    log.say("\n  2) Equivalencia PyCUDA directo - servicio (img_4000, antes de pasar a 8 bits)")
    img4k_bytes = (Path(a.data) / "img_4000.pgm").read_bytes()
    img4k = pr.decodificar(img4k_bytes)[0]
    salidas = {}
    for filtro, k in [("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3)]:
        c, d, _, png, val = solicitar(url, img4k_bytes, filtro, k, True)
        if c != 200:
            chequeo(False, f"4000x4000 {filtro} {k}x{k}: HTTP {c} {d.get('error')}")
            continue
        comparar(f"4000x4000 {filtro} {k}x{k}", img4k, filtro, k, val)
        if k == 3:
            salidas[filtro] = png
    # PNG del servicio frente a convertir a 8 bits la salida directa: mide el efecto de la codificación
    from PIL import Image
    for filtro, png in salidas.items():
        servicio8 = np.asarray(Image.open(io.BytesIO(png)))
        directo8 = pr.cv.a_8bits(directo_valores(img4k, filtro, 3), filtro)
        dif = int(np.count_nonzero(servicio8 != directo8))
        chequeo(dif == 0, f"PNG del servicio = salida directa en 8 bits ({filtro} 3x3): {dif} pixeles distintos")

    # 3) Resultado visual
    carpeta = RES / "images"
    carpeta.mkdir(parents=True, exist_ok=True)
    x, y, s = 2700, 900, 512
    Image.fromarray(pr.cv.reducir(img4k)).save(carpeta / "entrada_vista.png")
    Image.fromarray(img4k[y:y + s, x:x + s]).save(carpeta / "entrada_recorte.png")
    for filtro, png in salidas.items():
        im = np.asarray(Image.open(io.BytesIO(png)))
        Image.fromarray(pr.cv.reducir(im)).save(carpeta / f"{filtro}3x3_vista.png")
        Image.fromarray(im[y:y + s, x:x + s]).save(carpeta / f"{filtro}3x3_recorte.png")
    log.say(f"\n  3) Resultado visual: original y 3 filtros (vista reducida y recorte) en {carpeta}")

    # 4) Entradas inválidas y recuperación
    log.say("\n  4) Entradas invalidas")
    casos = [("archivo que no es imagen", b"esto no es una imagen" * 10, "media", 3, 400, "texto.png"),
             (f"archivo de {pr.MAX_BYTES // 2**20 + 5} MB (limite {pr.MAX_BYTES // 2**20} MB)",
              b"\x89PNG" + bytes(pr.MAX_BYTES + 5 * 2**20), "media", 3, 413, "grande.png"),
             ("filtro desconocido (sobel)", pgm(matriz), "sobel", 3, 400, "m.pgm"),
             ("media con mascara 5x5", pgm(matriz), "media", 5, 400, "m.pgm"),
             ("nitidez con mascara 9x9", pgm(matriz), "nitidez", 9, 400, "m.pgm")]
    for nombre, datos, filtro, k, esperado_cod, archivo in casos:
        c, d, _, _, _ = solicitar(url, datos, filtro, k, nombre=archivo)
        chequeo(c == esperado_cod, f"{nombre}: HTTP {c} (se esperaba {esperado_cod}) -> \"{d.get('error')}\"")
    c, d, _, _, _ = solicitar(url, pgm(matriz), "media", 3)
    chequeo(c == 200, f"recuperacion: una solicitud valida despues de las invalidas -> HTTP {c}")

    # 5) Aislamiento de solicitudes y límite de la cola
    log.say("\n  5) Aislamiento de solicitudes")
    img2k_bytes = (Path(a.data) / "img_2000.pgm").read_bytes()
    img2k = pr.decodificar(img2k_bytes)[0]
    pedidos = [("A", img2k_bytes, img2k, "media", 9), ("B", img4k_bytes, img4k, "laplaciano", 3)]
    resultados = {}

    def lanzar(etq, datos, filtro, k):
        resultados[etq] = solicitar(url, datos, filtro, k, True)

    hilos = [threading.Thread(target=lanzar, args=(e, d, f, k)) for e, d, _, f, k in pedidos]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    for etq, _, img, filtro, k in pedidos:
        c, d, _, _, val = resultados[etq]
        ok = c == 200 and d["filtro"] == filtro and d["k"] == k and (d["alto"], d["ancho"]) == img.shape
        chequeo(ok, f"solicitud {etq} ({filtro} {k}x{k}, {img.shape[1]}x{img.shape[0]}) recibio su propio "
                    f"resultado: id {d.get('id', '')[:8]}, {d.get('filtro')} {d.get('k')}x{d.get('k')}, "
                    f"{d.get('ancho')}x{d.get('alto')}")
        if c == 200:
            comparar(f"  valores de la solicitud {etq}", img, filtro, k, val)
    ids = {resultados[e][1].get("id") for e in resultados}
    chequeo(len(ids) == 2, "cada solicitud tiene un identificador distinto")

    n = salud.get("cola_max", 4) + 4
    log.say(f"     Limite de la cola: {n} solicitudes simultaneas (1 en la GPU + {salud.get('cola_max')} en cola)")
    codigos = []
    bloqueo = threading.Lock()

    def rafaga():
        c, d, _, _, _ = solicitar(url, img4k_bytes, "media", 21)
        with bloqueo:
            codigos.append((c, d.get("error")))

    hilos = [threading.Thread(target=rafaga) for _ in range(n)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    atendidas = sum(1 for c, _ in codigos if c == 200)
    rechazadas = [e for c, e in codigos if c == 503]
    log.say(f"     Atendidas: {atendidas}, rechazadas con 503: {len(rechazadas)}"
            + (f" -> \"{rechazadas[0]}\"" if rechazadas else ""))
    chequeo(atendidas + len(rechazadas) == n and atendidas >= 1, "las solicitudes se atienden o se rechazan con 503")
    c, _, _, _, _ = solicitar(url, pgm(matriz), "media", 3)
    chequeo(c == 200, f"el servicio sigue respondiendo despues de la rafaga -> HTTP {c}")

    log.say(f"\n  Resultado: {'TODAS LAS PRUEBAS PASARON' if not fallas else f'{len(fallas)} FALLAS'}   "
            f"(guardado en {log.ruta})")
    return 0 if not fallas else 1


# ----------------------------------------------------------------------------------------------
# Experimentos a través del servicio
# ----------------------------------------------------------------------------------------------
def medir(a):
    url = a.url.rstrip("/")
    log = Registro("medicion_servicio.txt")
    codigo, cuerpo = http(f"{url}/api/salud")
    salud = json.loads(cuerpo)
    if codigo != 200:
        sys.exit(f"El servicio no tiene la GPU lista: {salud.get('error')}")
    log.say(f"  Servicio: {url} | GPU: {salud['gpu']} | cola maxima {salud['cola_max']}")
    ini = salud["inicializacion"]
    log.say(f"  Arranque del servicio: contexto {ini['contexto_ms']:.0f} ms, compilar el kernel "
            f"{ini['compilacion_kernel_ms']:.0f} ms (se hace una vez al iniciar el contenedor)")
    log.say(f"  Cliente: {a.entorno}")
    filas = []
    archivos = {}

    def fila(d, cliente_ms, codigo_http, rep, conc=1, ronda="", ronda_ms="", rps="", estado="completada",
             filtro="", k="", lado=""):
        f = dict.fromkeys(directo.CAMPOS, "")
        t = d.get("tiempos", {}) if codigo_http == 200 else {}
        f.update(id_solicitud=d.get("id", ""), modo="servicio", filtro=d.get("filtro", filtro),
                 resolucion=f"{d.get('ancho', lado)}x{d.get('alto', lado)}", mascara=f"{d.get('k', k)}x{d.get('k', k)}",
                 hilos=d.get("hilos_por_bloque", ""), bloques=d.get("bloques", ""), repeticion=rep, concurrencia=conc,
                 ronda=ronda, cliente_ms=f"{cliente_ms:.4f}", ronda_ms=ronda_ms, solicitudes_por_s=rps, unidad="ms",
                 estado=estado if codigo_http == 200 else f"error: {d.get('error')}", codigo_http=codigo_http,
                 entorno=a.entorno, fecha=datetime.datetime.now().isoformat(timespec="seconds"))
        for c, v in t.items():
            if c in f:
                f[c] = f"{v:.4f}"
        return f

    medidas = {}
    for exp, filtro, k, lado in directo.CONFIGURACIONES:
        if lado not in archivos:
            ruta = Path(a.data) / f"img_{lado}.pgm"
            archivos[lado] = ruta.read_bytes()
            log.say(f"  {ruta.name}: sha256 {directo.sha256(ruta)[:16]}...")
        clave = (filtro, k, lado)
        if clave not in medidas:
            c, d, ms, _, _ = solicitar(url, archivos[lado], filtro, k)
            filas.append(fila(d, ms, c, 0, estado="primera_ejecucion", filtro=filtro, k=k, lado=lado))
            t = time.perf_counter()
            while time.perf_counter() - t < a.calentamiento_s:
                solicitar(url, archivos[lado], filtro, k)
            reps = [solicitar(url, archivos[lado], filtro, k) for _ in range(a.reps)]
            medidas[clave] = reps
            ok = [(d, ms) for c, d, ms, _, _ in reps if c == 200]
            if ok:
                log.say(f"  {filtro} {k}x{k} {lado}x{lado}: kernel {statistics.median(d['tiempos']['kernel_ms'] for d, _ in ok):8.3f} ms, "
                        f"servidor {statistics.median(d['tiempos']['servidor_ms'] for d, _ in ok):8.1f} ms, "
                        f"cliente {statistics.median(ms for _, ms in ok):8.1f} ms  ({len(ok)}/{len(reps)} correctas)")
        for rep, (c, d, ms, _, _) in enumerate(medidas[clave], 1):
            f = fila(d, ms, c, rep, filtro=filtro, k=k, lado=lado)
            f["id_solicitud"] = f["id_solicitud"] or f"{exp}-{rep}"
            filas.append(f)

    # Concurrencia: media 9x9 en 4000x4000; 1 y 2 clientes simultaneos, una solicitud por cliente, 5 rondas
    log.say("\n  Solicitudes concurrentes (media 9x9, 4000x4000, una solicitud por cliente, 5 rondas)")
    datos = archivos[4000]
    for conc in (1, 2):
        lat, esperas, rps_rondas, fallos = [], [], [], 0
        for ronda in range(1, a.rondas + 1):
            res = [None] * conc
            barrera = threading.Barrier(conc)

            def cliente(i):
                barrera.wait()
                res[i] = solicitar(url, datos, "media", 9)

            hilos = [threading.Thread(target=cliente, args=(i,)) for i in range(conc)]
            t0 = time.perf_counter()
            [h.start() for h in hilos]
            [h.join() for h in hilos]
            ronda_ms = (time.perf_counter() - t0) * 1000.0
            exitosas = sum(1 for r in res if r[0] == 200)
            rps = exitosas / (ronda_ms / 1000.0)
            rps_rondas.append(rps)
            for i, (c, d, ms, _, _) in enumerate(res, 1):
                filas.append(fila(d, ms, c, i, conc, ronda, f"{ronda_ms:.4f}", f"{rps:.4f}", filtro="media", k=9, lado=4000))
                if c == 200:
                    lat.append(ms)
                    esperas.append(d["tiempos"]["espera_cola_ms"])
                else:
                    fallos += 1
        log.say(f"    {conc} cliente(s): latencia mediana {statistics.median(lat):7.1f} ms (DE {statistics.stdev(lat) if len(lat) > 1 else 0:.1f}), "
                f"espera en cola mediana {statistics.median(esperas):6.1f} ms, "
                f"{statistics.median(rps_rondas):.2f} solicitudes exitosas/s, fallos {fallos}")

    ruta = RES / "tiempos_servicio.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=directo.CAMPOS)
        w.writeheader()
        w.writerows(filas)
    log.say(f"\n  Guardado: {ruta} y {log.ruta}")


def main():
    p = argparse.ArgumentParser(description="Cliente de pruebas del servicio de convolucion")
    sub = p.add_subparsers(dest="cmd", required=True)
    for nombre in ("validar", "medir"):
        s = sub.add_parser(nombre)
        s.add_argument("--url", default="http://localhost:8000")
        s.add_argument("--data", default=str(pr.RAIZ / "data"))
        s.add_argument("--entorno", default="cliente en el mismo equipo (anfitrion), red local")
        s.add_argument("--reps", type=int, default=10)
        s.add_argument("--rondas", type=int, default=5)
        s.add_argument("--calentamiento-s", type=float, default=2.0)
    a = p.parse_args()
    return validar(a) if a.cmd == "validar" else medir(a)


if __name__ == "__main__":
    sys.exit(main())
