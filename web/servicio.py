"""Servicio web de convolución: interfaz (GET /) y servicio HTTP de procesamiento (POST /api/procesar).

Arquitectura:  navegador -> servicio web Python (Flask) -> PyCUDA -> GPU -> resultado -> navegador

Política de atención: un solo trabajador de GPU (un hilo que crea y usa el contexto CUDA) atiende las
solicitudes en orden de llegada (FIFO). Mientras procesa una, las demás esperan en una cola de tamaño
COLA_MAX; si la cola está llena, la solicitud se rechaza con 503 en lugar de quedar esperando sin límite.

Rutas:
  GET  /                                  interfaz web
  GET  /api/salud                         estado del servicio y de la GPU
  POST /api/procesar                      procesa una imagen (multipart/form-data: imagen, filtro, k)
  GET  /api/resultados/<id>/<archivo>     descarga un resultado (resultado.png o valores.f32)

Arranque: con un servidor de producción, sin depurador ni recarga automática, por ejemplo:
  gunicorn --workers 1 --threads 8 --bind 0.0.0.0:8000 --timeout 600 servicio:app
"""
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import procesamiento as pr

COLA_MAX = int(os.environ.get("COLA_MAX", "4"))                    # solicitudes que pueden esperar
TTL_S = int(os.environ.get("RESULTADOS_TTL_S", "900"))             # los resultados se borran a los 15 min
ESPERA_MAX_S = int(os.environ.get("ESPERA_MAX_S", "600"))          # tiempo máximo esperando a la GPU
DIR_RESULTADOS = Path(os.environ.get("DIR_RESULTADOS", Path(tempfile.gettempdir()) / "convolucion_web"))
ARCHIVOS = {"resultado.png": "image/png", "valores.f32": "application/octet-stream"}
ID_VALIDO = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = pr.MAX_BYTES + 1024 * 1024  # el archivo más los demás campos


class TrabajadorGPU(threading.Thread):
    """Único hilo que toca la GPU. Si no hay GPU o falla CUDA, guarda el diagnóstico y el servicio responde
    503 con ese mensaje: no se cambia en silencio a la CPU."""

    def __init__(self):
        super().__init__(daemon=True, name="trabajador-gpu")
        self.cola = queue.Queue(maxsize=COLA_MAX)
        self.listo = threading.Event()
        self.motor, self.error, self.ocupado = None, None, False

    def run(self):
        try:
            self.motor = pr.MotorGPU()
        except Exception as e:  # sin GPU, sin controlador, sin acceso desde el contenedor, fallo al compilar...
            self.error = f"No se pudo iniciar CUDA: {type(e).__name__}: {e}"
        self.listo.set()
        if self.error:
            return
        while True:
            trabajo = self.cola.get()
            trabajo["inicio"] = time.perf_counter()
            self.ocupado = True
            try:
                trabajo["resultado"] = pr.procesar(self.motor, trabajo["datos"], trabajo["filtro"], trabajo["k"],
                                                   devolver_float=trabajo["float"])
            except pr.ErrorEntrada as e:
                trabajo["error"] = (e.mensaje, e.codigo)
            except Exception as e:  # p. ej. memoria insuficiente en la GPU (pycuda MemoryError) o error de CUDA
                trabajo["error"] = (f"Error al procesar en la GPU: {type(e).__name__}: {e}", 500)
            finally:
                self.ocupado = False
                trabajo["fin"].set()

    def enviar(self, trabajo):
        self.listo.wait(timeout=120)
        if self.error or not self.motor:
            return 503, self.error or "La GPU todavía se está iniciando."
        trabajo["fin"] = threading.Event()
        trabajo["encolado"] = time.perf_counter()
        try:
            self.cola.put_nowait(trabajo)
        except queue.Full:
            return 503, f"El servidor está ocupado: ya hay {COLA_MAX} solicitudes en espera. Intente de nuevo."
        if not trabajo["fin"].wait(timeout=ESPERA_MAX_S):
            return 503, "La solicitud esperó demasiado en la cola."
        return None, None


trabajador = TrabajadorGPU()
trabajador.start()
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


def limpiar_resultados():
    """Borra cada minuto las carpetas de resultados con más de TTL_S segundos."""
    while True:
        limite = time.time() - TTL_S
        for d in DIR_RESULTADOS.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < limite:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
        time.sleep(60)


threading.Thread(target=limpiar_resultados, daemon=True, name="limpieza").start()


def error(mensaje, codigo, id_solicitud=None):
    r = jsonify({"ok": False, "error": mensaje, "id": id_solicitud})
    r.status_code = codigo
    return r


@app.errorhandler(413)
def demasiado_grande(_e):
    return error(f"El archivo supera el límite de {pr.MAX_BYTES // 2**20} MB.", 413)


@app.errorhandler(404)
def no_encontrado(_e):
    return error("Ruta no encontrada.", 404)


@app.errorhandler(405)
def metodo_no_permitido(_e):
    return error("Método no permitido en esta ruta.", 405)


@app.errorhandler(500)
def error_interno(_e):
    return error("Error interno del servidor.", 500)


@app.get("/")
def inicio():
    return render_template("index.html", max_mb=pr.MAX_BYTES // 2**20, max_lado=pr.MAX_LADO,
                           mascaras=pr.MASCARAS_MEDIA)


@app.get("/api/salud")
def salud():
    trabajador.listo.wait(timeout=1)
    datos = {"ok": trabajador.motor is not None, "cola_max": COLA_MAX, "en_cola": trabajador.cola.qsize(),
             "procesando": trabajador.ocupado, "hilos_por_bloque": pr.HILOS_POR_BLOQUE,
             "limites": {"max_mb": pr.MAX_BYTES // 2**20, "max_lado": pr.MAX_LADO,
                         "mascaras_media": list(pr.MASCARAS_MEDIA)}}
    if trabajador.motor:
        datos.update(gpu=trabajador.motor.gpu, inicializacion=trabajador.motor.inicializacion)
    else:
        datos["error"] = trabajador.error or "La GPU todavía se está iniciando."
    return jsonify(datos), (200 if trabajador.motor else 503)


@app.post("/api/procesar")
def procesar():
    t0 = time.perf_counter()  # el cuerpo de la solicitud ya fue recibido por el servidor
    id_solicitud = uuid.uuid4().hex  # identificador único: cada solicitud tiene su propia carpeta de resultados
    archivo = request.files.get("imagen")
    if archivo is None:
        return error("Falta el campo 'imagen' (archivo).", 400, id_solicitud)
    filtro = (request.form.get("filtro") or "").strip().lower()
    k = request.form.get("k", "3")
    try:  # se validan los parámetros antes de ocupar la GPU
        pr.validar_parametros(filtro, k)
    except pr.ErrorEntrada as e:
        return error(e.mensaje, e.codigo, id_solicitud)
    datos = archivo.read()
    trabajo = {"datos": datos, "filtro": filtro, "k": k, "float": request.form.get("devolver_float") == "1"}
    codigo, mensaje = trabajador.enviar(trabajo)
    if codigo:
        return error(mensaje, codigo, id_solicitud)
    if "error" in trabajo:
        return error(*trabajo["error"], id_solicitud)
    r = trabajo["resultado"]
    t_escr = time.perf_counter()
    carpeta = DIR_RESULTADOS / id_solicitud
    carpeta.mkdir(parents=True)
    (carpeta / "resultado.png").write_bytes(r["png"])
    archivos = {"resultado": f"/api/resultados/{id_solicitud}/resultado.png"}
    if r["valores"] is not None:
        r["valores"].astype("float32").tofile(carpeta / "valores.f32")
        archivos["valores_float32"] = f"/api/resultados/{id_solicitud}/valores.f32"
    t_fin = time.perf_counter()
    tiempos = dict(r["tiempos"])
    tiempos["espera_cola_ms"] = (trabajo["inicio"] - trabajo["encolado"]) * 1000.0
    tiempos["escritura_ms"] = (t_fin - t_escr) * 1000.0
    tiempos["servidor_ms"] = (t_fin - t0) * 1000.0
    return jsonify({"ok": True, "id": id_solicitud, "filtro": r["filtro"], "k": r["k"],
                    "ancho": r["ancho"], "alto": r["alto"], "formato_entrada": r["formato"],
                    "hilos_por_bloque": r["hilos"], "bloques": r["bloques"], "archivos": archivos,
                    "unidad": "ms", "tiempos": tiempos})


@app.get("/api/resultados/<id_solicitud>/<nombre>")
def resultado(id_solicitud, nombre):
    if not ID_VALIDO.match(id_solicitud) or nombre not in ARCHIVOS:
        return error("Resultado no válido.", 404)
    carpeta = DIR_RESULTADOS / id_solicitud
    if not (carpeta / nombre).exists():
        return error("El resultado no existe o ya fue borrado.", 404, id_solicitud)
    descargar = request.args.get("descargar") == "1"
    return send_from_directory(carpeta, nombre, mimetype=ARCHIVOS[nombre], as_attachment=descargar,
                               download_name=f"{id_solicitud[:8]}_{nombre}")
