# Servicio web de convolución con PyCUDA.
# Imagen base con el CUDA Toolkit 12.5 completo ("devel"): PyCUDA se compila al instalarse y además compila
# el kernel con nvcc al arrancar el servicio. El controlador NVIDIA NO va dentro de la imagen: lo aporta el
# anfitrión y se expone al contenedor con el runtime de GPU (NVIDIA Container Toolkit / Docker Desktop con WSL2).
# Compatibilidad: CUDA 12.5 necesita un controlador NVIDIA 555 o superior en el anfitrión.
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_PATH=/usr/local/cuda

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt web/requirements.txt
RUN pip3 install --upgrade "pip==24.2" && pip3 install -r web/requirements.txt

# Código: el kernel y las funciones de la práctica 02 (python/) y el servicio web (web/)
COPY python/convolucion.py python/conv_pycuda.py python/kernel_convolucion.cu python/
COPY web/ web/

ENV COLA_MAX=4 \
    RESULTADOS_TTL_S=900 \
    DIR_RESULTADOS=/tmp/convolucion_web

EXPOSE 8000
WORKDIR /app/web
# Servidor de producción: 1 proceso (un solo contexto CUDA) con 8 hilos para recibir solicitudes;
# sin depurador ni recarga automática.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:8000", "--timeout", "600", "--access-logfile", "-", "servicio:app"]
