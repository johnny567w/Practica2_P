@echo off
rem Construye la imagen Docker, comprueba el acceso a la GPU e inicia el servicio web en el puerto 8000.
rem Requiere Docker Desktop (WSL2) con soporte de GPU NVIDIA y haber ejecutado antes run_all.bat (imagenes de data\).
setlocal
cd /d "%~dp0"
set "IMAGEN=convolucion-web:1.0"

echo  [1/3] Construyendo la imagen %IMAGEN%...
docker build -t %IMAGEN% . || exit /b 1

echo  [2/3] Comprobando el acceso a la GPU desde el contenedor...
docker run --rm --gpus all %IMAGEN% nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || (echo  El contenedor no ve la GPU & exit /b 1)

echo  [3/3] Iniciando el servicio en http://localhost:8000  (Ctrl+C para detenerlo)
docker rm -f convolucion-web >nul 2>&1
docker run --rm --name convolucion-web --gpus all -p 8000:8000 %IMAGEN%
