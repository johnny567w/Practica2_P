# Experimentos de la practica del servicio web, en orden:
#   1) PyCUDA directo en el anfitrion (Windows)        -> results\web\tiempos_directo_anfitrion.csv
#   2) PyCUDA directo dentro del contenedor            -> results\web\tiempos_directo_contenedor.csv
#   3) Solicitudes al servicio contenerizado           -> results\web\tiempos_servicio.csv
# El servicio se detiene durante 1) y 2) para que la GPU no tenga otro contexto activo, y se arranca de nuevo
# para 3) (el primer registro de 3) corresponde al arranque en frio del servicio).
# Uso (desde Practica_2):  powershell -ExecutionPolicy Bypass -File web\experimentos.ps1
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$imagen = 'convolucion-web:1.0'
New-Item -ItemType Directory -Force results\web | Out-Null
$raiz = (Get-Location).Path

Write-Output "`n=== Imagen Docker: $imagen ($(docker images $imagen --format '{{.ID}}'))"
docker rm -f convolucion-web 2>$null | Out-Null

Write-Output "`n=== 1) PyCUDA directo en el anfitrion"
py -3.13 web\directo.py --modo anfitrion
if ($LASTEXITCODE) { throw 'fallo directo en el anfitrion' }

Write-Output "`n=== 2) PyCUDA directo dentro del contenedor"
docker run --rm --gpus all -v "${raiz}\data:/app/data" -v "${raiz}\results:/app/results" $imagen python3 directo.py --modo contenedor
if ($LASTEXITCODE) { throw 'fallo directo en el contenedor' }

Write-Output "`n=== 3) Servicio contenerizado"
docker run -d --name convolucion-web --gpus all -p 8000:8000 $imagen | Out-Null
for ($i = 0; $i -lt 60; $i++) {
    try { $s = Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/salud -TimeoutSec 5; if ($s.StatusCode -eq 200) { break } } catch { }
    Start-Sleep -Seconds 2
}
py -3.13 web\cliente.py medir --url http://localhost:8000 --entorno "cliente en el anfitrion (mismo equipo), localhost"
if ($LASTEXITCODE) { throw 'fallo la medicion del servicio' }
$ErrorActionPreference = 'Continue'  # gunicorn escribe su registro en stderr; no es un error
cmd /c "docker logs convolucion-web > results\web\servicio_log.txt 2>&1"
Write-Output "`nListo. Resultados en results\web"
