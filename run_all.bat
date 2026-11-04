@echo off
setlocal
call build.bat || exit /b 1
if not exist results mkdir results
if exist scripts\prepare_images.ps1 (
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_images.ps1 || exit /b 1
) else (
    bin\prepare_data.exe --out data data\img3.bmp || (echo Falta data\img3.bmp: guarde data\img3.jpg como BMP de 24 bits & exit /b 1)
)
bin\validate.exe --matriz "data\matriz para probar.csv" || (echo La validacion fallo; revise results\validacion_pruebas.txt & exit /b 1)
bin\benchmark.exe --detalle 2700,900 > results\benchmark_log.txt || exit /b 1
type results\benchmark_log.txt
echo Listo: results\tiempos_crudos.csv y results\validacion.csv
