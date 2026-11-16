@echo off
rem Compila, prepara las imagenes, valida y mide CPU vs GPU. Conecte la laptop antes de medir.
rem Uso:  run_all.bat                     usa la foto de la practica (data\img3.jpg)
rem       run_all.bat ruta\otra_foto.jpg  usa otra foto (jpg, png o bmp)
setlocal
cd /d "%~dp0"
if not exist results mkdir results
set "TMPLOG=%TEMP%\conv_paso.txt"
set "FOTO=%~1"
set "DETALLE="
if "%FOTO%"=="" set "FOTO=data\img3.jpg"
rem Recorte del ojo del pato que se muestra en el informe; con otra foto se elige solo una zona con detalle.
if /i "%FOTO%"=="data\img3.jpg" set "DETALLE=--detalle 2700,900"
if not exist "%FOTO%" (echo  No existe la foto "%FOTO%" & exit /b 1)

echo.
echo  ================================================================
echo   Convolucion de imagenes: CPU (C++) vs GPU (CUDA)
echo   Foto: %FOTO%
echo  ================================================================
echo.

echo  [1/4] Compilando...
call "%~dp0build.bat" > "%TMPLOG%" 2>&1 || (type "%TMPLOG%" & echo  ERROR al compilar & exit /b 1)
echo        OK

echo  [2/4] Preparando las imagenes (2000, 4000 y 6000 pixeles de lado)...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_images.ps1 -Photo "%FOTO%" > "%TMPLOG%" 2>&1 || (type "%TMPLOG%" & echo  ERROR al preparar las imagenes & exit /b 1)
findstr /b /c:"ADVERTENCIA" "%TMPLOG%"
echo        OK

echo  [3/4] Comprobando que la GPU da el mismo resultado que la CPU...
bin\validate.exe || (echo  La validacion fallo: revise results\validacion_pruebas.txt & exit /b 1)

echo.
echo  [4/4] Midiendo tiempos CPU vs GPU (tarda unos minutos)...
bin\benchmark.exe %DETALLE% || exit /b 1
del "%TMPLOG%" 2>nul
