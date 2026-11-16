@echo off
rem Practica complementaria: Python secuencial y PyCUDA comparados con C/C++ y CUDA C.
rem Requiere haber ejecutado antes run_all.bat (compila y prepara las imagenes de data\).
rem Tarda unos 40 minutos: Python secuencial es muy lento. Para una prueba corta: run_python.bat --rapido
setlocal
cd /d "%~dp0"
if not exist bin\filter.exe (echo  Falta compilar: ejecute primero run_all.bat & exit /b 1)
if not exist data\img_500.pgm (echo  Faltan las imagenes: ejecute primero run_all.bat & exit /b 1)

echo.
echo  ================================================================
echo   Convolucion: C/C++, CUDA C, Python secuencial y PyCUDA
echo  ================================================================
echo.
echo  [1/2] Comprobando que las cuatro versiones dan el mismo resultado...
py -3.13 python\main.py validar || exit /b 1
echo.
echo  [2/2] Midiendo tiempos de las cuatro versiones...
py -3.13 python\main.py medir %*
