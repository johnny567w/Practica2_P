@echo off
rem Compila todos los ejecutables en bin\ con nvcc + MSVC (Visual Studio 2022, x64).
rem Arquitectura CUDA por defecto: sm_75 (GTX 1650). Para otra GPU:  set CUDA_ARCH=sm_86 ^& build.bat
setlocal
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_75

where cl >nul 2>nul
if errorlevel 1 (
    set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
    for /f "usebackq tokens=*" %%i in (`vswhere -latest -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
)
if defined VSDIR call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul
if errorlevel 1 (
    echo No se encontro el compilador de MSVC ^(cl.exe^). Instale Visual Studio 2022 con "Desarrollo para escritorio con C++".
    exit /b 1
)

if not exist bin mkdir bin
if not exist results\images mkdir results\images
if not exist data mkdir data

rem Host: /O2 (MSVC, /fp:precise por defecto). Dispositivo: -O3 por defecto de nvcc (FMA habilitado).
set NVFLAGS=-O3 -arch=%CUDA_ARCH% -std=c++17 -Xcompiler "/O2 /EHsc /W3 /utf-8" -Isrc
set COMMON=src\conv_cpu.cpp src\conv_gpu.cu src\filters.cpp src\image_io.cpp src\validation.cpp

echo [1/4] benchmark.exe
nvcc %NVFLAGS% -o bin\benchmark.exe src\benchmark.cpp %COMMON% || exit /b 1
echo [2/4] validate.exe
nvcc %NVFLAGS% -o bin\validate.exe src\validate.cpp %COMMON% || exit /b 1
echo [3/4] filter.exe
nvcc %NVFLAGS% -o bin\filter.exe src\filter.cpp %COMMON% || exit /b 1
echo [4/4] prepare_data.exe (solo C++)
cl /nologo /O2 /EHsc /W3 /utf-8 /std:c++17 /Fe:bin\prepare_data.exe /Fo:bin\ src\prepare_data.cpp src\image_io.cpp || exit /b 1
del /q bin\*.obj bin\*.exp bin\*.lib 2>nul
echo Compilacion completa.
