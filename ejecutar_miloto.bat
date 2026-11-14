@echo off
TITLE MiLoto - Analisis Estadistico y Pronostico
SETLOCAL EnableDelayedExpansion

echo =======================================================
echo        🎱 MiLoto - Analisis y Pronostico 🎱            
echo =======================================================
echo.

cd /d "%~dp0"

if not exist .venv (
    echo [ERROR] No se detecto el entorno virtual (.venv).
    echo Por favor, ejecuta primero 'instalar.bat'.
    pause
    exit /b 1
)

call .venv\Scripts\activate
if "%~1"=="" (
    .venv\Scripts\python miloto.py
    echo.
    .venv\Scripts\python .agents\skills\miloto-analisis-predictivo\scripts\ejecutar_pronostico_completo.py
) else (
    .venv\Scripts\python miloto.py %*
)

echo.
echo [INFO] Analisis finalizado.
pause
