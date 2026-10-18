@echo off
TITLE MiLoto - Actualizar Resultados
SETLOCAL EnableDelayedExpansion

echo =======================================================
echo        🎱 MiLoto - Actualizador de Resultados 🎱       
echo =======================================================
echo.

:: Ir al directorio del script
cd /d "%~dp0"

:: Verificar si el entorno virtual existe
if not exist .venv (
    echo [ERROR] No se detecto el entorno virtual (.venv).
    echo Por favor, ejecuta primero 'instalar.bat'.
    pause
    exit /b 1
)

:: Activar entorno e iniciar actualizacion
echo [SISTEMA] Activando entorno y buscando sorteos nuevos de MiLoto...
call .venv\Scripts\activate
.venv\Scripts\python actualizar_resultados.py %*
.venv\Scripts\python actualizar_resultados_excel.py

echo.
echo [INFO] Proceso de actualizacion finalizado.
echo Puedes cerrar esta ventana o presionar una tecla para salir.
pause
