@echo off
TITLE MiLoto - Instalador del Entorno
SETLOCAL EnableDelayedExpansion

echo =======================================================
echo        🎱 MiLoto - Instalador del Entorno 🎱          
echo =======================================================
echo.

:: Ir al directorio del script
cd /d "%~dp0"

:: 1. Verificar Python
echo [1/3] Verificando instalacion de Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python no detectado en el PATH. Instala Python 3.10 o superior.
    pause
    exit /b 1
)
echo [OK] Python detectado.

:: 2. Crear Entorno Virtual
echo.
echo [2/3] Creando entorno virtual (.venv)...
if not exist .venv (
    python -m venv .venv
    echo [OK] Entorno virtual creado.
) else (
    echo [INFO] El entorno virtual ya existe.
)

:: 3. Instalar Dependencias
echo.
echo [3/3] Instalando dependencias desde requirements.txt...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =======================================================
echo    ✓ Instalacion de MiLoto completada exitosamente    
echo =======================================================
echo Para actualizar resultados ejecuta: actualizar.bat
echo.
pause
