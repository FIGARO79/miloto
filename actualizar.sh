#!/bin/bash

# MiLoto - Actualizador de Resultados
echo "======================================================="
echo "       🎱 MiLoto - Actualizador de Resultados 🎱       "
echo "======================================================="
echo ""

# Ir al directorio del script
cd "$(dirname "$0")"

# Determinar el entorno virtual a utilizar según el sistema y su validez
if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    if [ -f ".venv_wsl/bin/activate" ]; then
        VENV_DIR=".venv_wsl"
    elif [ -f ".venv/bin/activate" ]; then
        VENV_DIR=".venv"
    else
        VENV_DIR=".venv_wsl"
    fi
else
    if [ -f ".venv/bin/activate" ]; then
        VENV_DIR=".venv"
    elif [ -f ".venv_wsl/bin/activate" ]; then
        VENV_DIR=".venv_wsl"
    else
        VENV_DIR=".venv"
    fi
fi

# Verificar si el entorno virtual existe
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] No se detectó el entorno virtual ($VENV_DIR)."
    echo "Por favor, ejecuta primero './instalar.sh'."
    exit 1
fi

# Activar entorno e iniciar actualización
echo "[SISTEMA] Activando entorno y buscando sorteos nuevos de MiLoto..."
source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" actualizar_resultados.py "$@"
"$VENV_DIR/bin/python" actualizar_resultados_excel.py

echo ""
echo "[INFO] Proceso de actualización finalizado."
# Solo pausar si es interactivo y no se paso --auto
if [ -t 0 ] && [[ ! "$*" =~ "--auto" ]] && [ "$AUTO_UPDATE" != "true" ]; then
    read -p "Presiona [Enter] para salir..."
fi
