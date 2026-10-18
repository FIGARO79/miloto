#!/bin/bash

# MiLoto - Ejecutor del Motor Estadístico y Predictivo
echo "======================================================="
echo "       🎱 MiLoto - Motor Estadístico y Predictivo 🎱    "
echo "======================================================="
echo ""

cd "$(dirname "$0")"

# Determinar el entorno virtual a utilizar
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

if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] No se detectó el entorno virtual ($VENV_DIR)."
    echo "Por favor, ejecuta primero './instalar.sh'."
    exit 1
fi

source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" miloto.py "$@"

if [ -t 0 ] && [[ ! "$*" =~ "--json" ]]; then
    echo ""
    read -p "Presiona [Enter] para salir..."
fi
