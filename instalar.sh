#!/bin/bash

# MiLoto - Instalador del Entorno
echo "======================================================="
echo "        🎱 MiLoto - Instalador del Entorno 🎱          "
echo "======================================================="
echo ""

# Ir al directorio del script
cd "$(dirname "$0")"

# Determinar nombre del venv
if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    VENV_DIR=".venv_wsl"
else
    VENV_DIR=".venv"
fi

# 1. Verificar Python
echo "[1/3] Verificando instalación de Python..."
if command -v python3 &> /dev/null; then
    echo "[OK] Python 3 detectado."
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    echo "[OK] Python detectado."
    PYTHON_CMD=python
else
    echo "[ERROR] Python no detectado. Instala Python 3.10 o superior."
    exit 1
fi

# 2. Crear Entorno Virtual
echo ""
echo "[2/3] Creando entorno virtual ($VENV_DIR)..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "[OK] Entorno virtual creado."
else
    echo "[INFO] El entorno virtual ya existe."
fi

# 3. Instalar Dependencias
echo ""
echo "[3/3] Instalando dependencias desde requirements.txt..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "    ✓ Instalación de MiLoto completada exitosamente    "
echo "======================================================="
echo "Para actualizar resultados ejecuta: ./actualizar.sh"
echo ""
