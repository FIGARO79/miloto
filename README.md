# 🎱 MiLoto - Extractor y Actualizador de Resultados

Herramienta automatizada y suite de sincronización para extraer, auditar y mantener actualizada la base de datos histórica del juego **MiLoto** de Baloto (Operador Nacional de Juegos - Colombia).

---

## 🎯 Sobre el Juego MiLoto
* **Balotas Ganadoras:** 5 números únicos comprendidos entre **1 y 39** (sin repetición en un mismo sorteo).
* **Super Balota:** MiLoto **NO** requiere superbalota.
* **Días de Sorteo:** Lunes, Martes, Jueves y Viernes a las 23:00 COT (UTC-5).
* **Histórico Actual:** Sorteos **#1 a #625** (20 de octubre de 2023 a la actualidad).

---

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias
En Linux / WSL:
```bash
./instalar.sh
```

En Windows:
```cmd
instalar.bat
```

### 2. Actualización de Resultados
Para buscar sorteos nuevos en el portal oficial y agregarlos a la base de datos:

En Linux / WSL:
```bash
./actualizar.sh
```

En Windows:
```cmd
actualizar.bat
```

---

## 🛠️ Herramientas Disponibles

### 🔄 1. Web Scraper (`actualizar_resultados.py`)
Consulta el portal oficial de Baloto (con soporte de paginación de la página 1 a la 62 y fuente de contingencia automática QueCayo para garantizar que no existan sorteos omitidos).

* **Modo Interactivo (reciente):**
  ```bash
  ./.venv/bin/python actualizar_resultados.py
  ```
  Muestra una tabla con los últimos sorteos y solicita confirmación antes de guardar.

* **Modo Automático / Desatendido (`--auto`):**
  ```bash
  ./.venv/bin/python actualizar_resultados.py --auto
  ```

* **Descarga o Reconstrucción Histórica Completa (`--backfill`):**
  ```bash
  ./.venv/bin/python actualizar_resultados.py --backfill --auto
  ```

---

### 📊 2. Sincronizador de Excel (`actualizar_resultados_excel.py`)
Genera o actualiza el archivo [`miloto.xlsx`](./miloto.xlsx) con formato tabular, cabeceras estilizadas y anchos de columna automáticos:
```bash
./.venv/bin/python actualizar_resultados_excel.py
```

---

### 🔍 3. Auditor de Integridad (`validar_datos.py`)
Valida minuciosamente que [`miloto.json`](./miloto.json) cumpla las especificaciones matemáticas y de negocio:
- Todas las balotas $B_1 \dots B_5$ están en el rango $[1, 39]$.
- No hay números repetidos en un sorteo.
- Formato de fecha estricto `YYYY-MM-DD` y orden cronológico.
- Continuidad de sorteos (sin saltos ni duplicados).

```bash
./.venv/bin/python validar_datos.py
```
Salida estructurada JSON:
```bash
./.venv/bin/python validar_datos.py --json
```

---

## 📁 Estructura del Proyecto

```text
miloto/
├── .agents/skills/miloto-actualizador/   # Skill para agentes IA Antigravity
├── actualizar_resultados.py             # Scraper multihilo oficial y contingencia
├── actualizar_resultados_excel.py       # Conversor/actualizador a Excel
├── validar_datos.py                     # Validador y auditor de integridad
├── actualizar.sh / actualizar.bat       # Lanzadores de actualización
├── instalar.sh / instalar.bat           # Instaladores del entorno virtual
├── requirements.txt                     # Dependencias (pandas, bs4, openpyxl, etc.)
├── miloto.json                          # Base de datos histórica (625 sorteos)
├── miloto.xlsx                          # Hoja de cálculo Excel sincronizada
├── PROMPT_RESULTADOS.md                 # Plantilla para contingencia manual
└── README.md                            # Documentación principal
```
