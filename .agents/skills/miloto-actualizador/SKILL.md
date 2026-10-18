---
name: miloto-actualizador
description: >-
  Use this skill when updating, auditing, verifying, or repairing the historical lottery dataset (miloto.json) for MiLoto (Baloto Colombia), running the web scraper, validating database integrity, or ingesting new draw results.
---

# Sincronización y Validación de Datos (MiLoto)

Esta skill proporciona las herramientas y procedimientos para mantener actualizada y libre de anomalías la base de datos histórica `miloto.json` y `miloto.xlsx`, tanto de forma automatizada mediante web scraping como manual por contingencia.

Para consultar el cronograma de sorteos, fuentes oficiales y arquitectura de sincronización, revisa:
[Fuentes Oficiales, Horarios y Arquitectura](./references/fuentes_y_horarios.md).

---

## Procedimientos

### 1. Actualización Automática de Sorteos (Web Scraper)
El script `actualizar_resultados.py` consulta el portal oficial de Baloto (páginas 1 a 62), detecta sorteos recientes y los agrega a `miloto.json` evitando duplicados.

**Ejecución estándar interactiva:**
```bash
./actualizar.sh
```
O directamente con Python:
```bash
./.venv/bin/python actualizar_resultados.py
```

**Ejecución desatendida / automatizada:**
```bash
./.venv/bin/python actualizar_resultados.py --auto
```

**Reconstrucción o escaneo del histórico completo (páginas 1 a 62):**
```bash
./.venv/bin/python actualizar_resultados.py --backfill --auto
```

---

### 2. Auditoría de Integridad de la Base de Datos
Antes y después de cualquier actualización o edición, se debe auditar `miloto.json` para verificar que no existan inconsistencias de formato, números duplicados dentro de una jugada, valores fuera de rango o fechas desordenadas.

**Comando de validación:**
```bash
./.venv/bin/python validar_datos.py
```

**Validaciones realizadas:**
* Rango de balotas: $B_1 \dots B_5 \in [1, 39]$.
* Balotas únicas por jugada (sin repetición interna).
* Formato de fecha estricto `YYYY-MM-DD` y orden cronológico.
* Detección de duplicados o saltos en la secuencia de sorteos.

**Modo JSON para integraciones:**
```bash
./.venv/bin/python validar_datos.py --json
```

---

### 3. Sincronización con Hoja de Cálculo Excel
Para generar o refrescar `miloto.xlsx`:
```bash
./.venv/bin/python actualizar_resultados_excel.py
```

---

### 4. Protocolo de Contingencia si Falla el Scraper
Si el portal oficial presenta intermitencias o cambios en su maquetación:
1. Seguir la plantilla de [PROMPT_RESULTADOS.md](../../../PROMPT_RESULTADOS.md) para consultar los números oficiales.
2. Comprobar que los 5 números pertenezcan a 1-39.
3. Ejecutar `validar_datos.py` para asegurar que no se introduzcan inconsistencias.
