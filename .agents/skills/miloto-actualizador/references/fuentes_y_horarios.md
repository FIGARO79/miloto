# Fuentes Oficiales, Horarios y Arquitectura de Sincronización de MiLoto

## 📅 Calendario de Sorteos (Colombia)
Los sorteos oficiales de **MiLoto** (operados por Operador Nacional de Juegos - Baloto) se realizan cuatro veces por semana:
- **Lunes en la noche**: aprox. 23:00 COT (UTC-5).
- **Martes en la noche**: aprox. 23:00 COT (UTC-5).
- **Jueves en la noche**: aprox. 23:00 COT (UTC-5).
- **Viernes en la noche**: aprox. 23:00 COT (UTC-5).

### Reglas de los Números
- **Balotas Principales**: 5 números enteros seleccionados entre **1 y 39** (sin repetición en un mismo sorteo).
- **Super Balota**: MiLoto **NO** utiliza súper balota.

---

## 🌐 Fuentes de Datos
1. **Portal Oficial Baloto (Paginación MiLoto - Fuente Primaria)**:
   - URL: `https://www.baloto.com/miloto/resultados/?page=N` (páginas 1 a 62+).
   - Detalle individual: `https://www.baloto.com/miloto/resultados-miloto/{sorteo}/`.
2. **QueCayo (Fuente de Contingencia y Recuperación de Vacíos)**:
   - URL: `http://quecayo.com/miloto`
   - Detalle por sorteo: `http://quecayo.com/miloto/sorteo-{sorteo}`

---

## 🤖 Modos de Ejecución del Scraper
- **Actualización Rápida / Reciente**:
  ```bash
  ./actualizar.sh
  ```
  Consulta las páginas más recientes (sorteos de la semana) y agrega las novedades a `miloto.json` y `miloto.xlsx`.

- **Histórico Completo (Backfill)**:
  ```bash
  ./.venv/bin/python actualizar_resultados.py --backfill --auto
  ```
  Recorre las 62 páginas del portal oficial en paralelo y rellena automáticamente cualquier hueco usando la fuente de contingencia.
