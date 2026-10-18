# Instrucciones para Búsqueda y Contingencia de Resultados MiLoto

Para obtener los resultados más recientes o de una fecha específica en caso de contingencia o mantenimiento en los portales web, utiliza el siguiente formato:

## Prompt Sugerido
"Realiza una búsqueda web de los resultados de MiLoto (Colombia) del [FECHA] y presenta los resultados en una tabla que incluya: Número de Sorteo, Números Ganadores (5 balotas del 1 al 39) y Acumulado Próximo Sorteo."

## Formato de Salida Esperado
| Sorteo | Fecha | Combinación Ganadora (1-39) | Acumulado Próximo Sorteo |
| :---: | :---: | :---: | :---: |
| #625 | 2026-10-16 | 07 - 08 - 18 - 38 - 39 | $200 Millones |

---
*Notas Importantes:*
- **Días de Sorteo:** Lunes, Martes, Jueves y Viernes en la noche (~23:00 COT).
- **Mecánica:** 5 números entre 1 y 39 (sin superbalota).
- Tras añadir sorteos manualmente a `miloto.json`, ejecuta siempre `./validar_datos.py` para asegurar que el dataset mantenga integridad absoluta.
