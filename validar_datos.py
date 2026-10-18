#!/usr/bin/env python3
"""
MiLoto - Auditoría y Validación de Integridad de Datos
Valida que miloto.json cumpla con todas las reglas de negocio de MiLoto:
- Balotas en rango [1, 39] sin repetición interna.
- Formato de fecha YYYY-MM-DD.
- Orden cronológico estricto.
- Continuidad y unicidad en la numeración de sorteos.
"""

import os
import sys
import json
import argparse
from datetime import datetime

FILE_PATH = "miloto.json"


def validar_integridad(file_path=FILE_PATH):
    errores = []
    advertencias = []
    estadisticas = {}

    if not os.path.exists(file_path):
        return {
            "valido": False,
            "errores": [f"El archivo {file_path} no existe."],
            "advertencias": [],
            "estadisticas": {},
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "valido": False,
            "errores": [f"Error de sintaxis JSON en {file_path}: {e}"],
            "advertencias": [],
            "estadisticas": {},
        }

    if "MiLoto" not in data or not isinstance(data["MiLoto"], list):
        return {
            "valido": False,
            "errores": ["La clave raíz 'MiLoto' es obligatoria y debe ser una lista."],
            "advertencias": [],
            "estadisticas": {},
        }

    sorteos = data["MiLoto"]
    total = len(sorteos)
    estadisticas["total_sorteos"] = total

    if total == 0:
        return {
            "valido": False,
            "errores": ["La lista de sorteos está vacía."],
            "advertencias": [],
            "estadisticas": estadisticas,
        }

    sorteos_vistos = set()
    fechas_vistas = set()
    fecha_anterior = None
    sorteo_anterior = None

    for i, item in enumerate(sorteos):
        sorteo_num = item.get("Sorteo")
        fecha_str = item.get("Fecha")

        # 1. Validación de número de sorteo
        if sorteo_num is None or not isinstance(sorteo_num, int) or sorteo_num <= 0:
            errores.append(f"Fila {i}: 'Sorteo' inválido o faltante ({sorteo_num}).")
        else:
            if sorteo_num in sorteos_vistos:
                errores.append(f"Fila {i}: Número de sorteo #{sorteo_num} duplicado.")
            sorteos_vistos.add(sorteo_num)

            if sorteo_anterior is not None:
                if sorteo_num <= sorteo_anterior:
                    errores.append(f"Fila {i}: Sorteo #{sorteo_num} no es mayor que el anterior #{sorteo_anterior}.")
                elif sorteo_num > sorteo_anterior + 1:
                    advertencias.append(f"Fila {i}: Salto en numeración entre sorteo #{sorteo_anterior} y #{sorteo_num}.")
            sorteo_anterior = sorteo_num

        # 2. Validación de Fecha
        if not fecha_str or not isinstance(fecha_str, str):
            errores.append(f"Fila {i} (Sorteo #{sorteo_num}): 'Fecha' inválida o ausente.")
        else:
            try:
                fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d").date()
                if fecha_str in fechas_vistas:
                    advertencias.append(f"Fila {i} (Sorteo #{sorteo_num}): Fecha {fecha_str} repetida en múltiples registros.")
                fechas_vistas.add(fecha_str)

                if fecha_anterior and fecha_dt < fecha_anterior:
                    errores.append(f"Fila {i} (Sorteo #{sorteo_num}): Fecha {fecha_str} no sigue orden cronológico (anterior: {fecha_anterior}).")
                fecha_anterior = fecha_dt
            except ValueError:
                errores.append(f"Fila {i} (Sorteo #{sorteo_num}): Formato de fecha '{fecha_str}' no cumple con YYYY-MM-DD.")

        # 3. Validación de balotas B1..B5
        balotas = []
        for col in ["B1", "B2", "B3", "B4", "B5"]:
            b_val = item.get(col)
            if b_val is None or not isinstance(b_val, int):
                errores.append(f"Fila {i} (Sorteo #{sorteo_num}): Balota '{col}' ausente o no entera ({b_val}).")
            elif not (1 <= b_val <= 39):
                errores.append(f"Fila {i} (Sorteo #{sorteo_num}): Balota '{col}' ({b_val}) fuera de rango legal [1, 39].")
            else:
                balotas.append(b_val)

        if len(balotas) == 5:
            if len(set(balotas)) != 5:
                errores.append(f"Fila {i} (Sorteo #{sorteo_num}): Balotas repetidas internamente ({balotas}).")

    estadisticas["primer_sorteo"] = sorteos[0].get("Sorteo")
    estadisticas["ultimo_sorteo"] = sorteos[-1].get("Sorteo")
    estadisticas["fecha_inicio"] = sorteos[0].get("Fecha")
    estadisticas["fecha_fin"] = sorteos[-1].get("Fecha")
    estadisticas["errores_count"] = len(errores)
    estadisticas["advertencias_count"] = len(advertencias)

    return {
        "valido": len(errores) == 0,
        "errores": errores,
        "advertencias": advertencias,
        "estadisticas": estadisticas,
    }


def main():
    parser = argparse.ArgumentParser(description="Validar integridad de la base de datos miloto.json")
    parser.add_argument("--archivo", default=FILE_PATH, help="Ruta al archivo miloto.json a validar.")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON estructurado.")
    args = parser.parse_args()

    resultado = validar_integridad(args.archivo)

    if args.json:
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        sys.exit(0 if resultado["valido"] else 1)

    print("=======================================================")
    print("      🔍 MiLoto - Auditoría de Integridad de Datos     ")
    print("=======================================================")
    print(f"Archivo analizado: {args.archivo}")
    stats = resultado["estadisticas"]
    if stats:
        print(f"Total sorteos:     {stats.get('total_sorteos', 0)}")
        print(f"Rango de sorteos:  #{stats.get('primer_sorteo')} -> #{stats.get('ultimo_sorteo')}")
        print(f"Rango de fechas:   {stats.get('fecha_inicio')} a {stats.get('fecha_fin')}")
    print("-------------------------------------------------------")

    if resultado["advertencias"]:
        print(f"\n[!] Advertencias ({len(resultado['advertencias'])}):")
        for adv in resultado["advertencias"][:10]:
            print(f"  - {adv}")
        if len(resultado["advertencias"]) > 10:
            print(f"  ... y {len(resultado['advertencias']) - 10} más.")

    if resultado["valido"]:
        print("\n[✓] ¡INTEGRIDAD CERTIFICADA! No se encontraron anomalías en los datos.")
        sys.exit(0)
    else:
        print(f"\n[X] SE ENCONTRARON {len(resultado['errores'])} ERRORES DE INTEGRIDAD:")
        for err in resultado["errores"][:15]:
            print(f"  - {err}")
        if len(resultado["errores"]) > 15:
            print(f"  ... y {len(resultado['errores']) - 15} errores adicionales.")
        sys.exit(1)


if __name__ == "__main__":
    main()
