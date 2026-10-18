#!/usr/bin/env python3
"""
MiLoto - Generador / Actualizador de Archivo Excel
Sincroniza los resultados de miloto.json a miloto.xlsx con formato profesional.
"""

import os
import sys
import json
import argparse
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

JSON_FILE = "miloto.json"
EXCEL_FILE = "miloto.xlsx"


def generar_excel(json_path=JSON_FILE, excel_path=EXCEL_FILE):
    if not os.path.exists(json_path):
        print(f"[!] Error: No se encontró el archivo {json_path}. Ejecuta primero 'actualizar_resultados.py'.")
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sorteos = data.get("MiLoto", [])
        if not sorteos:
            print("[!] La lista de sorteos en JSON está vacía.")
            return False

        df = pd.DataFrame(sorteos)
        # Ordenar columnas
        cols_orden = ["Sorteo", "Fecha", "B1", "B2", "B3", "B4", "B5"]
        cols_presentes = [c for c in cols_orden if c in df.columns]
        df = df[cols_presentes]

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="MiLoto", index=False)
            ws = writer.sheets["MiLoto"]

            # Estilos de cabecera
            header_fill = PatternFill(start_color="1C2F73", end_color="1C2F73", fill_type="solid")
            header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            border_thin = Side(style="thin", color="CCCCCC")
            border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)

            for col_idx in range(1, len(cols_presentes) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Estilo de celdas de datos
            for row in ws.iter_rows(min_row=2, max_row=len(df) + 1, min_col=1, max_col=len(cols_presentes)):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border

            # Autoajuste de ancho de columnas
            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    val = str(cell.value or "")
                    if len(val) > max_len:
                        max_len = len(val)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        print(f"[OK] Archivo Excel '{excel_path}' generado con éxito ({len(df)} sorteos).")
        return True

    except Exception as e:
        print(f"[!] Error al generar {excel_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Exportar / sincronizar miloto.json a miloto.xlsx")
    parser.add_argument("--json", default=JSON_FILE, help="Ruta de origen miloto.json")
    parser.add_argument("--excel", default=EXCEL_FILE, help="Ruta de destino miloto.xlsx")
    args = parser.parse_args()

    generar_excel(args.json, args.excel)


if __name__ == "__main__":
    main()
