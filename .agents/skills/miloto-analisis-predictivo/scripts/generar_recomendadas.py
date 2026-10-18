#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Generador de Combinaciones Recomendadas con Filtros
"""

import os
import sys
import argparse
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import miloto as ml


def main():
    parser = argparse.ArgumentParser(description="Generar jugadas recomendadas de MiLoto con filtros personalizados.")
    parser.add_argument("--cantidad", type=int, default=10, help="Número de jugadas a generar (default: 10).")
    parser.add_argument("--min-indice", type=float, default=65.0, help="Composite Score mínimo requerido (default: 65.0).")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON.")
    args = parser.parse_args()

    df = ml.cargar_datos()
    results = ml.analizar_miloto(df)
    recs = ml.generar_combinaciones_probables(args.cantidad * 2, results)

    filtradas = [r for r in recs if r["composite"] >= args.min_indice][:args.cantidad]

    if args.json:
        import json
        print(json.dumps(filtradas, indent=2, ensure_ascii=False))
    else:
        print(f"\n### 🎯 Top {len(filtradas)} Combinaciones Recomendadas (Filtro Score >= {args.min_indice}):")
        filas = []
        for i, item in enumerate(filtradas, 1):
            c_str = " - ".join(f"{x:02d}" for x in item["combinacion"])
            filas.append({
                "#": i,
                "Combinación": c_str,
                "Score v3": f"{item['composite']}/100",
                "Suma": item["suma"],
                "Paridad": item["paridad"],
                "B/A": item["bajos_altos"],
                "Atraso Prom": item["atraso_prom"],
            })
        print(pd.DataFrame(filas).to_markdown(index=False))
        print()


if __name__ == "__main__":
    main()
