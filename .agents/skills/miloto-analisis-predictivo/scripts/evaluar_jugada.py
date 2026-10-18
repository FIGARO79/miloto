#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Evaluador Rápido de Jugada Manual
Evalúa cualquier boleto de 5 balotas (1-39) contra los 7 modelos matemáticos.
"""

import os
import sys
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import miloto as ml


def main():
    parser = argparse.ArgumentParser(description="Evaluar una jugada manual de MiLoto.")
    parser.add_argument("--numeros", nargs=5, type=int, required=True, metavar=("B1", "B2", "B3", "B4", "B5"), help="5 balotas de la jugada.")
    parser.add_argument("--json", action="store_true", help="Salida en JSON estructurado.")
    args = parser.parse_args()

    df = ml.cargar_datos()
    results = ml.analizar_miloto(df)
    ev = ml.evaluar_jugada_manual(args.numeros, results)

    if args.json:
        import json
        print(json.dumps(ev, indent=2, ensure_ascii=False))
    else:
        c_str = " - ".join(f"{x:02d}" for x in ev["combinacion"])
        print("\n=======================================================")
        print("        🔍 MiLoto - Evaluación de Jugada Manual         ")
        print("=======================================================")
        print(f"Jugada: {c_str}")
        print(f"Suma: {ev['suma']} | Paridad: {ev['paridad']} | Bajos/Altos: {ev['bajos_altos']}")
        print(f"Atraso Promedio: {ev['atraso_promedio']} sorteos")
        print("\nScores por Modelo:")
        for mod, sc in ev["scores_detallados"].items():
            print(f"  • {mod}: {sc}")
        print("\nVeredicto:")
        for l in ev["veredicto"]:
            print(f"  {l}")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
