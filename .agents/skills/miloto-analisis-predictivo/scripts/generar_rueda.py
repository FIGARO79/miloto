#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Generador de Ruedas Combinatorias (Wheeling System)
"""

import os
import sys
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import miloto as ml


def main():
    parser = argparse.ArgumentParser(description="Generar rueda combinatoria para MiLoto.")
    parser.add_argument("--numeros", nargs="+", type=int, required=True, help="Grupo de 6 a 15 números para la rueda.")
    parser.add_argument("--garantia", type=int, default=3, help="Garantía de aciertos (default: 3).")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON.")
    args = parser.parse_args()

    rueda = ml.generar_rueda_combinatoria(args.numeros, garantia=args.garantia)

    if args.json:
        import json
        print(json.dumps(rueda, indent=2, ensure_ascii=False))
    else:
        print("\n=======================================================")
        print("        🎡 MiLoto - Rueda Combinatoria Optimizada      ")
        print("=======================================================")
        print(f"Números Seleccionados ({rueda['total_numeros']}): {rueda['numeros_seleccionados']}")
        print(f"Garantía Matemática:       {rueda['garantia']}")
        print(f"Total de Apuestas:         {rueda['total_apuestas']} líneas de 5 números")
        print(f"Costo Total:               ${rueda['total_apuestas'] * 4000:,} COP".replace(",", "."))
        print(f"Cobertura Alcanzada:       {rueda['cobertura_pct']}%")
        print("\nApuestas Generadas:")
        for i, j in enumerate(rueda["jugadas"], 1):
            print(f"  Jugada #{i:02d}: " + " - ".join(f"{x:02d}" for x in j))
        print("=======================================================\n")


if __name__ == "__main__":
    main()
