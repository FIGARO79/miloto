#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Ejecutor de Pronóstico Completo y Optimización de Inversión (Bankroll Management)
Genera el reporte integral estructurado para agentes y usuarios:
1. Cabecera de entorno y régimen dinámico HMM.
2. Top 5 Jugadas recomendadas por Composite Score v3 con insignias analíticas.
3. Auditoría exhaustiva de la Jugada Estrella (#1).
4. Rueda Combinatoria Reducida (7 números clave, garantía 3 aciertos en 5 tiquetes).
5. Asesoría Financiera: Estrategia de Inversión de 2 Tiquetes ($8.000 COP Total) con 10 balotas sin solape.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Añadir raíz al sys.path para importar miloto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import miloto as ml


def parse_args():
    parser = argparse.ArgumentParser(description="Generar pronóstico completo y optimización de inversión para MiLoto.")
    parser.add_argument("--json", action="store_true", help="Salida estructurada en formato JSON.")
    parser.add_argument("--archivo", default=os.path.join(BASE_DIR, "miloto.json"), help="Ruta a miloto.json.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad estocástica.")
    return parser.parse_args()


def insignias_de_jugada(item, ranking):
    ins = []
    if ranking == 1:
        ins.append("👑 **Top #1**")
    if item["composite"] >= 70.0:
        ins.append("🌟 **Óptimo**")
    if item["score_bayes"] >= 0.52:
        ins.append("🧬 **ADN Fuerte**")
    if item["score_hazard"] >= 0.60:
        ins.append("⏳ **Retorno Maduro**")
    if item["score_ising"] >= 0.55:
        ins.append("🧲 **Ising Coop**")
    if 90 <= item["suma"] <= 110:
        ins.append("🎯 **Gauss Centro**")
    return " ".join(ins) if ins else "⚖️ **Equilibrada**"


def construir_perfil_b(results, top_a_comb):
    """
    Construye la jugada del Perfil B (Cazador de Ruptura por Atraso Extremo):
    Toma balotas con mayor atraso acumulado garantizando CERO solape con el Perfil A
    para maximizar la cobertura a 10 balotas distintas de las 39 del bombo.
    """
    atrasos = results["atrasos_dict"]
    candidatos_b = [n for n in range(1, ml.N_BALLS + 1) if n not in top_a_comb]
    # Ordenar por atraso descendente
    candidatos_b.sort(key=lambda x: atrasos.get(x, 0), reverse=True)

    # Tomar las 5 balotas con mayor atraso sin solape
    balotas_b = sorted(candidatos_b[:5])
    return balotas_b


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    if not os.path.exists(args.archivo):
        print(f"[!] Error: No se encontró el archivo '{args.archivo}'.")
        sys.exit(1)

    with open(args.archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data.get("MiLoto", []))
    for col in ml.COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype(int)
    df["Sorteo"] = pd.to_numeric(df["Sorteo"]).astype(int)
    df = df.sort_values(by="Sorteo").reset_index(drop=True)

    results = ml.analizar_miloto(df)
    recomendadas = ml.generar_combinaciones_probables(5, results)
    top_1 = recomendadas[0]
    audit_top1 = ml.evaluar_jugada_manual(top_1["combinacion"], results)

    # Rueda de 7 números clave (cruce de frecuentes + maduras)
    hot_top = list(results["hot_numbers"].keys())[:4]
    atr_top = list(results["mas_atrasados"].keys())[:3]
    pool_7 = sorted(list(set(hot_top + atr_top)))
    while len(pool_7) < 7:
        for n in range(1, ml.N_BALLS + 1):
            if n not in pool_7:
                pool_7.append(n)
                if len(pool_7) == 7:
                    break
    pool_7 = sorted(pool_7)
    rueda = ml.generar_rueda_combinatoria(pool_7, garantia=3)

    # Estrategia 2 Tiquetes ($8.000 COP)
    comb_perfil_a = top_1["combinacion"]
    comb_perfil_b = construir_perfil_b(results, comb_perfil_a)
    eval_perfil_b = ml.evaluar_jugada_manual(comb_perfil_b, results)
    jp = results.get("jackpot_analysis", {})
    perfil_c = results.get("jackpot_play") or ml.generar_jugada_patron_premio_mayor(results)

    if args.json:
        payload = {
            "entorno": {
                "total_sorteos": results["total_draws"],
                "ultimo_sorteo": results["last_sorteo"],
                "ultima_fecha": results["last_fecha"],
                "ultimo_resultado": results["last_combination"],
                "lag_8": results["lag_8_combination"],
                "regimen": results["regime_info"],
                "ultimo_premio_mayor": jp.get("ultimo_ganador") if jp.get("disponible") else None,
                "proximo_sorteo": jp.get("proximo_sorteo") if jp.get("disponible") else None,
                "delta_desde_premio_mayor": jp.get("delta_actual") if jp.get("disponible") else None,
            },
            "top_5_recomendadas": recomendadas,
            "auditoria_top1": audit_top1,
            "rueda_reducida_7": rueda,
            "estrategia_financiera_2_tiquetes": {
                "costo_por_tiquete_cop": 4000,
                "presupuesto_total_cop": 8000,
                "cobertura_balotas_distintas": 10,
                "porcentaje_bombo_cubierto": round(10 / 39 * 100, 2),
                "tiquete_1_perfil_a": {
                    "tipo": "Perfil A (Momentum & Equilibrio Central)",
                    "combinacion": comb_perfil_a,
                    "score": top_1["composite"],
                    "suma": top_1["suma"],
                    "paridad": top_1["paridad"],
                },
                "tiquete_2_perfil_b": {
                    "tipo": "Perfil B (Cazador de Ruptura por Atraso Extremo)",
                    "combinacion": comb_perfil_b,
                    "score": eval_perfil_b["composite_score"],
                    "suma": eval_perfil_b["suma"],
                    "paridad": eval_perfil_b["paridad"],
                    "atraso_promedio": eval_perfil_b["atraso_promedio"],
                },
            },
            "analisis_ciclo_premio_mayor": jp,
            "tiquete_especial_perfil_c": perfil_c,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
        return

    # Salida estándar formateada para Agente / Consola
    print("==========================================================================")
    print("      🎱 MiLoto - Pronóstico Integral y Optimización de Inversión 🎱      ")
    print("==========================================================================")
    print(f"Dataset Oficial:   {results['total_draws']} sorteos analizados (miloto.json)")
    print(f"Último Sorteo:     #{results['last_sorteo']} ({results['last_fecha']}) -> " + " - ".join(f"{x:02d}" for x in results["last_combination"]))
    if results.get("lag_8_sorteo"):
        print(f"Resonancia Lag-8:  #{results['lag_8_sorteo']} -> " + " - ".join(f"{x:02d}" for x in results["lag_8_combination"]))
    reg = results["regime_info"]
    print(f"Régimen Estocástico: {reg['regimen_actual']} (Media sumas rec.: {reg['media_reciente_sumas']})")
    if jp.get("disponible"):
        ult = jp["ultimo_ganador"]
        print(
            f"Último Premio Mayor: #{ult['sorteo']} ({ult['fecha']}) -> {ult['monto']} | "
            f"Transcurridos hacia #{jp['proximo_sorteo']}: Δ={jp['delta_actual']} sorteo(s)"
        )
    print("--------------------------------------------------------------------------\n")

    # 1. Tabla Top 5
    print("### 1. Pronósticos Recomendados: MiLoto (Top 5)")
    print("* **Criterio Perfil Óptimo:** Composite Score $\\ge 70.0 / 100$\n")
    print("| # | Combinación | Suma | Índice v3 | Par | B/A | Gap Prom | Bayes | Weibull | Ising | Insignias |")
    print("|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for i, j in enumerate(recomendadas, 1):
        c_str = f"**`{'-'.join(f'{x:02d}' for x in j['combinacion'])}`**"
        ins = insignias_de_jugada(j, i)
        print(f"| **{i}** | {c_str} | {j['suma']} | **{j['composite']}** | {j['paridad']} | {j['bajos_altos']} | {j['atraso_prom']} | {j['score_bayes']} | {j['score_hazard']} | {j['score_ising']} | {ins} |")
    print()

    # 2. Auditoría Jugada Estrella
    c1_str = "-".join(f"{x:02d}" for x in top_1["combinacion"])
    print(f"### 2. Auditoría Detallada de la Jugada Estrella #1: `{c1_str}` (Score: {top_1['composite']}/100)")
    for l in audit_top1["veredicto"]:
        print(f"  {l}")
    print()

    # 3. Rueda Reducida
    print("### 3. Estrategia Avanzada: Rueda Combinatoria Reducida (Wheeling System)")
    pool_str = ", ".join(f"{x:02d}" for x in rueda["numeros_seleccionados"])
    print(f"Selección de {rueda['total_numeros']} balotas clave sincronizadas (`{pool_str}`), con **{rueda['garantia']}**:\n")
    for i, jug in enumerate(rueda["jugadas"], 1):
        j_str = " - ".join(f"{x:02d}" for x in jug)
        print(f"* **Tiquete {i:02d}:** `{j_str}`")
    print(f"\n*(Inversión total: {rueda['total_apuestas']} tiquetes x $4.000 = **${rueda['total_apuestas'] * 4000:,} COP**. Cobertura: {rueda['cobertura_pct']}%).*".replace(",", "."))
    print()

    # 4. Asesoría Financiera 2 Tiquetes
    print("### 4. Asesoría Financiera: Estrategia de 2 Tiquetes ($8.000 COP Total)\n")
    print("Para jugadores con disciplina financiera y gestión estricta de capital (Bankroll Management),")
    print("se recomienda jugar exactamente 2 tiquetes ($8.000 COP) con **10 balotas distintas sin solapes**,")
    print("cubriendo más del **25.6% del bombo completo (10 de 39 números)**:\n")

    a_str = "-".join(f"{x:02d}" for x in comb_perfil_a)
    b_str = "-".join(f"{x:02d}" for x in comb_perfil_b)

    print("#### 🟢 Tiquete 1 — Perfil A (🎯 PERFIL PRINCIPAL RECOMENDADO - MÁXIMA PROBABILIDAD ESTOCÁSTICA):")
    print(f"* **Jugada:** `{a_str}` (Score: **{top_1['composite']}**/100, Suma: {top_1['suma']}, {top_1['paridad']})")
    print("  *Enfocado en balotas calientes con alta densidad de frecuencia acumulada y suma centrada en el rango dorado (100).*\n")

    print("#### 🔴 Tiquete 2 — Perfil B (Alternativa Táctica de Alto Riesgo - Cazador de Ruptura):")
    print(f"* **Jugada:** `{b_str}` (Suma: {eval_perfil_b['suma']}, {eval_perfil_b['paridad']}, Atraso Promedio: {eval_perfil_b['atraso_promedio']} sorteos)")
    print("  *Enfocado en números fríos de alto retraso sin repetir ninguna balota del Tiquete 1 (búsqueda de reversión a la media).*\n")

    print("📌 **Presupuesto Sugerido:** 4 sorteos semanales x $8.000 COP = **$32.000 COP por semana**.\n")

    # 5. Análisis de Ciclo del Premio Mayor (5 Balotas) y Relación de Sorteos
    if jp.get("disponible"):
        ult = jp["ultimo_ganador"]
        m4 = jp["proximo_mod4"]
        arr_pool_str = " - ".join(f"{x:02d}" for x in ult["combinacion"])
        print("### 5. Análisis de Ciclo del Premio Mayor (5 Balotas) y Relación de Sorteos\n")
        print(f"* **Diagnóstico del próximo sorteo (`#{jp['proximo_sorteo']}`):**")
        print(f"  * **Salto actual (Δ):** `+{jp['delta_actual']}` sorteo(s) desde `#{ult['sorteo']}` ({jp['tramo_actual_label']}; media histórica `{jp['delta_media']}` / mediana `{jp['delta_mediana']}`).")
        print(f"  * **Módulo 4 (`{jp['proximo_sorteo']} % 4 = {m4['residuo']}`):** Día **{m4['dia']}** (tasa histórica de caída: **`{m4['tasa_pct']}%`** en {m4['ganadores']}/{m4['sorteos']} sorteos).")
        print(f"  * **Efecto Réplica:** {jp['efecto_replica_detalle']}.")
        print(f"  * **Arrastre del último ganador (`#{ult['sorteo']}`):** `{arr_pool_str}` (en el **`{jp['arrastre_pct']}%`** de caídas se repite ≥ 1 balota).\n")

        print("| Sorteo | Salto Δ | Fecha | Combinación | Monto Ganado |")
        print("|:---:|:---:|:---:|:---|:---|")
        for ug in jp["ultimos_ganadores"][-6:]:
            d_txt = f"+{ug['delta_sorteos']}" if ug["delta_sorteos"] is not None else "-"
            c_ug = " - ".join(f"{x:02d}" for x in ug["combinacion"])
            print(f"| **#{ug['sorteo']}** | `{d_txt}` | {ug['fecha']} | `{c_ug}` | {ug['monto']} |")
        print()

        if perfil_c:
            c_str = "-".join(f"{x:02d}" for x in perfil_c["combinacion"])
            arr_inc_str = ", ".join(f"{x:02d}" for x in perfil_c["arrastre_incluidas"])
            print("#### 🏆 Tiquete Especial — Perfil C (Cazador de Patrón de Premio Mayor):")
            print(
                f"* **Jugada:** `{c_str}` (Score v3: **{perfil_c['composite']}**/100 | "
                f"Suma: {perfil_c['suma']} | {perfil_c['paridad']} | "
                f"Balotas ≤ 31: {perfil_c['menores_31']}/5 | Arrastre incluidas: `{arr_inc_str}`)"
            )
            print("  *Optimizada con arrastre del último Premio Mayor, frecuencias del tramo actual y rango de suma donde históricamente caen las 5 balotas.*\n")

    print("==========================================================================")


if __name__ == "__main__":
    main()
