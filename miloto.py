#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Motor de Cálculo Estadístico, Probabilístico y Predictivo
Realiza análisis multivariable de los sorteos históricos de MiLoto:
- Frecuencias, Chi-cuadrado y contraste de aleatoriedad.
- Números calientes, fríos y análisis de atrasos (gaps).
- Campana de Gauss para la suma de balotas (media=100, std=23.8).
- Balance de paridad (par/impar) y distribución bajo/alto (1-19 / 20-39).
- Cadenas de Markov y entropía informacional de Shannon.
- Generador de jugadas recomendadas por Composite Score.
- Evaluador de jugadas manuales y ruedas combinatorias (Wheeling System).
"""

import os
import sys
import json
import math
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

FILE_PATH = "miloto.json"
N_BALLS = 39
K_BALLS = 5
COLUMNS = ["B1", "B2", "B3", "B4", "B5"]
TOTAL_COMBINATIONS = math.comb(N_BALLS, K_BALLS)  # 575,757
THEORETICAL_MEAN_SUM = 100.0
THEORETICAL_STD_SUM = 23.81
LOW_SPLIT = 19  # Bajos: 1..19, Altos: 20..39
RECENT_DRAWS_WINDOW = 50


def cargar_datos(file_path=FILE_PATH):
    """Carga y prepara el DataFrame histórico desde miloto.json."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo '{file_path}'. Ejecuta primero 'actualizar_resultados.py'.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sorteos = data.get("MiLoto", [])
    if not sorteos:
        raise ValueError("La lista de sorteos en 'miloto.json' está vacía.")

    df = pd.DataFrame(sorteos)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    df["Sorteo"] = pd.to_numeric(df["Sorteo"], errors="coerce").astype(int)
    df = df.sort_values(by="Sorteo").reset_index(drop=True)
    return df


def analizar_frecuencias(df):
    """Calcula frecuencias globales y realiza prueba Chi-cuadrado."""
    all_balls = df[COLUMNS].values.flatten()
    counts = pd.Series(all_balls).value_counts().reindex(range(1, N_BALLS + 1), fill_value=0)
    total_obs = counts.sum()

    expected_freq = total_obs / N_BALLS
    chi2_stat, p_val = stats.chisquare(counts, f_exp=[expected_freq] * N_BALLS)

    # Frecuencias posicionales B1..B5
    posicionales = {}
    for col in COLUMNS:
        posicionales[col] = df[col].value_counts().head(5).to_dict()

    return {
        "counts": counts.to_dict(),
        "total_obs": int(total_obs),
        "expected_per_ball": round(expected_freq, 2),
        "chi2_stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "interpretacion_chi2": "Aleatorio (distribución uniforme)" if p_val >= 0.05 else "Sesgo estadístico significativo (no uniforme)",
        "posicionales": posicionales,
    }


def analizar_calientes_frios(df, window=RECENT_DRAWS_WINDOW):
    """Identifica números calientes y fríos en la ventana reciente."""
    recent_df = df.tail(window)
    recent_balls = recent_df[COLUMNS].values.flatten()
    recent_counts = pd.Series(recent_balls).value_counts().reindex(range(1, N_BALLS + 1), fill_value=0)

    hot = recent_counts.sort_values(ascending=False).head(10).to_dict()
    cold = recent_counts.sort_values(ascending=True).head(10).to_dict()

    return {
        "ventana_sorteos": window,
        "calientes": hot,
        "frios": cold,
    }


def analizar_atrasos(df):
    """Calcula el atraso actual (sorteos transcurridos sin salir) y atraso máximo histórico."""
    ultimo_sorteo = int(df["Sorteo"].iloc[-1])
    atrasos = {}

    for num in range(1, N_BALLS + 1):
        # Encontrar todas las filas donde salió este número
        apariciones = df[df[COLUMNS].isin([num]).any(axis=1)]
        if not apariciones.empty:
            ultimo_visto = int(apariciones["Sorteo"].iloc[-1])
            gap_actual = ultimo_sorteo - ultimo_visto
            # Calcular gaps históricos
            indices = apariciones["Sorteo"].tolist()
            diffs = np.diff([0] + indices)
            gap_max = int(diffs.max()) if len(diffs) > 0 else gap_actual
            gap_prom = round(float(diffs.mean()), 2) if len(diffs) > 0 else float(gap_actual)
        else:
            gap_actual = len(df)
            gap_max = len(df)
            gap_prom = float(len(df))

        atrasos[num] = {
            "atraso_actual": gap_actual,
            "atraso_max": gap_max,
            "atraso_prom": gap_prom,
        }

    # Top más atrasados
    mas_atrasados = sorted(atrasos.items(), key=lambda x: x[1]["atraso_actual"], reverse=True)[:10]
    return {
        "detalle": atrasos,
        "mas_atrasados": {k: v["atraso_actual"] for k, v in mas_atrasados},
    }


def analizar_sumas_y_paridad(df):
    """Analiza la suma de balotas y balance de pares/impares y bajos/altos."""
    sumas = df[COLUMNS].sum(axis=1)

    paridades = []
    bajos_altos = []
    for _, row in df[COLUMNS].iterrows():
        nums = row.values
        pares = sum(1 for n in nums if n % 2 == 0)
        impares = 5 - pares
        paridades.append(f"{pares}P - {impares}I")

        bajos = sum(1 for n in nums if n <= LOW_SPLIT)
        altos = 5 - bajos
        bajos_altos.append(f"{bajos}B - {altos}A")

    dist_paridad = pd.Series(paridades).value_counts(normalize=True).mul(100).round(2).to_dict()
    dist_bajos_altos = pd.Series(bajos_altos).value_counts(normalize=True).mul(100).round(2).to_dict()

    return {
        "suma_media_empirica": round(float(sumas.mean()), 2),
        "suma_std_empirica": round(float(sumas.std()), 2),
        "suma_media_teorica": THEORETICAL_MEAN_SUM,
        "suma_std_teorica": THEORETICAL_STD_SUM,
        "suma_min": int(sumas.min()),
        "suma_max": int(sumas.max()),
        "distribucion_paridad_pct": dist_paridad,
        "distribucion_bajos_altos_pct": dist_bajos_altos,
    }


def construir_matriz_markov(df):
    """Construye matriz de transición de Markov de orden 1 entre números ordenados."""
    matriz = np.zeros((N_BALLS + 1, N_BALLS + 1), dtype=float)
    for _, row in df[COLUMNS].iterrows():
        nums = sorted(row.values)
        for i in range(len(nums) - 1):
            matriz[nums[i], nums[i + 1]] += 1.0

    # Normalizar por fila
    row_sums = matriz.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matriz_prob = matriz / row_sums
    return matriz_prob


def calcular_entropia_shannon(combinacion):
    """Calcula la entropía de espaciado de Shannon normalizada (0 a 1)."""
    nums = sorted(combinacion)
    diffs = np.diff([0] + nums + [N_BALLS + 1])
    probs = diffs / (N_BALLS + 1)
    probs = probs[probs > 0]
    entropia = -np.sum(probs * np.log2(probs))
    max_entropia = np.log2(len(diffs))
    return round(float(entropia / max_entropia), 4)


def calcular_composite_score(combinacion, stats_dict, markov_matrix):
    """
    Calcula una puntuación compuesta multivariable (0 a 100 puntos) para una jugada de MiLoto:
    - 25% Frecuencia ponderada (global + reciente)
    - 20% Ajuste a campana de Gauss de la suma
    - 15% Balance Par / Impar (óptimo 2P/3I o 3P/2I)
    - 15% Balance Bajo / Alto (óptimo 2B/3A o 3B/2A)
    - 15% Nivel de atraso (hazard / gap score)
    - 10% Cadenas de Markov y Entropía
    """
    nums = sorted(combinacion)
    counts = stats_dict["frecuencias"]["counts"]
    calientes = stats_dict["calientes_frios"]["calientes"]
    atrasos = stats_dict["atrasos"]["detalle"]

    # 1. Puntaje de frecuencia (0 a 1)
    max_c = max(counts.values()) or 1
    score_freq_global = sum(counts.get(n, 0) / max_c for n in nums) / 5.0
    score_caliente = sum(1.0 for n in nums if n in calientes) / 5.0
    score_freq = 0.6 * score_freq_global + 0.4 * score_caliente

    # 2. Puntaje Gaussiano de suma (0 a 1)
    s = sum(nums)
    z = abs(s - THEORETICAL_MEAN_SUM) / THEORETICAL_STD_SUM
    score_suma = max(0.0, 1.0 - (z / 3.0))

    # 3. Balance Par / Impar
    pares = sum(1 for n in nums if n % 2 == 0)
    score_par = 1.0 if pares in (2, 3) else (0.5 if pares in (1, 4) else 0.1)

    # 4. Balance Bajo / Alto
    bajos = sum(1 for n in nums if n <= LOW_SPLIT)
    score_ba = 1.0 if bajos in (2, 3) else (0.5 if bajos in (1, 4) else 0.1)

    # 5. Atrasos / Hazard Score
    gaps = [atrasos[n]["atraso_actual"] for n in nums]
    avg_gap = np.mean(gaps)
    # Recompensar un atraso promedio moderado (4 a 12 sorteos)
    if 4 <= avg_gap <= 15:
        score_gap = 1.0
    elif avg_gap < 4:
        score_gap = 0.7
    else:
        score_gap = max(0.2, 1.0 - (avg_gap - 15) / 20.0)

    # 6. Markov & Entropía
    markov_p = 1.0
    for i in range(len(nums) - 1):
        p = markov_matrix[nums[i], nums[i + 1]]
        markov_p *= (p if p > 0 else 0.001)
    score_markov = min(1.0, (markov_p ** (1.0 / 4.0)) * 5.0)
    score_entropia = calcular_entropia_shannon(nums)
    score_disp = 0.5 * score_markov + 0.5 * score_entropia

    # Puntuación final ponderada (0 a 100)
    score_total = (
        0.25 * score_freq +
        0.20 * score_suma +
        0.15 * score_par +
        0.15 * score_ba +
        0.15 * score_gap +
        0.10 * score_disp
    ) * 100.0

    return {
        "score_total": round(score_total, 2),
        "suma": s,
        "paridad": f"{pares}P - {5 - pares}I",
        "bajos_altos": f"{bajos}B - {5 - bajos}A",
        "atraso_promedio": round(float(avg_gap), 1),
        "entropia": score_entropia,
    }


def evaluar_jugada(combinacion, stats_dict, markov_matrix):
    """Evalúa una jugada manual de 5 balotas."""
    if len(combinacion) != 5 or len(set(combinacion)) != 5:
        raise ValueError("La jugada debe tener exactamente 5 números únicos.")
    for n in combinacion:
        if not (1 <= n <= N_BALLS):
            raise ValueError(f"El número {n} está fuera del rango legal de MiLoto [1, 39].")

    metrics = calcular_composite_score(combinacion, stats_dict, markov_matrix)
    sc = metrics["score_total"]

    if sc >= 85:
        veredicto = "🌟 EXCELENTE: Máxima alineación probabilística y balance estructural."
    elif sc >= 72:
        veredicto = "🟢 FUERTE: Muy buen balance de frecuencia, suma y paridad."
    elif sc >= 55:
        veredicto = "🟡 MODERADA: Jugada estándar, balance aceptable."
    else:
        veredicto = "🔴 ATÍPICA: Configuración con baja probabilidad histórica (suma extrema o desbalance de paridad)."

    return {
        "combinacion": sorted(combinacion),
        "metricas": metrics,
        "veredicto": veredicto,
    }


def generar_combinaciones_recomendadas(n=10, stats_dict=None, markov_matrix=None, df=None):
    """Genera las N combinaciones con mayor Composite Score usando muestreo heurístico."""
    counts = stats_dict["frecuencias"]["counts"]
    calientes = stats_dict["calientes_frios"]["calientes"]
    atrasos = stats_dict["atrasos"]["detalle"]

    # Pesos de selección para números
    pesos = np.zeros(N_BALLS + 1)
    for num in range(1, N_BALLS + 1):
        f_norm = counts.get(num, 1) / max(counts.values())
        hot_boost = 1.3 if num in calientes else 1.0
        gap = atrasos[num]["atraso_actual"]
        gap_boost = 1.2 if 3 <= gap <= 12 else 0.9
        pesos[num] = f_norm * hot_boost * gap_boost

    pesos_norm = pesos[1:] / pesos[1:].sum()

    # Generar candidatos
    candidatos = set()
    intentos = 0
    max_intentos = 15000

    while len(candidatos) < 2500 and intentos < max_intentos:
        intentos += 1
        comb = tuple(sorted(np.random.choice(range(1, N_BALLS + 1), size=5, replace=False, p=pesos_norm)))
        # Filtro preliminar de suma y paridad para acelerar
        s = sum(comb)
        if 65 <= s <= 135:
            pares = sum(1 for x in comb if x % 2 == 0)
            if 1 <= pares <= 4:
                candidatos.add(comb)

    # Evaluar y ordenar candidatos
    evaluados = []
    for c in candidatos:
        ev = calcular_composite_score(c, stats_dict, markov_matrix)
        evaluados.append({
            "combinacion": list(c),
            "score": ev["score_total"],
            "suma": ev["suma"],
            "paridad": ev["paridad"],
            "bajos_altos": ev["bajos_altos"],
            "atraso_prom": ev["atraso_promedio"],
        })

    evaluados.sort(key=lambda x: x["score"], reverse=True)

    # Seleccionar las mejores asegurando diversidad (no idénticas con 4 números compartidos)
    seleccionadas = []
    for item in evaluados:
        c = set(item["combinacion"])
        demasiado_similar = False
        for s in seleccionadas:
            if len(c.intersection(set(s["combinacion"]))) >= 4:
                demasiado_similar = True
                break
        if not demasiado_similar:
            seleccionadas.append(item)
            if len(seleccionadas) == n:
                break

    return seleccionadas


def generar_rueda_combinatoria(numeros_elegidos, garantia=3):
    """
    Genera un sistema de rueda combinatoria (Covering Wheel) para MiLoto.
    Garantiza al menos 'garantia' aciertos si las 5 balotas ganadoras caen dentro de numeros_elegidos.
    """
    numeros = sorted(list(set(numeros_elegidos)))
    v = len(numeros)
    if v < 6:
        raise ValueError(f"Se requieren al menos 6 números para una rueda combinatoria (proporcionados: {v}).")

    todas_las_tuplas_garantia = list(combinations(numeros, garantia))
    cubiertas = set()
    lineas = []

    # Búsqueda greedy de combinaciones de tamaño 5 que maximicen cobertura
    todas_posibles_5 = list(combinations(numeros, 5))
    np.random.shuffle(todas_posibles_5)

    while len(cubiertas) < len(todas_las_tuplas_garantia) and todas_posibles_5:
        # Encontrar la combinación que más tuplas no cubiertas aporte
        mejor_comb = None
        max_nuevas = -1
        for c in todas_posibles_5[:300]:
            sub_tuplas = set(combinations(c, garantia))
            nuevas = len(sub_tuplas - cubiertas)
            if nuevas > max_nuevas:
                max_nuevas = nuevas
                mejor_comb = c
            if max_nuevas == len(sub_tuplas):
                break

        if mejor_comb is None or max_nuevas == 0:
            break

        lineas.append(sorted(list(mejor_comb)))
        cubiertas.update(combinations(mejor_comb, garantia))
        todas_posibles_5.remove(mejor_comb)

    return {
        "numeros_seleccionados": numeros,
        "total_numeros": v,
        "garantia": f"{garantia} aciertos si caen dentro de los {v} números",
        "total_apuestas": len(lineas),
        "cobertura_pct": round(len(cubiertas) / len(todas_las_tuplas_garantia) * 100, 2),
        "jugadas": lineas,
    }


def ejecutar_analisis_completo():
    """Ejecuta el análisis general y genera el reporte completo."""
    df = cargar_datos()
    frecuencias = analizar_frecuencias(df)
    calientes_frios = analizar_calientes_frios(df)
    atrasos = analizar_atrasos(df)
    sumas = analizar_sumas_y_paridad(df)
    markov = construir_matriz_markov(df)

    stats_dict = {
        "frecuencias": frecuencias,
        "calientes_frios": calientes_frios,
        "atrasos": atrasos,
        "sumas": sumas,
    }

    recomendadas = generar_combinaciones_recomendadas(n=10, stats_dict=stats_dict, markov_matrix=markov, df=df)

    return {
        "resumen": {
            "total_sorteos": len(df),
            "primer_sorteo": int(df["Sorteo"].iloc[0]),
            "ultimo_sorteo": int(df["Sorteo"].iloc[-1]),
            "ultima_fecha": df["Fecha"].iloc[-1],
            "ultimo_resultado": sorted(df.iloc[-1][COLUMNS].tolist()),
            "universo_combinatorio": TOTAL_COMBINATIONS,
        },
        "estadisticas": stats_dict,
        "recomendadas": recomendadas,
    }


def imprimir_reporte_consola(res):
    """Imprime el reporte estadístico en terminal con formato legible."""
    r = res["resumen"]
    st = res["estadisticas"]
    recs = res["recomendadas"]

    print("==========================================================================")
    print("        🎱 MiLoto - Análisis Estadístico y Pronóstico Predictivo 🎱       ")
    print("==========================================================================")
    print(f"Base de Datos: {r['total_sorteos']} sorteos analizados (Sorteo #{r['primer_sorteo']} a #{r['ultimo_sorteo']})")
    print(f"Último Sorteo: #{r['ultimo_sorteo']} ({r['ultima_fecha']}) -> {r['ultimo_resultado']}")
    print(f"Espacio Muestral Teórico: C(39, 5) = {r['universo_combinatorio']:,} combinaciones")
    print("--------------------------------------------------------------------------")

    # Chi-cuadrado
    chi = st["frecuencias"]
    print("\n🔍 1. CONTRASTE DE UNIFORMIDAD ESTADÍSTICA (Chi-Cuadrado):")
    print(f"   • Chi2 Stat: {chi['chi2_stat']} | p-value: {chi['p_value']}")
    print(f"   • Veredicto: {chi['interpretacion_chi2']}")

    # Hot & Cold
    cf = st["calientes_frios"]
    print(f"\n🔥 2. NÚMEROS CALIENTES (Últimos {cf['ventana_sorteos']} sorteos):")
    hot_str = ", ".join([f"#{num} ({cnt} veces)" for num, cnt in list(cf['calientes'].items())[:7]])
    print(f"   {hot_str}")

    print(f"\n🧊 3. NÚMEROS FRÍOS / MENOS FRECUENTES (Últimos {cf['ventana_sorteos']} sorteos):")
    cold_str = ", ".join([f"#{num} ({cnt} veces)" for num, cnt in list(cf['frios'].items())[:7]])
    print(f"   {cold_str}")

    # Atrasos
    atr = st["atrasos"]
    print("\n⏳ 4. NÚMEROS CON MAYOR ATRASO (Sorteos sin aparecer):")
    atr_str = ", ".join([f"#{num} ({gap} sorteos)" for num, gap in list(atr['mas_atrasados'].items())[:7]])
    print(f"   {atr_str}")

    # Sumas y Paridad
    sm = st["sumas"]
    print("\n⚖️ 5. DISTRIBUCIÓN ESTRUCTURAL (Sumas y Paridad):")
    print(f"   • Suma Promedio Empírica: {sm['suma_media_empirica']} (Teórica: {sm['suma_media_teorica']} ± {sm['suma_std_teorica']})")
    print(f"   • Rango Observado de Sumas: {sm['suma_min']} a {sm['suma_max']}")
    top_par = list(sm["distribucion_paridad_pct"].items())[:3]
    top_ba = list(sm["distribucion_bajos_altos_pct"].items())[:3]
    print(f"   • Paridad dominante: {', '.join([f'{k}: {v}%' for k, v in top_par])}")
    print(f"   • Bajos/Altos dominante: {', '.join([f'{k}: {v}%' for k, v in top_ba])}")

    # Recomendadas
    print("\n🎯 6. TOP 10 COMBINACIONES RECOMENDADAS PARA EL PRÓXIMO SORTEO:")
    filas = []
    for idx, item in enumerate(recs, 1):
        c_str = " - ".join([f"{x:02d}" for x in item["combinacion"]])
        filas.append({
            "#": idx,
            "Combinación Sugerida": c_str,
            "Composite Score": f"{item['score']}/100",
            "Suma": item["suma"],
            "Paridad": item["paridad"],
            "B/A": item["bajos_altos"],
            "Atraso Prom": item["atraso_prom"],
        })
    df_recs = pd.DataFrame(filas)
    print(df_recs.to_markdown(index=False))
    print("==========================================================================")


def main():
    parser = argparse.ArgumentParser(description="Motor estadístico y generador predictivo para MiLoto.")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON para APIs o dashboards.")
    parser.add_argument("--evaluar", nargs=5, type=int, metavar=("B1", "B2", "B3", "B4", "B5"), help="Evaluar una jugada manual de 5 números (ej. --evaluar 7 8 18 38 39).")
    parser.add_argument("--rueda", nargs="+", type=int, help="Generar rueda combinatoria para los números dados (ej. --rueda 1 4 7 11 18 25 32 39).")
    parser.add_argument("--garantia", type=int, default=3, help="Garantía mínima de aciertos para la rueda combinatoria (default: 3).")
    args = parser.parse_args()

    df = cargar_datos()
    frecuencias = analizar_frecuencias(df)
    calientes_frios = analizar_calientes_frios(df)
    atrasos = analizar_atrasos(df)
    sumas = analizar_sumas_y_paridad(df)
    markov = construir_matriz_markov(df)
    stats_dict = {
        "frecuencias": frecuencias,
        "calientes_frios": calientes_frios,
        "atrasos": atrasos,
        "sumas": sumas,
    }

    if args.evaluar:
        evaluacion = evaluar_jugada(args.evaluar, stats_dict, markov)
        if args.json:
            print(json.dumps(evaluacion, indent=2, ensure_ascii=False))
        else:
            print("\n=======================================================")
            print("        🔍 MiLoto - Evaluación de Jugada Manual         ")
            print("=======================================================")
            c_str = " - ".join([f"{x:02d}" for x in evaluacion["combinacion"]])
            m = evaluacion["metricas"]
            print(f"Jugada:              {c_str}")
            print(f"Puntuación Global:   {m['score_total']} / 100")
            print(f"Veredicto:           {evaluacion['veredicto']}")
            print(f"Suma Total:          {m['suma']} (Óptimo: 75 a 125)")
            print(f"Balance Paridad:     {m['paridad']}")
            print(f"Balance Bajo/Alto:   {m['bajos_altos']}")
            print(f"Atraso Promedio:     {m['atraso_promedio']} sorteos")
            print(f"Entropía Shannon:    {m['entropia']}")
            print("=======================================================\n")
        return

    if args.rueda:
        rueda = generar_rueda_combinatoria(args.rueda, garantia=args.garantia)
        if args.json:
            print(json.dumps(rueda, indent=2, ensure_ascii=False))
        else:
            print("\n=======================================================")
            print("        🎡 MiLoto - Rueda Combinatoria Optimizada      ")
            print("=======================================================")
            print(f"Números Seleccionados ({rueda['total_numeros']}): {rueda['numeros_seleccionados']}")
            print(f"Garantía Matemática:       {rueda['garantia']}")
            print(f"Total de Apuestas:         {rueda['total_apuestas']} líneas de 5 números")
            print(f"Cobertura Alcanzada:       {rueda['cobertura_pct']}%")
            print("\nApuestas Generadas:")
            for i, j in enumerate(rueda["jugadas"], 1):
                print(f"  Jugada #{i:02d}: " + " - ".join([f"{x:02d}" for x in j]))
            print("=======================================================\n")
        return

    resultado = ejecutar_analisis_completo()
    if args.json:
        # Serializar numpy types para JSON
        print(json.dumps(resultado, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
    else:
        imprimir_reporte_consola(resultado)


if __name__ == "__main__":
    main()
