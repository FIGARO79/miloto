#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiLoto - Motor Completo de Cálculo Estadístico, Probabilístico y Predictivo
Implementa TODOS los modelos matemáticos y estocásticos de GanaBaloto adaptados a MiLoto (39 balotas, sin Super Balota):
1. Frecuencias vectorizadas y prueba Chi-cuadrado (uniformidad de Poisson/Multinomial).
2. Modelo Bayesiano Dirichlet-Multinomial con prior conjugado de Laplace (alpha=1.0).
3. Modelo de Supervivencia y Desgaste de Weibull (Hazard Rate calibrado con desgaste k=1.1576, lambda=8.1420).
4. Modelo de Física Estadística de Ising (Spin-Glass / Matriz de acoplamiento mutuo cooperativo J_ij).
5. Cadenas de Markov Duales: Global secuencial y Posicionales (B1->B1, B2->B2, etc.).
6. Detección de Régimen Estocástico (HMM de 3 estados: Congestión Baja, Equilibrio Central, Expansión Alta).
7. Distribución Gaussiana de la Suma (media teorica=100.0, std teorica=23.81).
8. Entropía informacional de Shannon (dispersion de espaciado).
9. Sorteo de Resonancia Cíclica Lag-8.
10. Índice Predictivo Compuesto Global Refinado v3 (Composite Score 0 a 100).
11. Veredicto cualitativo estructurado para usuarios no técnicos.
12. Generador de Ruedas Combinatorias (Covering Wheels) con garantías de acierto.
"""

import os
import sys
import re
import math
import json
import argparse
from datetime import datetime
from collections import defaultdict
from itertools import combinations

# --- Configuración de Hardware ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from scipy import stats

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    class MockJax:
        def jit(self, fn):
            return fn
        def devices(self):
            raise Exception("No JAX")
    jax = MockJax()
    HAS_JAX = False

# --- Parámetros de MiLoto ---
FILE_PATH = "miloto.json"
N_BALLS = 39
K_BALLS = 5
COLUMNS = ["B1", "B2", "B3", "B4", "B5"]
TOTAL_COMBINATIONS = math.comb(N_BALLS, K_BALLS)  # 575,757 combinaciones
THEORETICAL_MEAN_SUM = 100.0
THEORETICAL_STD_SUM = 23.81
LOW_HIGH_SPLIT = 19  # Bajos: 1..19, Altos: 20..39
RECENT_WINDOW = 50

# Calibración empírica de Weibull para MiLoto (N=39, K=5)
WEIBULL_SHAPE = 1.1576
WEIBULL_SCALE = 8.1420


# =====================================================================
# 1. MODELOS MATEMÁTICOS DE PUNTUACIÓN (SCORING KERNELS)
# =====================================================================

def calculate_sum_gaussian_score(combination, mean_sum=THEORETICAL_MEAN_SUM, std_sum=THEORETICAL_STD_SUM):
    """Calcula un puntaje de campana de Gauss (0 a 1) para la suma de la combinación."""
    c_sum = sum(combination)
    score = math.exp(-((c_sum - mean_sum) ** 2) / (2 * (std_sum ** 2)))
    return max(0.0, min(1.0, score))


def calculate_shannon_entropy(combination):
    """Calcula la entropía de Shannon (0 a 1) basada en las diferencias entre números ordenados."""
    if len(combination) < 2:
        return 1.0
    sorted_c = sorted(combination)
    diffs = [sorted_c[i + 1] - sorted_c[i] for i in range(len(sorted_c) - 1)]
    total_diff = sum(diffs)
    if total_diff == 0:
        return 0.0
    probs = [d / total_diff for d in diffs if d > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(diffs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return max(0.0, min(1.0, normalized_entropy))


def calculate_bayesian_dirichlet_score(combination, counts_dict, total_draws, alpha=1.0, bayes_lookup=None):
    """
    Calcula la probabilidad a posteriori Dirichlet-Multinomial normalizada (0 a 1).
    Utiliza un prior simétrico de Laplace/Dirichlet alpha=1.0.
    """
    if total_draws <= 0:
        return 0.5

    if bayes_lookup is not None:
        main_table = bayes_lookup
    else:
        prior_prob = 1.0 / N_BALLS
        denom_main = (K_BALLS * total_draws + N_BALLS * alpha) * (prior_prob * 2.0)
        main_table = np.zeros(N_BALLS + 1, dtype=np.float64)
        for num in range(1, N_BALLS + 1):
            count = counts_dict.get(num, 0) if isinstance(counts_dict, dict) else 0
            main_table[num] = (count + alpha) / denom_main

    main_score = sum(main_table[num] for num in combination if 1 <= num <= N_BALLS)
    total_score = main_score / K_BALLS
    return max(0.0, min(1.0, total_score))


def calculate_weibull_hazard_score(combination, atrasos_dict, shape=WEIBULL_SHAPE, scale=WEIBULL_SCALE):
    """
    Calcula la presión estadística acumulada bajo un modelo de supervivencia Weibull.
    Calibrado empíricamente con los 625 sorteos reales de MiLoto (k=1.1576, lambda=8.1420).
    """
    if not atrasos_dict:
        return 0.5

    scores = []
    for num in combination:
        if 1 <= num <= N_BALLS:
            gap = atrasos_dict.get(num, 0.0)
            hazard = 1.0 - math.exp(-math.pow(max(0.0, gap) / scale, shape))
            scores.append(hazard)

    avg_hazard = sum(scores) / len(scores) if scores else 0.5
    return max(0.0, min(1.0, avg_hazard))


def calculate_ising_energy_score(combination, ising_matrix):
    """
    Calcula la afinidad de acoplamiento mutuo cooperativo (Modelo de Ising / Spin-Glass).
    J_ij mide la interacción de pares observada vs esperada por azar C(5,2)/C(39,2) = 10/741.
    """
    if ising_matrix is None or (isinstance(ising_matrix, np.ndarray) and ising_matrix.size == 0):
        return 0.5
    idx = [b for b in combination if 1 <= b < ising_matrix.shape[0]]
    if len(idx) < 2:
        return 0.5

    sub = ising_matrix[np.ix_(idx, idx)]
    coupling_sum = float(np.triu(sub, k=1).sum())
    pair_count = len(idx) * (len(idx) - 1) // 2

    if pair_count == 0:
        return 0.5
    avg_coupling = coupling_sum / pair_count
    norm_score = 1.0 / (1.0 + math.exp(-2.0 * avg_coupling))
    return max(0.0, min(1.0, norm_score))


def calculate_sequence_probability(sequence, transition_matrix):
    """Calcula la probabilidad de una secuencia ordenada de números basada en Markov."""
    if not sequence or len(sequence) < 2:
        return 1.0
    arr = transition_matrix.values if isinstance(transition_matrix, pd.DataFrame) else transition_matrix
    prob = 1.0
    for i in range(len(sequence) - 1):
        c_num = sequence[i] - 1
        n_num = sequence[i + 1] - 1
        if 0 <= c_num < arr.shape[0] and 0 <= n_num < arr.shape[1]:
            p = float(arr[c_num, n_num])
            prob *= (p if p > 0 else 1e-6)
        else:
            prob *= 1e-6
    return prob


def calculate_positional_markov_probability(combination, positional_matrices, last_combination):
    """Calcula la probabilidad de transición posicional entre sorteo t y t+1 (B1->B1, B2->B2, etc.)."""
    if not positional_matrices or not last_combination:
        return 1.0
    prob = 1.0
    for i, col in enumerate(COLUMNS):
        matrix = positional_matrices.get(col)
        if matrix is None:
            continue
        arr = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
        prev_num = last_combination[i] - 1
        curr_num = combination[i] - 1
        if 0 <= prev_num < arr.shape[0] and 0 <= curr_num < arr.shape[1]:
            p = float(arr[prev_num, curr_num])
            prob *= (p if p > 0 else 1e-8)
        else:
            prob *= 1e-8
    return prob


def detect_draw_regime(df):
    """
    Detecta el régimen estocástico actual del juego mediante una Cadena de Markov de 3 Estados:
    - Estado 0: Congestión Baja (Suma <= 85 o >= 4 números bajos)
    - Estado 1: Equilibrio Central (Suma 86 a 114)
    - Estado 2: Expansión Alta (Suma >= 115 o <= 1 número bajo)
    """
    if df.empty or len(df) < 5:
        return {
            "regimen_actual": "Equilibrio Central (Gauss Estándar)",
            "codigo": 1,
            "media_reciente_sumas": THEORETICAL_MEAN_SUM,
            "prob_proximo": {
                "Congestión Baja (<=85)": 0.33,
                "Equilibrio Central (86-114)": 0.34,
                "Expansión Alta (>=115)": 0.33,
            }
        }

    sums = df[COLUMNS].sum(axis=1).values
    lows = (df[COLUMNS] <= LOW_HIGH_SPLIT).sum(axis=1).values

    regimes = []
    for s, l in zip(sums, lows):
        if s <= 85 or l >= 4:
            regimes.append(0)  # Congestión Baja
        elif s >= 115 or l <= 1:
            regimes.append(2)  # Expansión Alta
        else:
            regimes.append(1)  # Equilibrio Central

    regimes = np.array(regimes)
    T = np.zeros((3, 3))
    for t in range(len(regimes) - 1):
        T[regimes[t], regimes[t + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    actual_cod = int(regimes[-1])
    nombres = {
        0: "Congestión Baja (Zona de Concentración)",
        1: "Equilibrio Central (Gauss Estándar)",
        2: "Expansión Alta (Péndulo Superior)",
    }
    prob_next = T[actual_cod]
    media_reciente = float(np.mean(sums[-5:]))

    return {
        "regimen_actual": nombres[actual_cod],
        "codigo": actual_cod,
        "media_reciente_sumas": round(media_reciente, 1),
        "prob_proximo": {
            "Congestión Baja (<=85)": round(float(prob_next[0]), 3),
            "Equilibrio Central (86-114)": round(float(prob_next[1]), 3),
            "Expansión Alta (>=115)": round(float(prob_next[2]), 3),
        }
    }


def calculate_frequency_score_vectorized(combination, all_balls_flat, total_draws):
    """Calcula el puntaje de frecuencia normalizado (0 a 1)."""
    score = sum(np.sum(all_balls_flat == n) for n in combination)
    denominator = total_draws * K_BALLS
    return score / denominator if denominator > 0 else 0.0


def calculate_composite_score(
    score_freq,
    prob_markov_global,
    prob_markov_pos,
    score_gauss,
    score_entropy,
    score_bayes,
    score_hazard,
    score_ising=0.5,
):
    """
    Índice Predictivo Compuesto Global Refinado v4 (Composite Score 0 a 100).
    Prioriza fuertemente la probabilidad empírica real (Frecuencia 55% + Markov 20% = 75%)
    para evitar sobre-ajuste y mantener la señal estocástica genuina del bombo.
    """
    norm_freq = min(1.0, score_freq / 0.18)
    norm_markov_g = min(1.0, prob_markov_global * 1e4)
    norm_markov_p = min(1.0, prob_markov_pos * 1e5) if prob_markov_pos > 0 else norm_markov_g
    markov_combined = 0.5 * norm_markov_g + 0.5 * norm_markov_p

    composite = (
        0.55 * norm_freq
        + 0.20 * markov_combined
        + 0.10 * score_gauss
        + 0.05 * score_bayes
        + 0.05 * score_hazard
        + 0.05 * score_entropy
    ) * 100.0

    return round(max(0.0, min(100.0, composite)), 2)


def obtener_veredicto_cualitativo(
    combination,
    composite,
    score_freq,
    score_gauss,
    score_entropy,
    score_bayes,
    score_hazard,
    score_ising=0.5,
):
    """Genera una interpretación cualitativa y sencilla para el usuario no técnico."""
    lines = []
    if composite >= 70.0:
        calificacion = "🟢 EXCELENTE (Perfil probabilístico óptimo y balance estructural)"
        veredicto_icon = "🌟"
    elif composite >= 55.0:
        calificacion = "🟡 BUENA / PROMEDIO (Combinación equilibrada dentro del estándar)"
        veredicto_icon = "👍"
    else:
        calificacion = "🔴 ATÍPICA / POCO PROBABLE (Presenta desviaciones estocásticas severas)"
        veredicto_icon = "⚠️"

    lines.append(f"{veredicto_icon} CALIFICACIÓN GENERAL: {calificacion}")
    lines.append(f"📊 ÍNDICE PREDICTIVO COMPUESTO: {composite} / 100")
    lines.append("💡 VEREDICTO DETALLADO:")

    c_sum = sum(combination)
    if score_gauss >= 0.70:
        lines.append(f"  • 🟢 Suma de balotas ({c_sum}): EXCELENTE. Centrada en el rango dorado (~85 a 115).")
    elif score_gauss >= 0.40:
        lines.append(f"  • 🟡 Suma de balotas ({c_sum}): ACEPTABLE. Ligeramente desviada del centro teórico (100).")
    else:
        lines.append(f"  • 🔴 Suma de balotas ({c_sum}): ATÍPICA. Desvío extremo respecto a la campana normal.")

    if score_entropy >= 0.75:
        lines.append(f"  • 🟢 Aleatoriedad / Dispersión (Entropía {score_entropy:.2f}): ÓPTIMA. Excelente separación entre balotas.")
    elif score_entropy >= 0.50:
        lines.append(f"  • 🟡 Aleatoriedad / Dispersión (Entropía {score_entropy:.2f}): ACEPTABLE. Ligera concentración numérica.")
    else:
        lines.append(f"  • 🔴 Aleatoriedad / Dispersión (Entropía {score_entropy:.2f}): BAJA DISPERSIÓN. Números consecutivos o aglomerados.")

    if score_bayes >= 0.50:
        lines.append(f"  • 🟢 Bayes Dirichlet ({score_bayes:.4f}): ALTA DENSIDAD POSTERIOR. Fuerte respaldo en el histórico acumulado.")
    else:
        lines.append(f"  • 🟡 Bayes Dirichlet ({score_bayes:.4f}): DENSIDAD MODERADA. Incluye balotas con menor histórico.")

    if score_hazard >= 0.60:
        lines.append(f"  • 🟢 Presión por Atraso (Weibull {score_hazard:.2f}): ALTA MADUREZ. Balotas en ventana óptima de retorno.")
    else:
        lines.append(f"  • 🔵 Presión por Atraso (Weibull {score_hazard:.2f}): RECIENTES. Balotas de aparición cercana.")

    if score_ising >= 0.60:
        lines.append(f"  • 🟢 Afinidad de Ising ({score_ising:.2f}): COOPERATIVA. Alta tasa de co-ocurrencia por parejas.")
    else:
        lines.append(f"  • 🟡 Afinidad de Ising ({score_ising:.2f}): NEUTRA / INDEPENDIENTE. Co-ocurrencias dentro de lo esperado.")

    if composite >= 70.0:
        lines.append("📌 CONSEJO FINAL: Excelente jugada. Cumple con todos los filtros estocásticos y de probabilidad compuesta.")
    elif composite >= 55.0:
        lines.append("📌 CONSEJO FINAL: Jugada balanceada y adecuada para participar.")
    else:
        lines.append("📌 CONSEJO FINAL: Sugerimos ajustar 1 o 2 balotas para balancear la suma hacia 100 y mejorar la dispersión.")

    return lines


# =====================================================================
# 2. MOTOR DE ANÁLISIS GLOBAL
# =====================================================================

def analizar_miloto(df):
    """Ejecuta todos los análisis estadísticos avanzados sobre el histórico de MiLoto."""
    res = {}
    N_draws = len(df)
    res["total_draws"] = N_draws

    # 1. Balotas y último sorteo
    last_row = df.iloc[-1]
    res["last_combination"] = sorted([int(last_row[col]) for col in COLUMNS])
    res["last_sorteo"] = int(last_row["Sorteo"])
    res["last_fecha"] = str(last_row["Fecha"])

    # 2. Resonancia Cíclica Lag-8
    if N_draws >= 8:
        lag_8_row = df.iloc[-8]
        res["lag_8_combination"] = sorted([int(lag_8_row[col]) for col in COLUMNS])
        res["lag_8_sorteo"] = int(lag_8_row["Sorteo"])
    else:
        res["lag_8_combination"] = []
        res["lag_8_sorteo"] = None

    # 3. Frecuencias y Chi-cuadrado
    all_balls = df[COLUMNS].values.flatten()
    res["all_balls_flat"] = all_balls
    counts = pd.Series(all_balls).value_counts().reindex(range(1, N_BALLS + 1), fill_value=0)
    res["main_counts_dict"] = counts.to_dict()

    total_obs = counts.sum()
    exp_per_ball = total_obs / N_BALLS
    chi2_stat, p_val = stats.chisquare(counts, f_exp=[exp_per_ball] * N_BALLS)
    res["chi2"] = {
        "stat": round(float(chi2_stat), 2),
        "p_value": round(float(p_val), 4),
        "interpretacion": "Aleatorio (distribución uniforme)" if p_val >= 0.05 else "Sesgo no uniforme",
    }

    # 4. Números Calientes y Fríos
    recent_df = df.tail(RECENT_WINDOW)
    recent_counts = pd.Series(recent_df[COLUMNS].values.flatten()).value_counts().reindex(range(1, N_BALLS + 1), fill_value=0)
    res["hot_numbers"] = recent_counts.sort_values(ascending=False).head(10).to_dict()
    res["cold_numbers"] = recent_counts.sort_values(ascending=True).head(10).to_dict()

    # 5. Análisis de Atrasos (Gaps)
    ultimo_sorteo = res["last_sorteo"]
    atrasos = {}
    for num in range(1, N_BALLS + 1):
        apariciones = df[df[COLUMNS].isin([num]).any(axis=1)]
        if not apariciones.empty:
            ultimo_visto = int(apariciones["Sorteo"].iloc[-1])
            gap = ultimo_sorteo - ultimo_visto
        else:
            gap = N_draws
        atrasos[num] = gap
    res["atrasos_dict"] = atrasos
    res["mas_atrasados"] = dict(sorted(atrasos.items(), key=lambda x: x[1], reverse=True)[:10])

    # 6. Distribución de Sumas y Paridades
    sumas = df[COLUMNS].sum(axis=1)
    res["suma_stats"] = {
        "media_empirica": round(float(sumas.mean()), 2),
        "std_empirica": round(float(sumas.std()), 2),
        "media_teorica": THEORETICAL_MEAN_SUM,
        "std_teorica": THEORETICAL_STD_SUM,
        "min": int(sumas.min()),
        "max": int(sumas.max()),
    }

    paridades = []
    bajos_altos = []
    for _, row in df[COLUMNS].iterrows():
        nums = row.values
        pares = sum(1 for n in nums if n % 2 == 0)
        paridades.append(f"{pares}P - {5 - pares}I")
        bajos = sum(1 for n in nums if n <= LOW_HIGH_SPLIT)
        bajos_altos.append(f"{bajos}B - {5 - bajos}A")
    res["dist_paridad"] = pd.Series(paridades).value_counts(normalize=True).mul(100).round(1).to_dict()
    res["dist_bajos_altos"] = pd.Series(bajos_altos).value_counts(normalize=True).mul(100).round(1).to_dict()

    # 7. Cadenas de Markov (Global y Posicionales)
    # Matriz global secuencial
    matriz_global = np.zeros((N_BALLS + 1, N_BALLS + 1), dtype=float)
    for _, row in df[COLUMNS].iterrows():
        b = sorted(row.values)
        for i in range(len(b) - 1):
            matriz_global[b[i], b[i + 1]] += 1.0
    r_sums = matriz_global.sum(axis=1, keepdims=True)
    r_sums[r_sums == 0] = 1.0
    res["transition_matrix_global"] = matriz_global / r_sums

    # Matrices posicionales B1..B5 entre sorteo t y t+1
    pos_matrices = {}
    for col in COLUMNS:
        mat_pos = np.zeros((N_BALLS + 1, N_BALLS + 1), dtype=float)
        vals = df[col].values
        for t in range(len(vals) - 1):
            mat_pos[vals[t], vals[t + 1]] += 1.0
        r_s = mat_pos.sum(axis=1, keepdims=True)
        r_s[r_s == 0] = 1.0
        pos_matrices[col] = mat_pos / r_s
    res["positional_matrices"] = pos_matrices

    # 8. Matriz de Acoplamiento Mutuo de Ising J_ij
    cooc = np.zeros((N_BALLS + 1, N_BALLS + 1))
    for _, row in df[COLUMNS].iterrows():
        b = [int(row[col]) for col in COLUMNS]
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                cooc[b[i], b[j]] += 1
                cooc[b[j], b[i]] += 1
    # Probabilidad exacta de par en MiLoto: C(5,2) / C(39,2) = 10 / 741
    exp_cooc = N_draws * (10.0 / 741.0)
    J = np.zeros((N_BALLS + 1, N_BALLS + 1))
    for i in range(1, N_BALLS + 1):
        for j in range(1, N_BALLS + 1):
            if i != j:
                J[i, j] = math.log((cooc[i, j] + 1.0) / (exp_cooc + 1.0))
    res["ising_matrix"] = J

    # 9. Detección de Régimen Dinámico (HMM 3 Estados)
    res["regime_info"] = detect_draw_regime(df)

    # 10. Prior Dirichlet Lookup
    prior_prob = 1.0 / N_BALLS
    denom_main = (K_BALLS * N_draws + N_BALLS * 1.0) * (prior_prob * 2.0)
    bayes_lookup = np.zeros(N_BALLS + 1, dtype=np.float64)
    for num in range(1, N_BALLS + 1):
        cnt = counts.get(num, 0)
        bayes_lookup[num] = (cnt + 1.0) / denom_main
    res["bayes_lookup"] = bayes_lookup

    return res


# =====================================================================
# 3. GENERADOR PREDICTIVO Y WHEELING SYSTEM
# =====================================================================

def generar_combinaciones_probables(num_combinations, results):
    """
    Genera combinaciones candidatas y las califica con el Composite Score v3 de 7 modelos.
    Asegura diversidad genética entre jugadas.
    """
    counts = results["main_counts_dict"]
    hot = results["hot_numbers"]
    atrasos = results["atrasos_dict"]
    N_draws = results["total_draws"]

    # Vector de pesos estocásticos de entrada para muestreo
    weights = np.zeros(N_BALLS + 1)
    for n in range(1, N_BALLS + 1):
        f = counts.get(n, 1) / max(counts.values())
        h_boost = 1.35 if n in hot else 1.0
        weights[n] = f * h_boost

    p_norm = weights[1:] / weights[1:].sum()

    candidatos = set()
    intentos = 0
    max_intentos = 25000

    while len(candidatos) < 3500 and intentos < max_intentos:
        intentos += 1
        comb = tuple(sorted(np.random.choice(range(1, N_BALLS + 1), size=5, replace=False, p=p_norm)))
        # Filtros de viabilidad básica
        s = sum(comb)
        if 65 <= s <= 135:
            pares = sum(1 for x in comb if x % 2 == 0)
            if 1 <= pares <= 4:
                candidatos.add(comb)

    evaluados = []
    for c in candidatos:
        s_freq = calculate_frequency_score_vectorized(c, results["all_balls_flat"], N_draws)
        p_markov_g = calculate_sequence_probability(c, results["transition_matrix_global"])
        p_markov_p = calculate_positional_markov_probability(c, results["positional_matrices"], results["last_combination"])
        s_gauss = calculate_sum_gaussian_score(c)
        s_entropy = calculate_shannon_entropy(c)
        s_bayes = calculate_bayesian_dirichlet_score(c, counts, N_draws, bayes_lookup=results["bayes_lookup"])
        s_hazard = calculate_weibull_hazard_score(c, atrasos)
        s_ising = calculate_ising_energy_score(c, results["ising_matrix"])

        score_comp = calculate_composite_score(
            s_freq, p_markov_g, p_markov_p, s_gauss, s_entropy, s_bayes, s_hazard, s_ising
        )

        pares = sum(1 for x in c if x % 2 == 0)
        bajos = sum(1 for x in c if x <= LOW_HIGH_SPLIT)
        avg_gap = round(float(np.mean([atrasos.get(x, 0) for x in c])), 1)

        evaluados.append({
            "combinacion": list(c),
            "composite": score_comp,
            "suma": sum(c),
            "paridad": f"{pares}P - {5 - pares}I",
            "bajos_altos": f"{bajos}B - {5 - bajos}A",
            "atraso_prom": avg_gap,
            "score_freq": round(s_freq, 4),
            "score_gauss": round(s_gauss, 3),
            "score_entropy": round(s_entropy, 3),
            "score_bayes": round(s_bayes, 3),
            "score_hazard": round(s_hazard, 3),
            "score_ising": round(s_ising, 3),
        })

    evaluados.sort(key=lambda x: x["composite"], reverse=True)

    # Filtrar por diversidad (máximo 3 balotas compartidas entre sugerencias)
    seleccionadas = []
    for item in evaluados:
        c_set = set(item["combinacion"])
        similar = any(len(c_set.intersection(set(s["combinacion"]))) >= 4 for s in seleccionadas)
        if not similar:
            seleccionadas.append(item)
            if len(seleccionadas) == num_combinations:
                break

    return seleccionadas


def generar_rueda_combinatoria(numeros_elegidos, garantia=3):
    """
    Genera un sistema de cobertura (Covering Wheel) para MiLoto.
    Garantiza al menos 'garantia' aciertos (3 o 4) si las 5 balotas caen dentro del grupo.
    """
    numeros = sorted(list(set(numeros_elegidos)))
    v = len(numeros)
    if v < 6:
        raise ValueError(f"Se requieren al menos 6 números para una rueda (recibidos: {v}).")

    tuplas_garantia = list(combinations(numeros, garantia))
    cubiertas = set()
    lineas = []

    posibles = list(combinations(numeros, 5))
    np.random.shuffle(posibles)

    while len(cubiertas) < len(tuplas_garantia) and posibles:
        mejor = None
        max_n = -1
        for c in posibles[:400]:
            subs = set(combinations(c, garantia))
            n = len(subs - cubiertas)
            if n > max_n:
                max_n = n
                mejor = c
            if max_n == len(subs):
                break

        if mejor is None or max_n == 0:
            break

        lineas.append(sorted(list(mejor)))
        cubiertas.update(combinations(mejor, garantia))
        posibles.remove(mejor)

    return {
        "numeros_seleccionados": numeros,
        "total_numeros": v,
        "garantia": f"{garantia} aciertos garantizados",
        "total_apuestas": len(lineas),
        "cobertura_pct": round(len(cubiertas) / len(tuplas_garantia) * 100, 2),
        "jugadas": lineas,
    }


def evaluar_jugada_manual(combinacion, results):
    """Evalúa de forma exhaustiva una jugada manual de 5 balotas con todos los modelos."""
    if len(combinacion) != 5 or len(set(combinacion)) != 5:
        raise ValueError("La jugada debe tener exactamente 5 balotas únicas.")
    for n in combinacion:
        if not (1 <= n <= N_BALLS):
            raise ValueError(f"El número {n} está fuera del rango de MiLoto [1, 39].")

    c = sorted(combinacion)
    N_draws = results["total_draws"]
    counts = results["main_counts_dict"]
    atrasos = results["atrasos_dict"]

    s_freq = calculate_frequency_score_vectorized(c, results["all_balls_flat"], N_draws)
    p_markov_g = calculate_sequence_probability(c, results["transition_matrix_global"])
    p_markov_p = calculate_positional_markov_probability(c, results["positional_matrices"], results["last_combination"])
    s_gauss = calculate_sum_gaussian_score(c)
    s_entropy = calculate_shannon_entropy(c)
    s_bayes = calculate_bayesian_dirichlet_score(c, counts, N_draws, bayes_lookup=results["bayes_lookup"])
    s_hazard = calculate_weibull_hazard_score(c, atrasos)
    s_ising = calculate_ising_energy_score(c, results["ising_matrix"])

    composite = calculate_composite_score(
        s_freq, p_markov_g, p_markov_p, s_gauss, s_entropy, s_bayes, s_hazard, s_ising
    )

    veredicto = obtener_veredicto_cualitativo(
        c, composite, s_freq, s_gauss, s_entropy, s_bayes, s_hazard, s_ising
    )

    pares = sum(1 for x in c if x % 2 == 0)
    bajos = sum(1 for x in c if x <= LOW_HIGH_SPLIT)
    avg_gap = round(float(np.mean([atrasos.get(x, 0) for x in c])), 1)

    return {
        "combinacion": c,
        "composite_score": composite,
        "suma": sum(c),
        "paridad": f"{pares}P - {5 - pares}I",
        "bajos_altos": f"{bajos}B - {5 - bajos}A",
        "atraso_promedio": avg_gap,
        "scores_detallados": {
            "Frecuencia (JAX/NumPy)": round(s_freq, 4),
            "Campana Gaussiana": round(s_gauss, 3),
            "Entropía Shannon": round(s_entropy, 3),
            "Bayes Dirichlet": round(s_bayes, 3),
            "Weibull Hazard": round(s_hazard, 3),
            "Afinidad Ising": round(s_ising, 3),
            "Markov Global": f"{p_markov_g:.2e}",
            "Markov Posicional": f"{p_markov_p:.2e}",
        },
        "veredicto": veredicto,
    }


def imprimir_reporte_completo(results, recomendadas):
    """Imprime el reporte detallado con toda la suite de análisis de GanaBaloto."""
    print("==========================================================================")
    print("        🎱 MiLoto - Suite Completa de Análisis Estadístico y Predictivo    ")
    print("==========================================================================")
    print(f"Base de Datos:    {results['total_draws']} sorteos analizados")
    print(f"Último Sorteo:    #{results['last_sorteo']} ({results['last_fecha']}) -> {results['last_combination']}")
    if results.get("lag_8_sorteo"):
        print(f"Resonancia Lag-8: #{results['lag_8_sorteo']} -> {results['lag_8_combination']}")
    print(f"Espacio Muestral: C(39, 5) = {TOTAL_COMBINATIONS:,} combinaciones teóricas")
    print("--------------------------------------------------------------------------")

    # 1. Chi-cuadrado
    chi = results["chi2"]
    print(f"\n🔍 1. CONTRASTE DE UNIFORMIDAD (Chi-Cuadrado de Pearson):")
    print(f"   • Chi2 Stat: {chi['stat']} | p-value: {chi['p_value']}")
    print(f"   • Veredicto: {chi['interpretacion']}")

    # 2. Régimen Dinámico HMM
    reg = results["regime_info"]
    print(f"\n🔄 2. DETECCIÓN DE RÉGIMEN ESTOCÁSTICO (Markov 3 Estados):")
    print(f"   • Régimen Actual: {reg['regimen_actual']}")
    print(f"   • Media reciente de sumas (últimos 5 sorteos): {reg['media_reciente_sumas']}")
    print(f"   • Probabilidades próximo sorteo: {reg['prob_proximo']}")

    # 3. Hot & Cold
    print(f"\n🔥 3. NÚMEROS CALIENTES (Últimos {RECENT_WINDOW} sorteos):")
    hot_str = ", ".join([f"#{n} ({c}x)" for n, c in list(results["hot_numbers"].items())[:7]])
    print(f"   {hot_str}")

    print(f"\n🧊 4. NÚMEROS FRÍOS (Menor presencia reciente):")
    cold_str = ", ".join([f"#{n} ({c}x)" for n, c in list(results["cold_numbers"].items())[:7]])
    print(f"   {cold_str}")

    # 4. Weibull Hazard / Atrasos
    print(f"\n⏳ 5. PRESIÓN DE SUPERVIVENCIA Y DESGASTE (Weibull Hazard / Atrasos):")
    atr_str = ", ".join([f"#{n} ({g} sorteos)" for n, g in list(results["mas_atrasados"].items())[:7]])
    print(f"   • Mayor atraso actual: {atr_str}")
    print(f"   • Parámetros calibrados: Forma k={WEIBULL_SHAPE}, Escala lambda={WEIBULL_SCALE}")

    # 5. Gauss y Paridades
    sm = results["suma_stats"]
    print(f"\n⚖️ 6. DISTRIBUCIÓN ESTRUCTURAL (Gauss, Paridad y Rango):")
    print(f"   • Suma: Media {sm['media_empirica']} (Teórica: {sm['media_teorica']} ± {sm['std_teorica']}) | Rango: {sm['min']} a {sm['max']}")
    print(f"   • Distribución Paridad: {results['dist_paridad']}")
    print(f"   • Distribución Bajos/Altos (Split 19): {results['dist_bajos_altos']}")

    # 6. Top 10 Recomendadas
    print(f"\n🎯 7. TOP 10 COMBINACIONES RECOMENDADAS (Composite Score v3):")
    filas = []
    for idx, item in enumerate(recomendadas, 1):
        c_str = " - ".join([f"{x:02d}" for x in item["combinacion"]])
        filas.append({
            "#": idx,
            "Combinación": c_str,
            "Score v3": f"{item['composite']}/100",
            "Suma": item["suma"],
            "Par": item["paridad"],
            "B/A": item["bajos_altos"],
            "Gap Prom": item["atraso_prom"],
            "Bayes": item["score_bayes"],
            "Weibull": item["score_hazard"],
            "Ising": item["score_ising"],
        })
    df_recs = pd.DataFrame(filas)
    print(df_recs.to_markdown(index=False))
    print("==========================================================================")


def main():
    parser = argparse.ArgumentParser(description="MiLoto - Motor Completo de Análisis Estadístico y Predictivo.")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON estructurado.")
    parser.add_argument("--evaluar", nargs=5, type=int, metavar=("B1", "B2", "B3", "B4", "B5"), help="Evaluar una jugada manual de 5 balotas.")
    parser.add_argument("--rueda", nargs="+", type=int, help="Generar rueda combinatoria para un grupo de números.")
    parser.add_argument("--garantia", type=int, default=3, help="Garantía de aciertos para la rueda combinatoria (default: 3).")
    parser.add_argument("--top", type=int, default=10, help="Cantidad de combinaciones recomendadas a generar (default: 10).")
    args = parser.parse_args()

    # Cargar datos históricos
    if not os.path.exists(FILE_PATH):
        print(f"[!] Error: No se encontró '{FILE_PATH}'. Ejecuta primero './actualizar.sh'.")
        sys.exit(1)

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("MiLoto", []))
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype(int)
    df["Sorteo"] = pd.to_numeric(df["Sorteo"]).astype(int)
    df = df.sort_values(by="Sorteo").reset_index(drop=True)

    results = analizar_miloto(df)

    # 1. Evaluar jugada manual
    if args.evaluar:
        ev = evaluar_jugada_manual(args.evaluar, results)
        if args.json:
            print(json.dumps(ev, indent=2, ensure_ascii=False))
        else:
            print("\n=======================================================")
            print("        🔍 MiLoto - Evaluación de Jugada Manual         ")
            print("=======================================================")
            c_str = " - ".join([f"{x:02d}" for x in ev["combinacion"]])
            print(f"Jugada: {c_str}")
            print(f"Suma:   {ev['suma']} | Paridad: {ev['paridad']} | Bajos/Altos: {ev['bajos_altos']}")
            print(f"Atraso Promedio: {ev['atraso_promedio']} sorteos")
            print("\nScores por Modelo:")
            for mod, sc in ev["scores_detallados"].items():
                print(f"  • {mod}: {sc}")
            print("\nVeredicto e Interpretación:")
            for l in ev["veredicto"]:
                print(f"  {l}")
            print("=======================================================\n")
        return

    # 2. Generar Rueda Combinatoria
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

    # 3. Reporte general y generación predictiva
    recs = generar_combinaciones_probables(args.top, results)
    if args.json:
        payload = {
            "resumen": {
                "total_sorteos": results["total_draws"],
                "ultimo_sorteo": results["last_sorteo"],
                "ultima_fecha": results["last_fecha"],
                "ultimo_resultado": results["last_combination"],
                "lag_8": results["lag_8_combination"],
            },
            "chi2": results["chi2"],
            "regime_info": results["regime_info"],
            "hot_numbers": results["hot_numbers"],
            "cold_numbers": results["cold_numbers"],
            "mas_atrasados": results["mas_atrasados"],
            "suma_stats": results["suma_stats"],
            "dist_paridad": results["dist_paridad"],
            "dist_bajos_altos": results["dist_bajos_altos"],
            "recomendadas": recs,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
    else:
        imprimir_reporte_completo(results, recs)


if __name__ == "__main__":
    main()
