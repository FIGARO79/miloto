---
name: miloto-analisis-predictivo
description: >-
  Use this skill when the user wants to analyze, evaluate, audit, or generate lottery combinations for MiLoto (Baloto Colombia) using the 7-dimension statistical engine (Frequencies, Markov chains, Gaussian sum, Bayesian Dirichlet, Weibull Hazard rate, Ising energy, Shannon entropy) and the 5-ball jackpot cycle pattern analyzer, or when optimizing investment capital (bankroll management) with reduced wheels.
  Palabras activadoras: PRONOSTICO, JUGADAS, MILOTO, PREMIO MAYOR, PATRON.
---

# Análisis Predictivo y Optimización de Inversión (MiLoto)

Esta skill proporciona las herramientas y procedimientos automatizados para auditar combinaciones de MiLoto, generar pronósticos basados en el **Composite Score v3 (0 a 100)** y optimizar el capital de juego del usuario (**Bankroll Management**).

---

## ⚡ Palabra Activadora Rápida (Trigger)

Cuando el usuario ingrese la palabra activadora:
* **`PRONOSTICO`** (o sus alias **`JUGADAS`**, **`MILOTO`**, **`PREMIO MAYOR`**, **`PATRON`**)

El agente debe ejecutar de inmediato el flujo de pronóstico completo estandarizado y presentar el reporte estructurado en 5 secciones (incluyendo el Análisis de Ciclo del Premio Mayor y el Tiquete Especial Perfil C), asumiendo su rol como **Consejero y Analista Profesional de Inversión**.

### 🏛️ Los 3 Pilares Permanentes del Consejero
1. **Auditoría de Brechas:** Desglosar rigurosamente tanto el Top por Momentum (Perfil A) como la Alternativa por Atraso Extremo (Perfil B) bajo el modelo Weibull.
2. **Optimización Financiera:** Enfoque estricto en los 2 tiquetes de $4.000 COP ($8.000 COP total) para cubrir **10 balotas distintas sin solapes**, abarcando el **25.6% del bombo** de 39 balotas.
3. **Rotación Transparente:** Exploración dinámica del bombo completo (1-39).

---

## Capacidades y Flujos de Trabajo

### 1. Pronóstico Completo y Optimización de Inversión (Flujo Principal)
Ejecuta el script integral que calcula candidatos optimizados, audita la jugada #1, genera la rueda reducida de 7 números y presenta la estrategia financiera óptima de 2 tiquetes ($8.000 COP).

**Comando:**
```bash
./.venv/bin/python .agents/skills/miloto-analisis-predictivo/scripts/ejecutar_pronostico_completo.py
```

Soporta `--json` para integraciones programáticas.

---

### 2. Diagnóstico y Auditoría de una Jugada Manual
Evalúa cualquier boleto de 5 balotas (1-39) contra el histórico completo y los 7 modelos matemáticos.

**Comando:**
```bash
./.venv/bin/python .agents/skills/miloto-analisis-predictivo/scripts/evaluar_jugada.py --numeros 7 8 18 38 39
```

---

### 3. Generación Personalizada de Jugadas
Genera combinaciones candidatas con filtros de puntuación mínima.

**Comando:**
```bash
./.venv/bin/python .agents/skills/miloto-analisis-predictivo/scripts/generar_recomendadas.py --cantidad 10 --min-indice 70.0
```

---

### 4. Ruedas Combinatorias Reducidas (Wheeling System)
Permite al usuario seleccionar un grupo de números (ej. 7 a 10 balotas) y generar el conjunto mínimo de tiquetes optimizados que garantiza matemáticamente 3 o 4 aciertos.

**Comando:**
```bash
./.venv/bin/python .agents/skills/miloto-analisis-predictivo/scripts/generar_rueda.py --numeros 2 7 17 18 19 23 31 --garantia 3
```
