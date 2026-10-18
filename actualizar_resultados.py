#!/usr/bin/env python3
"""
MiLoto - Actualizador de Resultados
Herramienta para consultar, raspar y actualizar los resultados históricos de MiLoto (Baloto).
"""

import sys
import os
import re
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
import pandas as pd

FILE_PATH = "miloto.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

MESES = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}


def parsear_fecha_es(fecha_texto):
    """Convierte texto como '16 de Octubre de 2026' a formato 'YYYY-MM-DD'."""
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", fecha_texto, re.IGNORECASE)
    if m:
        dia = m.group(1).zfill(2)
        mes = MESES.get(m.group(2).lower(), "01")
        anio = m.group(3)
        return f"{anio}-{mes}-{dia}"
    return None


def extraer_sorteos_pagina_baloto(page_num, session=None):
    """Extrae la lista de sorteos desde una página del portal oficial de Baloto."""
    url = f"https://www.baloto.com/miloto/resultados/?page={page_num}"
    client = session or requests
    try:
        r = client.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return []

        sorteos = []
        for row in tables[0].find_all("tr"):
            cols = [td.text.strip() for td in row.find_all(["td", "th"])]
            links = [a.get("href") for a in row.find_all("a")]
            detail_links = [l for l in links if "/miloto/resultados-miloto/" in (l or "")]

            if len(cols) >= 2 and detail_links:
                link = detail_links[0]
                m_sorteo = re.search(r"/miloto/resultados-miloto/(\d+)", link)
                sorteo_num = int(m_sorteo.group(1)) if m_sorteo else None

                fecha_iso = parsear_fecha_es(cols[0])
                numeros = [int(n) for n in re.findall(r"\d+", cols[1])]

                if sorteo_num and fecha_iso and len(numeros) == 5:
                    sorteos.append({
                        "Sorteo": sorteo_num,
                        "Fecha": fecha_iso,
                        "B1": numeros[0],
                        "B2": numeros[1],
                        "B3": numeros[2],
                        "B4": numeros[3],
                        "B5": numeros[4],
                    })
        return sorteos
    except Exception as e:
        print(f"[!] Error al consultar página {page_num} en Baloto: {e}")
        return []


def extraer_sorteo_quecayo(sorteo_num, session=None):
    """Fuente de contingencia: extrae un sorteo específico desde QueCayo."""
    url = f"http://quecayo.com/miloto/sorteo-{sorteo_num}"
    client = session or requests
    try:
        r = client.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text()

        fecha_iso = parsear_fecha_es(text)
        if not fecha_iso:
            return None

        balls = []
        for elem in soup.find_all(class_=lambda x: x and ("ball" in x.lower() or "numero" in x.lower() or "circle" in x.lower())):
            val = elem.text.strip()
            if val.isdigit() and len(balls) < 5:
                balls.append(int(val))

        if len(balls) == 5:
            return {
                "Sorteo": sorteo_num,
                "Fecha": fecha_iso,
                "B1": balls[0],
                "B2": balls[1],
                "B3": balls[2],
                "B4": balls[3],
                "B5": balls[4],
            }
    except Exception as e:
        print(f"[!] Error en contingencia QueCayo para sorteo {sorteo_num}: {e}")
    return None


def obtener_resultados(modo="reciente", max_pages=62, max_workers=10):
    """
    Obtiene los resultados de MiLoto.
    - 'reciente': consulta las 2 primeras páginas (últimos 10 a 20 sorteos).
    - 'historico': consulta en paralelo todas las páginas (1 a max_pages) y llena vacíos con contingencia.
    """
    total_pages = 2 if modo == "reciente" else max_pages
    print(f"[*] Consultando portal oficial de Baloto (páginas 1 a {total_pages})...")

    sorteos_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resultados_paginas = list(executor.map(extraer_sorteos_pagina_baloto, range(1, total_pages + 1)))

    for lista_p in resultados_paginas:
        for s in lista_p:
            sorteos_dict[s["Sorteo"]] = s

    # En modo histórico, verificar si hay huecos en la secuencia y completarlos con QueCayo
    if modo == "historico" and sorteos_dict:
        min_s = min(sorteos_dict.keys())
        max_s = max(sorteos_dict.keys())
        faltantes = set(range(min_s, max_s + 1)) - set(sorteos_dict.keys())

        if faltantes:
            print(f"[*] Se detectaron {len(faltantes)} sorteos omitidos en portal oficial: {sorted(list(faltantes))}")
            print("[*] Recuperando desde fuente de contingencia (QueCayo)...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                res_recuperados = list(executor.map(extraer_sorteo_quecayo, sorted(list(faltantes))))

            recuperados_count = 0
            for r in res_recuperados:
                if r:
                    sorteos_dict[r["Sorteo"]] = r
                    recuperados_count += 1
            print(f"[✓] {recuperados_count} sorteos recuperados exitosamente.")

    lista_final = sorted(sorteos_dict.values(), key=lambda x: x["Sorteo"])
    return lista_final


def presentar_resumen(nuevos_sorteos):
    """Muestra un resumen en consola de los sorteos encontrados."""
    if not nuevos_sorteos:
        print("[!] No se encontraron nuevos resultados.")
        return False

    print("\n### 🎱 Sorteos de MiLoto detectados:")
    filas = []
    for s in nuevos_sorteos:
        combinacion = f"{s['B1']:02d} - {s['B2']:02d} - {s['B3']:02d} - {s['B4']:02d} - {s['B5']:02d}"
        filas.append({
            "Sorteo": f"#{s['Sorteo']}",
            "Fecha": s["Fecha"],
            "Combinación Ganadora": combinacion,
        })

    df = pd.DataFrame(filas)
    print(df.to_markdown(index=False))
    return True


def actualizar_json(sorteos_obtenidos, file_path=FILE_PATH):
    """Guarda o actualiza la base de datos histórica en JSON."""
    datos_actuales = {"MiLoto": []}
    sorteos_existentes = set()

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                datos_actuales = json.load(f)
                if "MiLoto" not in datos_actuales:
                    datos_actuales["MiLoto"] = []
                for item in datos_actuales["MiLoto"]:
                    if "Sorteo" in item:
                        sorteos_existentes.add(int(item["Sorteo"]))
        except Exception as e:
            print(f"[!] Advertencia al leer {file_path}: {e}")

    nuevos = []
    for s in sorteos_obtenidos:
        if s["Sorteo"] not in sorteos_existentes:
            nuevos.append(s)

    if not nuevos and os.path.exists(file_path):
        print("\n[!] La base de datos ya está al día. No hay sorteos nuevos por agregar.")
        return False

    # Combinar existentes y nuevos
    combinados = {int(x["Sorteo"]): x for x in datos_actuales["MiLoto"]}
    for s in sorteos_obtenidos:
        combinados[int(s["Sorteo"])] = s

    lista_ordenada = sorted(combinados.values(), key=lambda x: int(x["Sorteo"]))
    datos_actuales["MiLoto"] = lista_ordenada

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(datos_actuales, f, ensure_ascii=False, indent=4)

    print(f"\n[OK] Base de datos '{file_path}' actualizada con éxito.")
    print(f"     Total registros: {len(lista_ordenada)} sorteos (Sorteo #{lista_ordenada[0]['Sorteo']} a #{lista_ordenada[-1]['Sorteo']}).")
    if nuevos:
        print(f"     Nuevos sorteos agregados: {len(nuevos)}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Actualizador de resultados de MiLoto (Baloto).")
    parser.add_argument("--auto", action="store_true", help="Modo automático: guarda sin confirmación interactiva.")
    parser.add_argument("--backfill", "--historico", dest="historico", action="store_true", help="Descargar todo el historial completo (páginas 1 a 62).")
    parser.add_argument("--pages", type=int, default=62, help="Cantidad máxima de páginas a consultar en modo histórico (default: 62).")
    args = parser.parse_args()

    es_auto = args.auto or os.getenv("AUTO_UPDATE") == "true"
    modo = "historico" if args.historico or not os.path.exists(FILE_PATH) else "reciente"

    print("=======================================================")
    print("        🎱 MiLoto - Actualizador de Resultados 🎱      ")
    print("=======================================================")
    print(f"Modo: {'Histórico Completo' if modo == 'historico' else 'Actualización Reciente'}")
    if es_auto:
        print("Ejecución: Automática (--auto)")
    print("")

    sorteos = obtener_resultados(modo=modo, max_pages=args.pages)

    if not sorteos:
        print("[!] No se obtuvieron resultados. Revisa la conexión a internet.")
        sys.exit(1)

    # Identificar cuáles son nuevos comparando con miloto.json
    existentes = set()
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                existentes = {int(x["Sorteo"]) for x in d.get("MiLoto", [])}
        except Exception:
            pass

    nuevos = [s for s in sorteos if s["Sorteo"] not in existentes]
    mostrar = nuevos if nuevos else sorteos[-5:]

    presentar_resumen(mostrar)

    if not nuevos and os.path.exists(FILE_PATH):
        print("\n[INFO] No se encontraron sorteos nuevos pendientes de guardar.")
        return

    if es_auto:
        print("\n[AUTO] Guardando automáticamente en miloto.json...")
        actualizar_json(sorteos)
    else:
        confirmar = input(f"\n¿Deseas guardar los resultados en {FILE_PATH}? (s/n): ")
        if confirmar.lower() in ["s", "si", "y", "yes"]:
            actualizar_json(sorteos)
        else:
            print("Operación cancelada por el usuario.")


if __name__ == "__main__":
    main()
