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
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import pandas as pd

FILE_PATH = "miloto.json"
_SSL_WARNED = False
_SSL_LOCK = threading.Lock()
PRIZE_FIELDS = ("Monto_Ganado",)
DEPRECATED_PRIZE_FIELDS = (
    "Acumulado",
    "Cayo_Acumulado",
    "Ganadores_5",
    "Premio_Mayor",
    "Total_Ganadores",
    "Premiacion_Total",
)
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


def _limpiar_campos_obsoletos(sorteo):
    """Devuelve una copia del sorteo sin campos de premios obsoletos."""
    return {k: v for k, v in sorteo.items() if k not in DEPRECATED_PRIZE_FIELDS}


def _peticion_get(url, client=None, timeout=15):
    """Realiza una petición GET con reintento automático sin verificación SSL si el certificado está vencido."""
    global _SSL_WARNED
    http_client = client or requests
    if _SSL_WARNED:
        return http_client.get(url, headers=HEADERS, timeout=timeout, verify=False)
    try:
        return http_client.get(url, headers=HEADERS, timeout=timeout)
    except requests.exceptions.SSLError:
        with _SSL_LOCK:
            if not _SSL_WARNED:
                print("[!] Aviso SSL: certificado de Baloto vencido, reintentando sin verificación SSL...")
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                _SSL_WARNED = True
        return http_client.get(url, headers=HEADERS, timeout=timeout, verify=False)


def extraer_sorteos_pagina_baloto(page_num, session=None):
    """Extrae la lista de sorteos desde una página del portal oficial de Baloto."""
    url = f"https://www.baloto.com/miloto/resultados/?page={page_num}"
    try:
        r = _peticion_get(url, client=session, timeout=15)
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


def extraer_detalle_sorteo_baloto(sorteo_num, session=None):
    """Extrae el monto ganado con las 5 balotas desde la página de detalle de Baloto."""
    url = f"https://www.baloto.com/miloto/resultados-miloto/{sorteo_num}/"
    try:
        r = _peticion_get(url, client=session, timeout=15)
        if r.status_code != 200:
            return {}

        soup = BeautifulSoup(r.text, "html.parser")
        texto = soup.get_text("\n", strip=True)
        detalle = {}

        m_5 = re.search(
            r"Aciertos\s+5\s+Premio\s+total\s+(\$[\d\.]+)\s+Premio por ganador\s+(\$[\d\.]+)\s+Ganadores\s+([\d\.]+)",
            texto,
            re.IGNORECASE,
        )
        if m_5:
            ganadores_5 = int(m_5.group(3).replace(".", ""))
            detalle["Monto_Ganado"] = m_5.group(1) if ganadores_5 > 0 else "$0"

        return detalle
    except Exception as e:
        print(f"[!] Error al consultar detalle del sorteo {sorteo_num}: {e}")
        return {}


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


def obtener_resultados(modo="reciente", max_pages=62, max_workers=20):
    """
    Obtiene los resultados de MiLoto.
    - 'reciente': consulta las 2 primeras páginas (últimos 10 a 20 sorteos).
    - 'historico': consulta todas las páginas (o precarga desde miloto.json si ya supera 600 sorteos) y completa detalles de 5 balotas.
    """
    prev_by_sorteo = {}
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                data_prev = json.load(f)
                prev_by_sorteo = {
                    int(item["Sorteo"]): _limpiar_campos_obsoletos(item)
                    for item in data_prev.get("MiLoto", [])
                    if "Sorteo" in item
                }
        except Exception:
            pass

    sorteos_dict = {}
    with requests.Session() as session:
        adapter = HTTPAdapter(pool_connections=30, pool_maxsize=30)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        if modo == "historico" and len(prev_by_sorteo) > 600:
            sorteos_dict = {k: dict(v) for k, v in prev_by_sorteo.items()}
            print(f"[*] Precargados {len(sorteos_dict)} sorteos desde {FILE_PATH}; verificando páginas 1 a 2...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                resultados_paginas = list(
                    executor.map(lambda p: extraer_sorteos_pagina_baloto(p, session=session), [1, 2])
                )
            for lista_p in resultados_paginas:
                for s in lista_p:
                    sorteos_dict[s["Sorteo"]] = s
        else:
            total_pages = 2 if modo == "reciente" else max_pages
            print(f"[*] Consultando portal oficial de Baloto (páginas 1 a {total_pages})...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                resultados_paginas = list(
                    executor.map(
                        lambda p: extraer_sorteos_pagina_baloto(p, session=session),
                        range(1, total_pages + 1),
                    )
                )
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
                        res_recuperados = list(
                            executor.map(lambda n: extraer_sorteo_quecayo(n, session=session), sorted(list(faltantes)))
                        )

                    recuperados_count = 0
                    for r in res_recuperados:
                        if r:
                            sorteos_dict[r["Sorteo"]] = r
                            recuperados_count += 1
                    print(f"[✓] {recuperados_count} sorteos recuperados exitosamente.")

        # Reutilizar Ganadores_5 y Monto_Ganado ya almacenados en miloto.json si existen
        for s_num, sorteo in sorteos_dict.items():
            prev = prev_by_sorteo.get(s_num)
            if prev and "Monto_Ganado" in prev:
                for field in PRIZE_FIELDS:
                    if field in prev:
                        sorteo[field] = prev[field]

        # Consultar detalle de premios (5 balotas) para sorteos sin Monto_Ganado
        pendientes = sorted(k for k, v in sorteos_dict.items() if "Monto_Ganado" not in v)
        if pendientes:
            print(f"[*] Consultando ganadores y monto ganado con 5 balotas ({len(pendientes)} sorteos)...")
            with ThreadPoolExecutor(max_workers=25) as executor:
                detalles = list(
                    executor.map(lambda n: extraer_detalle_sorteo_baloto(n, session=session), pendientes)
                )
            for s_num, detalle in zip(pendientes, detalles):
                if detalle:
                    sorteos_dict[s_num].update(detalle)

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
            "Monto Ganado (5 Balotas)": s.get("Monto_Ganado", "-"),
        })

    df = pd.DataFrame(filas)
    print(df.to_markdown(index=False))

    ganadores_mayor = [s for s in nuevos_sorteos if s.get("Monto_Ganado") not in (None, "-", "$0")]
    if ganadores_mayor:
        print("\n🏆 Sorteos con ganador del Premio Mayor (5 balotas):")
        for g in ganadores_mayor:
            combinacion = f"{g['B1']:02d} - {g['B2']:02d} - {g['B3']:02d} - {g['B4']:02d} - {g['B5']:02d}"
            print(
                f"   • Sorteo #{g['Sorteo']} ({g['Fecha']}) | {combinacion} | "
                f"Monto ganado: {g.get('Monto_Ganado', '-')}"
            )

    return True


def actualizar_json(sorteos_obtenidos, file_path=FILE_PATH):
    """Guarda o actualiza la base de datos histórica en JSON."""
    datos_actuales = {"MiLoto": []}
    existentes_dict = {}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                datos_actuales = json.load(f)
                if "MiLoto" not in datos_actuales:
                    datos_actuales["MiLoto"] = []
                for item in datos_actuales["MiLoto"]:
                    if "Sorteo" in item:
                        existentes_dict[int(item["Sorteo"])] = item
        except Exception as e:
            print(f"[!] Advertencia al leer {file_path}: {e}")

    nuevos = []
    enriquecidos = []
    for s in sorteos_obtenidos:
        sid = int(s["Sorteo"])
        s_clean = _limpiar_campos_obsoletos(s)
        if sid not in existentes_dict:
            nuevos.append(s_clean)
        else:
            combinado = _limpiar_campos_obsoletos({**existentes_dict[sid], **s_clean})
            if combinado != existentes_dict[sid]:
                enriquecidos.append(s_clean)

    tiene_obsoletos = any(
        any(field in item for field in DEPRECATED_PRIZE_FIELDS)
        for item in existentes_dict.values()
    )

    if not nuevos and not enriquecidos and not tiene_obsoletos and os.path.exists(file_path):
        print("\n[!] La base de datos ya está al día. No hay sorteos nuevos ni premios pendientes por agregar.")
        return False

    # Combinar existentes, enriquecidos y nuevos limpiando campos obsoletos
    combinados = {
        sid: _limpiar_campos_obsoletos(item)
        for sid, item in existentes_dict.items()
    }
    for s in sorteos_obtenidos:
        sid = int(s["Sorteo"])
        s_clean = _limpiar_campos_obsoletos(s)
        if sid in combinados:
            combinados[sid] = _limpiar_campos_obsoletos({**combinados[sid], **s_clean})
        else:
            combinados[sid] = s_clean

    lista_ordenada = sorted(combinados.values(), key=lambda x: int(x["Sorteo"]))
    datos_actuales["MiLoto"] = lista_ordenada

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(datos_actuales, f, ensure_ascii=False, indent=4)

    print(f"\n[OK] Base de datos '{file_path}' actualizada con éxito.")
    print(f"     Total registros: {len(lista_ordenada)} sorteos (Sorteo #{lista_ordenada[0]['Sorteo']} a #{lista_ordenada[-1]['Sorteo']}).")
    if nuevos:
        print(f"     Nuevos sorteos agregados: {len(nuevos)}.")
    if enriquecidos:
        print(f"     Sorteos actualizados con ganadores y monto (5 balotas): {len(enriquecidos)}.")
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

    # Identificar sorteos nuevos o enriquecidos con información de ganadores y premios
    existentes_dict = {}
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                existentes_dict = {
                    int(x["Sorteo"]): x
                    for x in d.get("MiLoto", [])
                    if "Sorteo" in x
                }
        except Exception:
            pass

    nuevos = [s for s in sorteos if int(s["Sorteo"]) not in existentes_dict]
    enriquecidos = [
        s for s in sorteos
        if int(s["Sorteo"]) in existentes_dict
        and _limpiar_campos_obsoletos({**existentes_dict[int(s["Sorteo"])], **s}) != existentes_dict[int(s["Sorteo"])]
    ]
    mostrar = nuevos if nuevos else (enriquecidos if enriquecidos else sorteos[-5:])

    presentar_resumen(mostrar)

    if not nuevos and not enriquecidos and os.path.exists(FILE_PATH):
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
