import csv
import json
import sys

ruta_csv = sys.argv[1]
codigos = json.load(sys.stdin)["codigos"]
solicitados = set(codigos)
encontrados = {}

with open(ruta_csv, encoding="utf-8-sig", newline="") as archivo:
    lector = csv.DictReader(archivo)
    if not {"codigo", "impacto", "urgencia"}.issubset(lector.fieldnames or []):
        raise ValueError("El CSV debe tener las columnas codigo, impacto y urgencia")

    for fila in lector:
        codigo = fila["codigo"].strip().upper()
        if codigo in solicitados:
            encontrados[codigo] = {
                "codigo": codigo,
                "impacto": int(fila["impacto"]),
                "urgencia": int(fila["urgencia"]),
            }

# Solo devolvemos los registros solicitados, en el orden de la pregunta.
solicitudes = [encontrados[c] for c in codigos if c in encontrados]
no_encontradas = [c for c in codigos if c not in encontrados]
print(json.dumps({
    "solicitudes": solicitudes,
    "no_encontradas": no_encontradas,
}, ensure_ascii=False))