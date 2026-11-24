import json
import sys

solicitudes = json.load(sys.stdin)["solicitudes"]
resultados = []

for solicitud in solicitudes:
    puntaje = solicitud["impacto"] * solicitud["urgencia"]

    # Regla didáctica de clasificación.
    if puntaje >= 6:
        prioridad = "alta"
    elif puntaje >= 3:
        prioridad = "media"
    else:
        prioridad = "baja"

    resultados.append({
        "codigo": solicitud["codigo"],
        "impacto": solicitud["impacto"],
        "urgencia": solicitud["urgencia"],
        "puntaje": puntaje,
        "prioridad": prioridad,
    })

print(json.dumps({"resultados": resultados}, ensure_ascii=False))