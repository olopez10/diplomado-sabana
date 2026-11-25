import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# 1. Conectarnos a OpenRouter usando el mismo .env de los chats.
BASE = Path(__file__).resolve().parent

LEER = BASE / "IntroServicios" / "leer_csv.py"
ANALIZAR = BASE / "IntroServicios" / "analizar_solicitudes.py"
CSV = BASE / "data" / "solicitudes.csv"

load_dotenv(BASE / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)


# 2. Metaprompts de los agentes.
system_orquestador = """
    Eres un clasificador de agentes especializados. Dada la instrucción
    del usuario, identifica si quiere consultar o analizar una solicitud.

    Documental: Cuando se requiera consultar información de algún documento.
    Analista: Cuando se requiera analizar algo sobre el documento.

    Identifica también los códigos de las solicitudes mencionadas
    por el usuario. No inventes códigos ni incluyas códigos repetidos.

    Responde únicamente con un JSON con la siguiente estructura:

    {
        "especializado": "Documental o Analista",
        "codigos": ["SOL-014", "SOL-015"]
    }

    Si el usuario no proporciona códigos, devuelve una lista vacía.
    No incluyas explicaciones ni bloques de Markdown.
"""

system_Analista = """
    Eres el especialista analista.
    Responde con: 1) problema, 2) información faltante,
    3) dos recomendaciones. No inventes datos ni acciones realizadas.
"""

system_Documental = """
    Eres el especialista documental.
    Responde únicamente con la información del texto proporcionado.
    Si el texto no contiene la respuesta, indícalo. No inventes datos.
    Para registrar una solicitud se requiere:
    código, descripción del problema y fecha de registro.
"""


# 3. Definición del estado.
class Estado(TypedDict, total=False):
    prompt_usuario: str
    especializado: str
    respuesta: str
    codigos: list[str]
    datos: dict


# 4. Definición de los nodos.
def orquestador(estado: Estado):

    usuario = estado["prompt_usuario"]

    instruccion_orquestador = SystemMessage(content=system_orquestador)
    solicitud_orquestador = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    # Convertir la respuesta del LLM en un diccionario.
    resultado = json.loads(respuesta.content)

    seleccion = resultado["especializado"]
    codigos = resultado["codigos"]

    if not codigos:
        return {
            "especializado": "Fin",
            "respuesta": "Indica un código, por ejemplo SOL-014."
        }

    if seleccion not in ["Documental", "Analista"]:
        seleccion = "Analista"

    return {
        "especializado": seleccion,
        "codigos": codigos
    }


def Analista(estado: Estado):

    usuario = estado["prompt_usuario"]

    # Ejecutar el servicio externo de análisis.
    proceso = subprocess.run(
        [sys.executable, str(ANALIZAR)],
        input=json.dumps({
            "solicitudes": estado["datos"]["solicitudes"]
        }),
        text=True,
        capture_output=True,
        check=True
    )

    # Construir los mensajes para el LLM.
    instruccion_orquestador = SystemMessage(content=system_Analista)

    solicitud_orquestador = HumanMessage(
        content=f"""
        Pregunta: {usuario}
        Resultados del análisis: {proceso.stdout}
        Solicitudes no encontradas: {estado["datos"]["no_encontradas"]}

        Explica las prioridades utilizando únicamente los resultados
        del servicio y siguiendo el formato indicado.
        """
    )

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    return {"respuesta": respuesta.content}


def Documental(estado: Estado):

    usuario = estado["prompt_usuario"]

    # Ejecutar el servicio externo de lectura.
    proceso = subprocess.run(
        [sys.executable, str(LEER), str(CSV)],
        input=json.dumps({
            "codigos": estado["codigos"]
        }),
        text=True,
        capture_output=True,
        check=True
    )

    # Convertir el resultado JSON en un diccionario de Python.
    datos = json.loads(proceso.stdout)

    # Verificar si se encontraron las solicitudes.
    if not datos["solicitudes"]:
        return {
            "datos": datos,
            "respuesta": "No se encontraron las solicitudes indicadas."
        }

    # Si se requiere un análisis, almacenar los datos en el estado.
    # Posteriormente, estos serán utilizados por el agente Analista.
    if estado["especializado"] == "Analista":
        return {"datos": datos}

    # Si únicamente se requiere consultar, construir la respuesta.
    instruccion_orquestador = SystemMessage(content=system_Documental)

    solicitud_orquestador = HumanMessage(
        content=f"""
        Pregunta: {usuario}
        Datos: {datos}

        Para esta consulta utiliza únicamente los registros recibidos.
        """
    )

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    return {
        "datos": datos,
        "respuesta": respuesta.content
    }


# 5. Funciones de enrutamiento.
def seleccionar_agente(estado: Estado):

    # Si no hay códigos, finalizar el proceso.
    if estado["especializado"] == "Fin":
        return "Fin"

    # Tanto para consultar como para analizar,
    # primero se debe recuperar la información.
    return "Documental"


def ruta_Documental(estado: Estado):

    # Si se requiere analizar y existen datos,
    # continuar hacia el agente Analista.
    if estado["especializado"] == "Analista" and estado["datos"]["solicitudes"]:
        return "Analista"

    # En caso contrario, finalizar el proceso.
    return "Fin"


# 6. Construcción del grafo.
grafo = StateGraph(Estado)

grafo.add_node("orquestador", orquestador)
grafo.add_node("Documental", Documental)
grafo.add_node("Analista", Analista)

grafo.add_edge(START, "orquestador")

grafo.add_conditional_edges(
    "orquestador",
    seleccionar_agente,
    {"Documental": "Documental", "Fin": END},
)

grafo.add_conditional_edges(
    "Documental",
    ruta_Documental,
    {"Analista": "Analista", "Fin": END},
)

grafo.add_edge("Analista", END)

agente = grafo.compile()


# 7. Ejecución del chat.
while True:

    usuario = input("Usuario: ").strip()

    if usuario.lower() == "salir":
        break

    if not usuario:
        continue

    resultado = agente.invoke({"prompt_usuario": usuario})

    print("Agente seleccionado:", resultado["especializado"])
    print("Respuesta:", resultado["respuesta"], "\n")