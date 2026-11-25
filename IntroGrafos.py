import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# 1. Conectarnos a OpenRouter usando el mismo .env de los chats.
load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

# 2. Metaprompts de los agentes.
system_orquestador = """
    Eres un clasificador de agentes especializados. Dada la instrucción del usuario, identifica si quiere
    consultar o analizar una solicitud. Dependiendo de esto devuelve únicamente Analista o Documental.
    Documental: Cuando se requiera consultar información de algún documento.
    Analista: Cuando se requiera analizar algo sobre el documento.
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


# 4. Definición de los nodos.
def orquestador(estado: Estado):

    usuario = estado["prompt_usuario"]
    instruccion_orquestador = SystemMessage(content=system_orquestador)
    solicitud_orquestador = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    seleccion = str(respuesta.content).strip().capitalize()

    if seleccion not in ["Documental", "Analista"]:
        seleccion = "Analista"

    return {"especializado": seleccion}


def Analista(estado: Estado):

    usuario = estado["prompt_usuario"]
    instruccion_orquestador = SystemMessage(content=system_Analista)
    solicitud_orquestador = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    return {"respuesta": respuesta.content}


def Documental(estado: Estado):

    usuario = estado["prompt_usuario"]
    instruccion_orquestador = SystemMessage(content=system_Documental)
    solicitud_orquestador = HumanMessage(content=usuario)

    respuesta = llm.invoke([
        instruccion_orquestador,
        solicitud_orquestador
    ])

    return {"respuesta": respuesta.content}


# 5. Función para recuperar el agente seleccionado.
def seleccionar_agente(estado: Estado):

    return estado["especializado"]


# 6. Construcción del grafo.
grafo = StateGraph(Estado)

grafo.add_node("orquestador", orquestador)
grafo.add_node("Documental", Documental)
grafo.add_node("Analista", Analista)

grafo.add_edge(START, "orquestador")

grafo.add_conditional_edges(
    "orquestador",
    seleccionar_agente,  # Debe ser una función de enrutamiento
    {"Documental": "Documental", "Analista": "Analista"},
)

grafo.add_edge("Documental", END)
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