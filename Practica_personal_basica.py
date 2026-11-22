import os
from pathlib import Path

from dotenv import load_dotenv
#from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

#llm = ChatOpenRouter(
    #api_key=os.environ["OPENROUTER_API_KEY"],
    #model=os.environ["LLM_MODEL"],
#)
llm = ChatOpenAI(
    api_key=os.environ["OPENIA_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

mensaje_systema=SystemMessage(content="Responde de manera breve")
mensaje_cliente= HumanMessage(content="Ques es la IA")

#mensajes=[SystemMessage(content="Responde de forma breve."),HumanMessage(content="Que es Python?")]

respuesta=llm.invoke([mensaje_systema, mensaje_cliente])

print(respuesta.content)
