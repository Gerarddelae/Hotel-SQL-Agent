import os
os.environ["LANGCHAIN_TRACING"] = "false"  # Desactiva el seguimiento de LangChain

import re
import traceback
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable.config import RunnableConfig


# Handler mejorado para capturar el proceso de pensamiento
class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thoughts = []
        self.current_step = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.thoughts.append({
            "type": "pensamiento",
            "content": "🧠 Analizando la consulta..."
        })

    def on_llm_end(self, response, **kwargs):
        self.thoughts.append({
            "type": "razonamiento",
            "content": f"💭 {response.generations[0][0].text}"
        })

    def on_agent_action(self, action, **kwargs):
        self.current_step = {
            "type": "acción",
            "content": f"🤔 {action.log}"
        }
        self.thoughts.append(self.current_step)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_step = {
            "type": "herramienta",
            "name": serialized["name"],
            "content": f"```sql\n{input_str}\n```"
        }
        self.thoughts.append(self.current_step)

    def on_tool_end(self, output, **kwargs):
        if not self.current_step.get("output"):
            self.current_step["output"] = f"```json\n{output}\n```"


# Configuración inicial
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
    available_tables = set(db.get_usable_table_names())
except Exception as e:
    print(f"Error en DB: {e}")
    db = None

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="gemma-3-4b-it@q4_k_m",
    temperature=0
)

# Prompt original conservado
prompt = f"""
Eres un asistente de hotel que responde únicamente con información de la base de datos.
Si la consulta del usuario no puede ser respondida con datos de la base de datos, responde con:
'No puedo responder a esa pregunta con la información disponible.'

Reglas:
1 No inventes datos.
2 Si la consulta devuelve una lista de datos, usa formato Markdown en tablas.
3 Si el usuario pregunta algo fuera del alcance de la base de datos, indica que no puedes responder.
5 **Usa los nombres correctos de las tablas de la base de datos:** {available_tables}
6 Si el usuario pregunta en otro idioma, traduce la respuesta al mismo idioma.
7 **Si generas una consulta SQL, ejecútala y proporciona los resultados en lugar del SQL.**
8 el usuario puede preguntar por la tabla reservaciones pero quizas en la base de datos se llame booking en ingles
"""


@cl.on_chat_start
async def start_chat():
    if not db:
        await cl.Message(content="❌ Error en conexión a DB").send()
        return

    try:
        # Inicializamos el buffer de memoria
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Creamos el agente con memoria y desactivamos callbacks de tracing
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            callbacks=[]  # Desactiva callbacks relacionados con tracing
        )

        # Guardamos el agente en la sesión del usuario
        cl.user_session.set("agent", agent)
        cl.user_session.set("runnable", agent)  # Aseguramos que 'runnable' esté configurado

        await cl.Message(content="✅ Asistente listo con memoria activada").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    # Obtenemos el runnable de la sesión del usuario
    runnable = cl.user_session.get("runnable")

    if not runnable:
        await cl.Message(content="❌ Error: El agente no está configurado.").send()
        return

    msg = cl.Message(content="Procesando...")  # Mensaje inicial
    await msg.send()

    try:
        # Ejecutar el runnable y obtener la respuesta completa
        response = runnable.invoke(  # Sin 'await' aquí
            {"input": message.content},  # Cambiado 'question' a 'input'
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        )

        # Actualizar el mensaje con la respuesta completa
        msg.content = response["output"]
        await msg.update()
    except Exception as e:
        # Manejo de errores
        error_msg = f"❌ Error: {str(e)}"
        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)