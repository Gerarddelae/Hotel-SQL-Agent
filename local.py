import os
import re
import traceback
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler

# Handler personalizado para mostrar pensamientos
class SQLAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thoughts = []
    
    def on_agent_action(self, action, **kwargs):
        thought = f"🤔 **Pensamiento:**\n{action.log}"
        self.thoughts.append(thought)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_input = f"🔧 **Consulta SQL generada:**\n```sql\n{input_str}\n```"
        self.thoughts.append(tool_input)
    
    def on_tool_end(self, output, **kwargs):
        tool_output = f"📊 **Resultado intermedio:**\n```json\n{output}\n```"
        self.thoughts.append(tool_output)

# Conexión a la base de datos
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
    available_tables = set(db.get_usable_table_names())
    print(f"✅ Tablas disponibles: {available_tables}")
except Exception as e:
    print(f"❌ Error en la base de datos: {e}")
    db = None

# Configuración del LLM (LM Studio)
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="gemma-3-4b-it@q4_k_m",
    temperature=0
)

# PROMPT ORIGINAL (conservado exactamente como en tu código)
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

# Creación del agente
try:
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
except Exception as e:
    print(f"❌ Error en el agente: {e}")
    agent_executor = None

@cl.on_chat_start
async def start_chat():
    if agent_executor:
        await cl.Message(content="✅ Asistente de hotel listo. ¿En qué puedo ayudarte?").send()
    else:
        await cl.Message(content="❌ Error: El agente no está disponible").send()

@cl.on_message
async def handle_message(message: cl.Message):
    if not agent_executor:
        await cl.Message(content="🔴 Error: Servicio no disponible").send()
        return

    callback_handler = SQLAgentCallbackHandler()
    msg = cl.Message(content="")  # Mensaje inicial vacío
    await msg.send()
    
    try:
        # Combinar prompt original con el mensaje
        input_con_prompt = f"{prompt}\n\nUsuario: {message.content}"
        
        # Ejecutar el agente
        respuesta = await cl.make_async(agent_executor.invoke)(
            {"input": input_con_prompt},
            callbacks=[callback_handler]
        )

        # Formatear respuesta
        respuesta_final = respuesta.get("output", str(respuesta))
        
        if callback_handler.thoughts:
            proceso = "\n\n".join(callback_handler.thoughts)
            msg.content = f"## 💬 Respuesta\n{respuesta_final}\n\n<details><summary>🔍 Ver detalles técnicos</summary>\n\n{proceso}\n</details>"
        else:
            msg.content = respuesta_final

        await msg.update()  # Actualizar SIN parámetros

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()  # Actualizar SIN parámetros
        print(traceback.format_exc())

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)