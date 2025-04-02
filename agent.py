import os
os.environ["LANGCHAIN_TRACING"] = "false"  # Desactiva el seguimiento de LangChain
import logging

# Configuración inicial de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False  # Evita propagación duplicada de logs

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
        thought = {
            "type": "pensamiento",
            "content": "🧠 Analizando la consulta..."
        }
        if not any(t['content'] == thought['content'] for t in self.thoughts):
            self.thoughts.append(thought)

    def on_llm_end(self, response, **kwargs):
        content = f"💭 {response.generations[0][0].text}"
        if not any(t['content'] == content for t in self.thoughts):
            self.thoughts.append({
                "type": "razonamiento",
                "content": content
            })

    def on_agent_action(self, action, **kwargs):
        content = f"🤔 {action.log}"
        if not any(t['content'] == content for t in self.thoughts):
            self.current_step = {
                "type": "acción",
                "content": content
            }
            self.thoughts.append(self.current_step)

    def on_tool_start(self, serialized, input_str, **kwargs):
        content = f"```sql\n{input_str}\n```"
        if not any(t['content'] == content for t in self.thoughts):
            self.current_step = {
                "type": "herramienta",
                "name": serialized["name"],
                "content": content
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
    logger.error(f"Error en DB: {e}")
    db = None

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="gemma-3-4b-it@q4_k_m",
    temperature=0,
    verbose=False  # Desactiva logs internos del LLM
)

@cl.on_chat_start
async def start_chat():
    if not db:
        await cl.Message(content="❌ Error en conexión a DB").send()
        return

    try:
        # Inicializamos el buffer de memoria
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Creamos el agente con memoria y nuestro callback personalizado
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,  # Desactivamos verbose para evitar logs duplicados
            handle_parsing_errors=True,
            memory=memory,
            callbacks=[AgentCallbackHandler()],  # Usamos solo nuestro callback
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        # Guardamos el agente en la sesión del usuario
        cl.user_session.set("agent", agent)
        cl.user_session.set("runnable", agent)

        await cl.Message(content="✅ Asistente listo con memoria activada").send()
    except Exception as e:
        logger.error(f"Error en start_chat: {str(e)}")
        await cl.Message(content=f"❌ Error: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    if not agent:
        await cl.Message(content="❌ Error: El agente no está configurado.").send()
        return

    msg = cl.Message(content="Procesando...")
    await msg.send()

    try:
        # Usamos ainvoke para operaciones asíncronas
        response = await agent.ainvoke(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        )

        # Actualizamos el mensaje con la respuesta
        msg.content = response.get("output", "No se recibió respuesta")
        await msg.update()
    except Exception as e:
        logger.error(f"Error en on_message: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        await cl.Message(content=error_msg).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)