import os
os.environ["LANGCHAIN_TRACING"] = "false"
import logging

# Configuración inicial de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable.config import RunnableConfig


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


# Configuración de la base de datos y LLM
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
except Exception as e:
    logger.error(f"Error en DB: {e}")
    db = None

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="gemma-3-4b-it@q4_k_m",
    temperature=0,
    verbose=False
)


@cl.on_chat_start
async def start_chat():
    if not db:
        await cl.Message(content="❌ Error en conexión a DB").send()
        return

    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            memory=memory,
            callbacks=[AgentCallbackHandler()],
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        cl.user_session.set("agent", agent)
        await cl.Message(content="✅ Asistente listo con memoria activada").send()
    except Exception as e:
        logger.error(f"Error en start_chat: {str(e)}")
        await cl.Message(content=f"❌ Error: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    callback_handler = AgentCallbackHandler()

    if not agent:
        await cl.Message(content="❌ Error: El agente no está configurado.").send()
        return

    msg = cl.Message(content="Procesando...")
    await msg.send()

    try:
        response = await agent.ainvoke(
            {"input": message.content},
            config=RunnableConfig(callbacks=[callback_handler]),
        )

        # Construir el contenido del proceso con detalles desplegables
        process_content = "### Proceso de pensamiento\n\n"
        for step in callback_handler.thoughts:
            step_content = step['content']
            if "output" in step:
                step_content += f"\n\n**Resultado:**\n{step.get('output', '')}"

            # Usamos el formato de detalles HTML que soporta Markdown
            process_content += f"""
<details>
<summary><strong>{step['type'].title()}</strong></summary>

{step_content}

</details>
\n\n
            """

        # Enviar primero el proceso en acordeones
        await cl.Message(content=process_content.strip()).send()
        
        # Luego enviar la respuesta final
        await cl.Message(content=f"## Respuesta final\n\n{response.get('output', 'No se recibió respuesta')}").send()

    except Exception as e:
        logger.error(f"Error en on_message: {str(e)}")
        await cl.Message(content=f"❌ Error: {str(e)}").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)