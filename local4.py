import os
os.environ["LANGCHAIN_TRACING"] = "false"
import logging

# Configuración de logging
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

class FormattedAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []
        self.current_step = {}
        self.last_output = None

    async def on_llm_start(self, serialized, prompts, **kwargs):
        # No necesitamos mostrar esto en la UI
        pass

    async def on_llm_end(self, response, **kwargs):
        # No necesitamos mostrar esto en la UI
        pass

    async def on_agent_action(self, action, **kwargs):
        # Mostrar el razonamiento del agente
        if action.log and "Action Input" not in action.log:
            await cl.Message(
                content=f"**Razonamiento:** {action.log}",
                author="Agent"
            ).send()

    async def on_tool_start(self, serialized, input_str, **kwargs):
        # Formatear consulta SQL como bloque de código
        sql_content = cl.Text(content=input_str, language="sql")
        await cl.Message(
            content="**Consulta SQL a ejecutar:**",
            elements=[sql_content],
            author="Database"
        ).send()
        self.current_step = {"sql": input_str}

    async def on_tool_end(self, output, **kwargs):
        if output and output != self.last_output:
            # Formatear output como JSON o texto plano según corresponda
            try:
                # Intentar formatear como JSON si es posible
                import json
                formatted_output = json.dumps(output, indent=2) if not isinstance(output, str) else output
                lang = "json" if not isinstance(output, str) else "text"
            except:
                formatted_output = str(output)
                lang = "text"

            output_content = cl.Text(content=formatted_output, language=lang)
            await cl.Message(
                content="**Resultado de la consulta:**",
                elements=[output_content],
                author="Database"
            ).send()
            self.last_output = output

# Configuración inicial de la base de datos
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
    available_tables = set(db.get_usable_table_names())
    logger.info(f"Tablas disponibles: {available_tables}")
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

prompt = f"""
Eres un asistente de hotel que responde únicamente con información de la base de datos.
Reglas:
1. Usa los nombres correctos de las tablas: {available_tables}
2. Para listas de datos, usa formato Markdown en tablas.
3. Si no puedes responder con los datos disponibles, indícalo claramente.
4. Muestra las consultas SQL y resultados en formato adecuado.
"""

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
            callbacks=[FormattedAgentCallbackHandler()],
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": True
            }
        )

        cl.user_session.set("agent", agent)
        await cl.Message(content="✅ Asistente listo. Puedes hacer preguntas sobre la base de datos del hotel.").send()
    except Exception as e:
        logger.error(f"Error en start_chat: {str(e)}")
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    callback_handler = FormattedAgentCallbackHandler()  # Usamos el callback personalizado

    if not agent:
        await cl.Message(content="❌ Error: El agente no está configurado.").send()
        return

    msg = cl.Message(content="Procesando...")
    await msg.send()

    try:
        # Ejecutamos el agente de forma asíncrona con el callback personalizado
        response = await agent.ainvoke(
            {"input": message.content},
            config=RunnableConfig(callbacks=[callback_handler]),
        )

        # Construimos los elementos desplegables para los pasos de la Chain of Thought
        elements = []
        for step in callback_handler.steps:
            step_title = f"{step['type'].title()}"
            step_content = step["content"]
            if "output" in step:
                step_content += f"\n{step.get('output', '')}"
            elements.append(
                cl.Accordion(
                    title=step_title,
                    content=step_content.strip()
                )
            )

        # Construimos el mensaje final con los pasos desplegables y la respuesta al final
        final_content = f"## Respuesta\n{response.get('output', 'No se recibió respuesta')}"

        # Actualizamos el mensaje con el contenido final y los elementos desplegables
        msg.content = final_content
        msg.elements = elements
        await msg.update()
    except Exception as e:
        logger.error(f"Error en on_message: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)