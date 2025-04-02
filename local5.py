import os
os.environ["LANGCHAIN_TRACING"] = "false"
import logging
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable.config import RunnableConfig
import json
from typing import Dict, Any

class UnifiedFormatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message_history = []
    
    async def format_content(self, content: str, content_type: str) -> str:
        """Formatea el contenido según su tipo"""
        format_map = {
            'thought': '💭 {}',
            'sql': '```sql\n{}\n```',
            'json': '```json\n{}\n```',
            'action': '🔧 {}',
            'result': '📊 {}',
            'text': '{}'
        }
        
        # Detección automática si no se especifica tipo
        if content_type == 'auto':
            if content.startswith('SELECT') or content.startswith('CREATE') or 'FROM' in content:
                content_type = 'sql'
            elif ('{' in content and '}' in content) or ('[' in content and ']' in content):
                try:
                    json.loads(content)
                    content_type = 'json'
                except:
                    content_type = 'text'
            else:
                content_type = 'text'
        
        formatted = format_map.get(content_type, '{}').format(content)
        
        # Limpieza adicional para JSON mal formado
        if content_type == 'json':
            try:
                json_content = json.loads(content)
                formatted = '```json\n' + json.dumps(json_content, indent=2) + '\n```'
            except:
                formatted = '```text\n' + content + '\n```'
        
        return formatted
    
    async def send_formatted_message(self, title: str, content: str, content_type: str = 'auto'):
        """Envía un mensaje formateado consistentemente"""
        formatted_content = await self.format_content(content, content_type)
        full_content = f"**{title}**\n{formatted_content}"
        
        # Almacenar en historial para referencia
        self.message_history.append(full_content)
        
        await cl.Message(
            content=full_content,
            author="System" if title == "Pensamiento" else "Database"
        ).send()

    async def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            await self.send_formatted_message(
                title="Pensamiento", 
                content=prompt,
                content_type='thought'
            )

    async def on_llm_end(self, response, **kwargs):
        if response.generations:
            for generation in response.generations[0]:
                await self.send_formatted_message(
                    title="Razonamiento",
                    content=generation.text,
                    content_type='thought'
                )

    async def on_agent_action(self, action, **kwargs):
        if action.log:
            await self.send_formatted_message(
                title="Acción",
                content=action.log,
                content_type='action'
            )

    async def on_tool_start(self, serialized, input_str, **kwargs):
        await self.send_formatted_message(
            title="Consulta SQL",
            content=input_str,
            content_type='sql'
        )

    async def on_tool_end(self, output, **kwargs):
        if output:
            await self.send_formatted_message(
                title="Resultado",
                content=str(output),
                content_type='json' if isinstance(output, (dict, list)) else 'auto'
            )

# Configuración inicial
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
    available_tables = set(db.get_usable_table_names())
    logging.info(f"Tablas disponibles: {available_tables}")
except Exception as e:
    logging.error(f"Error en DB: {e}")
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
            callbacks=[UnifiedFormatCallbackHandler()],
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": True
            }
        )

        cl.user_session.set("agent", agent)
        await cl.Message(
            content="✅ Asistente listo. Puedes preguntar sobre: "
                   f"{', '.join(available_tables)}",
            author="Assistant"
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error inicial: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(content="❌ Error: Agente no configurado").send()
        return

    try:
        processing_msg = await cl.Message(content="⏳ Procesando...").send()
        
        response = await agent.ainvoke(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        )
        
        await processing_msg.remove()
        
        if "output" in response:
            # Crear handler local para formatear la respuesta final
            formatter = UnifiedFormatCallbackHandler()
            await formatter.send_formatted_message(
                title="Respuesta Final",
                content=response["output"],
                content_type='auto'
            )
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logging.error(error_msg)
        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)