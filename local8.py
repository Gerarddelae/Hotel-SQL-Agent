import os
os.environ["LANGCHAIN_TRACING"] = "false"
import logging
import re

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
        self.step_counter = 0
        self.processing_msg = None

    def _get_step_title(self, step_type):
        self.step_counter += 1
        icons = {
            "pensamiento": "🧠",
            "razonamiento": "💭",
            "acción": "🤔",
            "herramienta": "🔧",
            "resultado": "📊"
        }
        return f"Paso {self.step_counter}: {icons.get(step_type, '')} {step_type.title()}"

    def _format_content(self, content):
        """Formatea el contenido para mejorar la visualización de listas y tablas"""
        # Detectar y formatear listas simples
        if isinstance(content, str):
            # Formatear listas con viñetas
            if "\n- " in content or "\n* " in content:
                content = re.sub(r'(?<=\n)([-*])(?=\s)', r'•', content)
            
            # Formatear listas numeradas
            content = re.sub(r'(?<=\n)(\d+)\.(?=\s)', r'\1.', content)
            
            # Detectar y formatear tablas básicas
            if "|" in content and "-|-" in content:
                lines = content.split('\n')
                if len(lines) > 2 and all("|" in line for line in lines[:3]):
                    # Mejorar el formato de la tabla
                    content = "\n".join(
                        [f"| {line.strip().replace('|',' | ').strip(' |')} |" 
                         for line in lines]
                    )
        
        return content

    async def on_llm_start(self, serialized, prompts, **kwargs):
        thought = {
            "type": "pensamiento",
            "title": self._get_step_title("pensamiento"),
            "content": "Analizando la consulta..."
        }
        self.thoughts.append(thought)
        await self._update_processing_message()

    async def on_llm_end(self, response, **kwargs):
        content = self._format_content(response.generations[0][0].text)
        self.thoughts.append({
            "type": "razonamiento",
            "title": self._get_step_title("razonamiento"),
            "content": content
        })
        await self._update_processing_message()

    async def on_agent_action(self, action, **kwargs):
        content = self._format_content(action.log)
        self.current_step = {
            "type": "acción",
            "title": self._get_step_title("acción"),
            "content": content
        }
        self.thoughts.append(self.current_step)
        await self._update_processing_message()

    async def on_tool_start(self, serialized, input_str, **kwargs):
        content = f"```sql\n{input_str}\n```"
        self.current_step = {
            "type": "herramienta",
            "title": self._get_step_title("herramienta"),
            "name": serialized["name"],
            "content": content
        }
        self.thoughts.append(self.current_step)
        await self._update_processing_message()

    async def on_tool_end(self, output, **kwargs):
        if self.current_step:
            content = self._format_content(output)
            result_step = {
                "type": "resultado",
                "title": self._get_step_title("resultado"),
                "content": f"```json\n{content}\n```",
                "parent_step": self.current_step["title"]
            }
            self.thoughts.append(result_step)
            await self._update_processing_message()

    async def _update_processing_message(self):
        if not self.processing_msg:
            return

        process_content = """
## 🔍 Proceso de Ejecución

<style>
.details-container {
    margin-bottom: 12px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.details-summary {
    padding: 10px 15px;
    cursor: pointer;
    font-weight: bold;
    background-color: var(--details-summary-bg);
    color: var(--details-summary-text);
    border: 1px solid var(--details-border);
    border-radius: 8px;
    transition: background-color 0.2s;
}
.details-summary:hover {
    background-color: var(--details-summary-hover);
}
.details-content {
    padding: 12px 15px;
    background-color: var(--details-content-bg);
    color: var(--details-content-text);
    border-left: 1px solid var(--details-border);
    border-right: 1px solid var(--details-border);
    border-bottom: 1px solid var(--details-border);
    border-radius: 0 0 8px 8px;
}
pre {
    background-color: var(--code-bg) !important;
    border-radius: 6px;
    padding: 10px !important;
    border: 1px solid var(--code-border);
}
/* Estilos para listas y tablas */
.markdown-list {
    padding-left: 20px;
    margin: 8px 0;
}
.markdown-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}
.markdown-table th, .markdown-table td {
    padding: 8px 12px;
    border: 1px solid var(--border-color);
}
.markdown-table th {
    background-color: var(--table-header-bg);
}
</style>
"""
        
        for step in self.thoughts:
            step_content = step["content"]
            
            # Aplicar formato adicional a listas y tablas
            if isinstance(step_content, str):
                # Convertir listas simples a HTML
                if "\n• " in step_content or "\n1." in step_content:
                    step_content = re.sub(
                        r'(?<=\n)(•|\d+\.)\s+(.*)',
                        r'<li>\2</li>',
                        step_content
                    )
                    step_content = step_content.replace("<li>", '<div class="markdown-list"><li>', 1)
                    step_content += "</div>"
                
                # Mejorar formato de tablas
                if "|" in step_content and "-|-" in step_content:
                    step_content = step_content.replace("|-|-", "|--|--")
                    step_content = f'<div class="markdown-table">{step_content}</div>'
            
            process_content += f"""
<div class="details-container">
<details>
<summary class="details-summary">{step['title']}</summary>
<div class="details-content">
{step_content}
</div>
</details>
</div>
"""

        # Asignar el contenido y actualizar
        self.processing_msg.content = process_content.strip()
        await self.processing_msg.update()


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

    try:
        # Inicializar el mensaje de procesamiento
        callback_handler.processing_msg = cl.Message(content="Procesando tu consulta...")
        await callback_handler.processing_msg.send()
        
        # Ejecutar el agente
        response = await agent.ainvoke(
            {"input": message.content},
            config=RunnableConfig(callbacks=[callback_handler]),
        )

        # Formatear la respuesta final
        final_output = response.get('output', 'No se recibió respuesta')
        
        


        # Enviar la respuesta final con formato
        final_response = f"""
## ✅ Respuesta Final

{final_output}

<small style="opacity: 0.7;">Consulta procesada correctamente</small>
"""
        await cl.Message(content=final_response).send()

    except Exception as e:
        logger.error(f"Error en on_message: {str(e)}")
        if callback_handler.processing_msg:
            await callback_handler.processing_msg.update(content=f"❌ Error: {str(e)}")
        else:
            await cl.Message(content=f"❌ Error: {str(e)}").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)