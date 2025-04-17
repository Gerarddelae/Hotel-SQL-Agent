import chainlit as cl
from openai import OpenAI
import time
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.schema.runnable.config import RunnableConfig


client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "lmstudio-community/qwen2.5-7b-instruct"

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



# Initialize SQL Database
try:
    db = SQLDatabase.from_uri("sqlite:///hotel.db")
    available_tables = set(db.get_usable_table_names())
    
    # Get schema information for each table
    schema_info = []
    for table in available_tables:
        table_schema = db.run(f"PRAGMA table_info({table})")
        columns = [f"{col[1]} ({col[2]})" for col in eval(table_schema)]
        schema_info.append(f"{table}:\n  - " + "\n  - ".join(columns))
    
    table_info = "\n\n".join(schema_info)
    
except Exception as e:
    print(f"Error connecting to database: {e}")
    db = None
    table_info = "No database connection"

# Modify the SQL_TOOL description to include detailed schema info
SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": f"""Execute SQL queries on the hotel database. 
Available tables with their columns:\n\n{table_info}""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute"
                }
            },
            "required": ["query"]
        }
    }
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time, only if asked",
        "parameters": {"type": "object", "properties": {}},
    },
}

def get_current_time():
    return {"time": time.strftime("%H:%M:%S")}

def query_database(query: str):
    if not db:
        return {"error": "Database connection not available"}
    try:
        result = db.run(query)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@cl.on_chat_start
async def start_chat():
    if not db:
        await cl.Message(content="❌ Database connection error").send()
        return

    # Define the command
    await cl.context.emitter.set_commands([
        {
            "id": "consulta",
            "icon": "database",
            "description": "Realizar consulta SQL"
        }
    ])

    try:
        # Initialize memory once (removed duplicate initialization)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        cl.user_session.set("memory", memory)
        
        # Initialize SQL agent with better configuration
        sql_agent = create_sql_agent(
            llm=ChatOpenAI(
                openai_api_base="http://127.0.0.1:1234/v1",
                openai_api_key="lm-studio",
                model_name="lmstudio-community/qwen2.5-7b-instruct",
                temperature=0.1,
                streaming=True
            ),
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            callbacks=[AgentCallbackHandler()],
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": True
            }
        )
        cl.user_session.set("sql_agent", sql_agent)
        await cl.Message(content="✅ SQL Assistant initialized and ready").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.on_message 
async def main(message: cl.Message):
    # Get or initialize memory
    memory = cl.user_session.get("memory")
    if not memory:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        cl.user_session.set("memory", memory)

    # Load previous messages from memory and ensure they're properly formatted
    history = memory.load_memory_variables({})
    messages = [msg for msg in history.get("chat_history", []) if isinstance(msg, dict) and "role" in msg]
    
    # Add new user message
    messages.append({"role": "user", "content": message.content})

    # Check if message is a SQL command
    if message.command == "consulta":
        sql_agent = cl.user_session.get("sql_agent")
        if sql_agent:
            msg = cl.Message(content="🔍 Processing your database request...")
            await msg.send()
            try:
                response = await sql_agent.ainvoke({"input": message.content},config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]))
                msg.content = response['output']
                await msg.update()
                memory.save_context(
                    {"input": message.content},
                    {"output": response['output']}
                )
                return
            except Exception as e:
                msg.content = f"❌ SQL Error: {str(e)}"
                await msg.update()
                return

    # Check if tools are available
    tools = []
    if TIME_TOOL:
        tools.append(TIME_TOOL)
    if SQL_TOOL:
        tools.append(SQL_TOOL)
    
    # Get response from OpenAI with full message history
    msg = cl.Message(content="")
    await msg.send()
    
    # Create a regular synchronous stream first
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools if tools else None,  # Only include tools if available
        tool_choice="auto",
        temperature=0.2,
        stream=True
    )
    
    # Process streaming response
    content = ""
    final_response = None
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
            await msg.stream_token(chunk.choices[0].delta.content)
        final_response = chunk
    
    await msg.update()
    
    # Handle tool calls
    if final_response and hasattr(final_response.choices[0].delta, 'tool_calls') and final_response.choices[0].delta.tool_calls:
        # Mostrar mensaje inicial sobre uso de herramientas
        tool_use_msg = cl.Message(content="🔧 El asistente está utilizando herramientas...")
        await tool_use_msg.send()
        
        tool_messages = []
        for tool_call in final_response.choices[0].delta.tool_calls:
            if tool_call.function.name == "query_database":
                # Mostrar mensaje específico para SQL
                sql_tool_msg = cl.Message(content="📡 Conectando con la base de datos...")
                await sql_tool_msg.send()
                
                try:
                    import json
                    query = json.loads(tool_call.function.arguments)["query"]
                    
                    # Mostrar la consulta SQL con formato
                    await cl.Message(
                        content=f"**Consulta SQL generada:**\n```sql\n{query}\n```"
                    ).send()
                    
                    # Mostrar mensaje de procesamiento
                    processing_msg = cl.Message(content="🔄 Ejecutando consulta...")
                    await processing_msg.send()
                    
                    result = query_database(query)
                    await processing_msg.remove()
                    
                    # Mostrar resultados con formato
                    await cl.Message(
                        content=f"**Resultados obtenidos:**\n```json\n{result}\n```"
                    ).send()
                    
                    tool_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "query_database",
                        "content": str(result)
                    })
                    
                except Exception as e:
                    await cl.Message(content=f"❌ Error al ejecutar consulta: {str(e)}").send()
                    continue
        
        if tool_messages:
            messages.extend(tool_messages)
            
            # Obtener respuesta final con streaming visible
            final_msg = cl.Message(content="🔁 Generando respuesta final...")
            await final_msg.send()
            
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True
                )
                
                final_msg.content = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        final_msg.content += chunk.choices[0].delta.content
                        await final_msg.update()
                
                # Guardar en memoria
                if final_msg.content:
                    memory.save_context(
                        {"input": message.content},
                        {"output": final_msg.content}
                    )
                    
            except Exception as e:
                await cl.Message(content=f"❌ Error al obtener respuesta final: {str(e)}").send()

def format_as_markdown_list(text: str) -> str:
    """Convert text to markdown list format"""
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip():  # Skip empty lines
            formatted_lines.append(f"- {line.strip()}")
    return '\n'.join(formatted_lines)
