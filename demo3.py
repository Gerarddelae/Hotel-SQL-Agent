import chainlit as cl
from openai import AsyncOpenAI
import time
import json
import traceback
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import AgentExecutor
from langchain.schema.runnable.config import RunnableConfig

async_client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
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
        "description": "Get the current time and date, only if asked",
        "parameters": {"type": "object", "properties": {}},
    },
}

def get_current_time():
    return {
        "time": time.strftime("%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d")
    }

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

    await cl.context.emitter.set_commands([
        {
            "id": "consulta",
            "icon": "database",
            "description": "Realizar consulta SQL"
        }
    ])

    try:
        # Change to ConversationBufferWindowMemory for better context retention
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 messages
        )
        cl.user_session.set("memory", memory)
        
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
        await cl.Message(content="🏨 HSPA-SQL Assistant initialized and ready").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

async def process_tool_calls(tool_calls, original_message, memory):
    tool_use_msg = cl.Message(content="🔧 El asistente está utilizando herramientas...")
    await tool_use_msg.send()
    
    tool_messages = []
    messages = []
    
    for tool_call in tool_calls:
        if tool_call["function"]["name"] == "query_database":
            debug_msg = cl.Message(content=f"🔍 Procesando tool call: {tool_call['function']['name']} (ID: {tool_call['id']})")
            await debug_msg.send()
            
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
                query = arguments.get("query", "")
                
                if not query:
                    await cl.Message(content="⚠️ No se encontró una consulta SQL válida").send()
                    continue
                
                query_msg = cl.Message(content=f"**Consulta SQL:**\n```sql\n{query}\n```")
                await query_msg.send()
                
                exec_msg = cl.Message(content="🔄 Ejecutando consulta...")
                await exec_msg.send()
                
                result = query_database(query)
                exec_msg.content = "✅ Consulta ejecutada"
                await exec_msg.update()
                
                result_str = json.dumps(result, ensure_ascii=False)
                result_msg = cl.Message(content=f"**Resultados:**\n```json\n{result_str}\n```")
                await result_msg.send()
                
                tool_messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": "query_database",
                    "content": result_str
                })
                
            except Exception as e:
                error_msg = cl.Message(content=f"❌ Error al procesar consulta: {str(e)}\n```\n{traceback.format_exc()}\n```")
                await error_msg.send()
        
        elif tool_call["function"]["name"] == "get_current_time":
            try:
                result = get_current_time()
                time_msg = cl.Message(content=f"⏰ Hora y fecha actual: {result['time']} - {result['date']}")
                await time_msg.send()
                
                tool_messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": "get_current_time",
                    "content": json.dumps(result)
                })
            except Exception as e:
                await cl.Message(content=f"❌ Error al obtener la hora y fecha: {str(e)}").send()
    
    if tool_messages:
        assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                } for tc in tool_calls
            ]
        }
        messages.append(assistant_message)
        messages.extend(tool_messages)
        
        clean_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            clean_msg = {
                'role': msg.get('role', ''),
                'content': str(msg.get('content', ''))
            }
            
            if msg['role'] == 'assistant' and 'tool_calls' in msg:
                clean_msg['tool_calls'] = [{
                    'id': tc.get('id', ''),
                    'type': 'function',
                    'function': {
                        'name': tc['function']['name'],
                        'arguments': tc['function']['arguments']
                    }
                } for tc in msg['tool_calls']]
            
            if msg['role'] == 'tool' and 'tool_call_id' in msg:
                clean_msg['tool_call_id'] = msg['tool_call_id']
                clean_msg['name'] = msg.get('name', '')
            
            clean_messages.append(clean_msg)

        final_msg = cl.Message(content="")
        await final_msg.send()
        
        try:
            final_response = await async_client.chat.completions.create(
                model=MODEL,
                messages=clean_messages,
                stream=True
            )
            
            final_content = ""
            async for chunk in final_response:
                if chunk.choices[0].delta.content:
                    final_content += chunk.choices[0].delta.content
                    await final_msg.stream_token(chunk.choices[0].delta.content)
            
            await final_msg.update()
            
            if final_content:
                # Save both input and output to memory
                memory.save_context(
                    {"input": original_message.content},
                    {"output": final_content}
                )
                # Also save the tool responses
                memory.chat_memory.add_user_message(original_message.content)
                memory.chat_memory.add_ai_message(final_content)
                
        except Exception as e:
            await cl.Message(content=f"❌ Error al generar respuesta final: {str(e)}\n```\n{traceback.format_exc()}\n```").send()

@cl.on_message 
async def main(message: cl.Message):
    memory = cl.user_session.get("memory")
    if not memory:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )
        cl.user_session.set("memory", memory)

    # Load history and ensure proper message format
    history = memory.load_memory_variables({})
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg.content} 
        if hasattr(msg, 'content') else msg
        for i, msg in enumerate(history.get("chat_history", []))
    ]
    
    # Add current message with proper role
    messages.append({"role": "user", "content": message.content})

    if message.command == "consulta":
        sql_agent = cl.user_session.get("sql_agent")
        if sql_agent:
            msg = cl.Message(content="🔍 Processing your database request...")
            await msg.send()
            try:
                response = await sql_agent.ainvoke({"input": message.content}, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]))
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

    tools = []
    if TIME_TOOL:
        tools.append(TIME_TOOL)
    if SQL_TOOL:
        tools.append(SQL_TOOL)
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        response_stream = await async_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto",
            temperature=0.2,
            stream=True
        )

        full_content = []
        tool_calls_detected = False
        current_tool_calls = []
        
        async for chunk in response_stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                full_content.append(delta.content)
                await msg.stream_token(delta.content)
            
            if delta.tool_calls:
                tool_calls_detected = True
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx >= len(current_tool_calls):
                        current_tool_calls.append({
                            "id": "",
                            "function": {"arguments": "", "name": ""}
                        })
                    
                    current_tool_calls[idx]["id"] += getattr(tool_call, "id", "") or ""
                    current_tool_calls[idx]["function"]["name"] += getattr(tool_call.function, "name", "") or ""
                    current_tool_calls[idx]["function"]["arguments"] += getattr(tool_call.function, "arguments", "") or ""

        await msg.update()

        if tool_calls_detected:
            await process_tool_calls(current_tool_calls, message, memory)
        else:
            memory.save_context(
                {"input": message.content},
                {"output": "".join(full_content)}
            )

    except Exception as e:
        await cl.Message(content=f"❌ Error general: {str(e)}\n```\n{traceback.format_exc()}\n```").send()

def format_as_markdown_list(text: str) -> str:
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip():
            formatted_lines.append(f"- {line.strip()}")
    return '\n'.join(formatted_lines)