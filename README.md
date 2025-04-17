# Asistente SQL para Hotel

## Descripción General

Este proyecto implementa un asistente interactivo de SQL para la gestión de bases de datos de hoteles utilizando Chainlit y LangChain. El asistente permite a los usuarios consultar una base de datos de hotel utilizando lenguaje natural, que luego se traduce en consultas SQL y se ejecuta contra una base de datos SQLite.

## Características

- **Lenguaje Natural a SQL**: Convierte preguntas del usuario en consultas SQL
- **Interfaz Interactiva**: Construida con Chainlit para una experiencia tipo chat
- **Visibilidad del Proceso de Pensamiento**: Muestra los pasos de razonamiento que toma la IA
- **Sistema de Memoria**: Mantiene el contexto de la conversación para preguntas de seguimiento
- **Soporte de LLM Local**: Utiliza LM Studio para inferencia de modelos locales
- **Conocimiento del Esquema de Base de Datos**: Comprende la estructura de la base de datos del hotel
- **Herramienta de Tiempo**: Incluye una herramienta especial para consultar la fecha y hora actual

## Componentes Técnicos

### Modelos y Herramientas

- **LLM**: Utiliza el modelo Qwen 2.5 7B Instruct a través de LM Studio
- **Base de Datos**: Base de datos SQLite (`hotel.db`)
- **Framework**: LangChain para la creación de agentes y Chainlit para la UI
- **Memoria**: Memoria de ventana de buffer de conversación para mantener el contexto
- **Herramientas Adicionales**: 
  - SQL_TOOL: Para ejecutar consultas en la base de datos
  - TIME_TOOL: Para obtener la fecha y hora actual cuando se solicita

### Archivos Clave

- **demo3.py**: Aplicación principal con agente SQL e integración de API OpenAI
- **hotel.db**: Base de datos SQLite con información del hotel

## Cómo Funciona

1. La aplicación inicializa una conexión a la base de datos del hotel
2. Crea un agente SQL utilizando el toolkit de LangChain
3. Cuando un usuario envía un mensaje, el agente:
   - Analiza la consulta
   - Formula SQL para responder la pregunta
   - Ejecuta el SQL contra la base de datos
   - Devuelve resultados formateados
4. El proceso de pensamiento del agente es capturado y mostrado al usuario
5. Adicionalmente, el agente puede proporcionar información sobre la fecha y hora actual cuando se le solicita

## Manejador de Callbacks del Agente

La clase `AgentCallbackHandler` captura el proceso de razonamiento del agente:

- Rastrea pensamientos y pasos de razonamiento
- Captura consultas SQL que se están ejecutando
- Registra resultados de la base de datos
- Formatea todo para mostrarlo en la interfaz de usuario

## Herramientas Disponibles

El asistente cuenta con dos herramientas principales:

1. **SQL_TOOL**: Permite ejecutar consultas SQL en la base de datos del hotel
2. **TIME_TOOL**: Proporciona la fecha y hora actual cuando el usuario lo solicita

## Configuración y Ejecución

### Prerrequisitos

- Python 3.8+
- LM Studio ejecutándose localmente en el puerto 1234
- Paquetes Python requeridos (ver requirements.txt)

### Instalación

```bash
pip install -r requirements.txt
```

## Esquema de la Base de Datos
La base de datos del hotel incluye tablas para:

- Huéspedes
- Habitaciones
- Reservas
- Pagos

## Ejemplos de Consultas
- "Muéstrame todas las habitaciones disponibles para mañana"
- "¿Cuál es el ingreso total del mes pasado?"
- "¿Quiénes son nuestros huéspedes más frecuentes?"
- "Lista todas las reservas con pagos pendientes"
- "¿Qué hora es actualmente?" (utilizará la herramienta TIME_TOOL)
## Mejoras Futuras
- Agregar sistema de autenticación
- Implementar operaciones de escritura en la base de datos con flujo de aprobación
- Mejorar el manejo de errores y recuperación
- Agregar visualización para resultados de consultas
- Soporte para múltiples conexiones de bases de datos
- Expandir las capacidades de la herramienta de tiempo para incluir zonas horarias
