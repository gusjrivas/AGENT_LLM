# visualize_agent.py
"""
Script para generar una visualización del grafo del agente LangGraph definido en app.py.
Usa herramientas dummy para mostrar la estructura general del flujo.
"""

import os
import re
import sys
from dotenv import load_dotenv

# --- Asegurarse de que se pueda importar desde app.py ---
# Añadir el directorio actual al path por si acaso
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import Agent # Importar la clase Agent
except ImportError as e:
    print(f"Error: No se pudo importar la clase 'Agent' desde 'app.py'.")
    print(f"Asegúrate de que 'visualize_agent.py' esté en el mismo directorio que 'app.py'.")
    print(f"Detalle del error: {e}")
    sys.exit(1)

# --- Importaciones Necesarias para Instanciar el Agente ---
from langchain_openai import ChatOpenAI
from langchain.tools import Tool # Para crear herramientas dummy

# --- Configuración (similar a app.py) ---
load_dotenv() # Cargar .env si existe
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: La variable de entorno 'OPENAI_API_KEY' no está configurada.")
    print("Por favor, configúrala para poder instanciar el modelo LLM.")
    sys.exit(1)

# --- Función de Sanitización (COPIADA de app.py para consistencia) ---
# Es importante usar la MISMA lógica para generar nombres de herramientas válidos.
def sanitize_tool_name(person_name: str, db_suffix: str) -> str:
    """Sanitiza un nombre para usarlo como nombre de herramienta OpenAI."""
    tool_name = f"tool_{db_suffix}_default" # Valor inicial
    try:
        tool_base = f"cv_{person_name}"
        tool_base = re.sub(r'[\s.]+', '_', tool_base.lower())
        tool_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', tool_base)
        tool_sanitized = re.sub(r'[_]+', '_', tool_sanitized)
        tool_sanitized = re.sub(r'[-]+', '-', tool_sanitized)
        tool_sanitized = tool_sanitized.strip('_-')
        tool_sanitized = tool_sanitized[:64]
        if not tool_sanitized or not re.match(r'^[a-zA-Z0-9_-]+$', tool_sanitized):
             tool_name = f"buscar_cv_{db_suffix}_fallback_{os.urandom(2).hex()}"
        else:
             tool_name = tool_sanitized
        # Validar final
        if not re.match(r'^[a-zA-Z0-9_-]+$', tool_name):
             tool_name = f"tool_cv_lookup_{db_suffix}_{os.urandom(4).hex()}"
        return tool_name
    except Exception:
        return tool_name # Devolver el default si algo falla

# --- Crear Instancia del LLM ---
try:
    dummy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
except Exception as e:
    print(f"Error al inicializar ChatOpenAI: {e}")
    sys.exit(1)

# --- Crear Herramientas Dummy ---
# Nombres de ejemplo para las personas
person1_dummy_name = "Juan Ejemplo"
person2_dummy_name = "Ana Prueba"

# Generar nombres de herramienta sanitizados usando la misma lógica
tool1_name = sanitize_tool_name(person1_dummy_name, "cv1")
tool2_name = sanitize_tool_name(person2_dummy_name, "cv2")

# Funciones placeholder para las herramientas
def dummy_search_cv1(query: str): return f"Resultado dummy para {person1_dummy_name} sobre: {query}"
def dummy_search_cv2(query: str): return f"Resultado dummy para {person2_dummy_name} sobre: {query}"

# Crear los objetos Tool
tool1 = Tool(
    name=tool1_name,
    func=dummy_search_cv1,
    description=f"Busca info en CV de {person1_dummy_name}. SOLO para preguntas sobre {person1_dummy_name}."
)
tool2 = Tool(
    name=tool2_name,
    func=dummy_search_cv2,
    description=f"Busca info en CV de {person2_dummy_name}. SOLO para preguntas sobre {person2_dummy_name}."
)
dummy_tools = [tool1, tool2]

# --- Crear Prompt del Sistema Dummy ---
# Usar los nombres de herramienta sanitizados
dummy_system_prompt = f"""
Eres un asistente de CVs. Tienes herramientas:
- `{tool1_name}` para {person1_dummy_name}
- `{tool2_name}` para {person2_dummy_name}
Usa la herramienta correcta según la pregunta o ambas si es general.
"""

# --- Instanciar el Agente ---
print("Instanciando el agente con herramientas dummy...")
try:
    agent_instance = Agent(dummy_llm, dummy_tools, system=dummy_system_prompt)
    print("Agente instanciado correctamente.")
except Exception as e:
    print(f"Error al instanciar la clase Agent: {e}")
    print("Asegúrate de que la clase Agent en app.py no tenga errores y pueda ser inicializada.")
    traceback.print_exc()
    sys.exit(1)

# --- Generar y Guardar el Gráfico ---
print("Generando gráfico del agente...")
output_filename = "agent_flow_graph.png"
try:
    # Obtener los datos PNG del grafo compilado
    png_data = agent_instance.graph.get_graph().draw_png()

    # Guardar los datos en un archivo
    with open(output_filename, "wb") as f:
        f.write(png_data)
    print(f"¡Gráfico del agente guardado exitosamente como '{output_filename}'!")
    print("El gráfico muestra los nodos 'llm' y 'action' y el flujo condicional entre ellos.")

except ImportError:
    print("\n--- Error: Dependencia Faltante ---")
    print("La librería 'pygraphviz' es necesaria para generar el gráfico.")
    print("Instálala con: pip install pygraphviz")
except FileNotFoundError:
    print("\n--- Error: Dependencia del Sistema Faltante ---")
    print("El software 'Graphviz' (dot) no se encontró en el PATH del sistema.")
    print("Asegúrate de haber instalado Graphviz correctamente para tu sistema operativo.")
    print("Consulta: https://graphviz.org/download/")
except Exception as e:
    print(f"\n--- Error Inesperado al Generar el Gráfico ---")
    print(f"Ocurrió un error: {e}")
    traceback.print_exc()