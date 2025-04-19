# -*- coding: utf-8 -*-
"""
app.py: Solución RAG genérica con LangGraph y Streamlit
para comparar dos CVs cualesquiera.
Busca API Key en variables de entorno del SO y sanitiza nombres para ChromaDB.
Versión con corrección para error "Unsupported function".
"""

# --- 1. Importaciones Necesarias ---
import os
import re # Importar el módulo de expresiones regulares
import operator
import tempfile
from typing import TypedDict, Annotated, List, Tuple, Optional, Dict

import streamlit as st
# from dotenv import load_dotenv # Opcional si usas .env en desarrollo

# LangChain/LangGraph imports
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END

import pprint
import traceback # Para imprimir errores detallados

# --- 2. Configuración Inicial ---
# load_dotenv() # Descomentar si usas .env localmente

# --- BUSCAR API KEY EN VARIABLES DE ENTORNO DEL SISTEMA OPERATIVO ---
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(
        "La variable de entorno 'OPENAI_API_KEY' no está configurada o no es accesible. "
        "Por favor, asegúrate de que esté definida en tu sistema operativo o en los secretos de Streamlit."
    )
    st.stop()

# --- Modelos (Inicializarlos usando la clave obtenida) ---
try:
    # Usar st.cache_resource para evitar reinicializar en cada rerun si es posible
    # Nota: Puede dar problemas con objetos complejos no serializables. Probar con cuidado.
    # @st.cache_resource
    # def get_llm(api_key):
    #     return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    # @st.cache_resource
    # def get_embeddings_model(api_key):
    #     return OpenAIEmbeddings(api_key=api_key)
    # llm = get_llm(openai_api_key)
    # embeddings_model = get_embeddings_model(openai_api_key)

    # Por ahora, inicialización directa (más simple, puede ser menos eficiente en reruns)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
    print("Modelos de OpenAI inicializados correctamente.")
except Exception as e:
    st.error(f"Error al inicializar los modelos de OpenAI con la API key proporcionada: {e}")
    st.stop()


# --- 3. Funciones de Procesamiento de CVs y Extracción de Nombres ---

def load_and_split_cv(file_path: str) -> Tuple[Optional[List], Optional[str]]:
    """Carga un CV (PDF), lo divide y devuelve los chunks y una muestra de texto."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents:
            print(f"Advertencia: No se pudo cargar contenido de {file_path}")
            return None, None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        sample_text = " ".join([doc.page_content for doc in documents[:2]])[:2000]
        print(f"Cargado y dividido '{os.path.basename(file_path)}' en {len(chunks)} fragmentos.")
        return chunks, sample_text
    except Exception as e:
        print(f"Error cargando o dividiendo {file_path}: {e}")
        st.warning(f"Error procesando {os.path.basename(file_path)}: {e}. Verifique el archivo.")
        return None, None

def extract_name_from_cv(cv_sample_text: str, file_name: str) -> str:
    """Usa un LLM para extraer el nombre completo del propietario del CV."""
    default_name = f"Persona_{os.path.basename(file_name).split('.')[0]}"
    if not cv_sample_text:
        return default_name

    prompt = f"""
    Eres un asistente de RRHH experto en analizar CVs.
    Texto inicial de un CV del archivo '{file_name}':
    ---
    {cv_sample_text}
    ---
    Extrae el NOMBRE COMPLETO de la persona a la que pertenece este CV. Considera el nombre más prominente.
    Devuelve ÚNICAMENTE el nombre completo. Si no puedes determinarlo con confianza, devuelve 'Nombre Desconocido {file_name}'.
    Nombre completo extraído:
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        extracted_name = response.content.strip()
        extracted_name = re.sub(r"^\W+|\W+$", "", extracted_name) # Limpieza básica
        if not extracted_name or "Nombre Desconocido" in extracted_name or len(extracted_name) > 50:
             print(f"Advertencia: No se pudo extraer un nombre claro de {file_name}. Usando nombre por defecto.")
             st.info(f"No se pudo extraer un nombre claro de {file_name}, se usará '{default_name}'.")
             return default_name
        print(f"Nombre extraído de '{file_name}': {extracted_name}")
        return extracted_name
    except Exception as e:
        print(f"Error al extraer nombre de {file_name} usando LLM: {e}")
        st.warning(f"Error al intentar extraer el nombre de {file_name}. Se usará '{default_name}'.")
        return default_name

def create_cv_tools(chunks: List, person_name: str, file_name: str, db_suffix: str) -> Optional[Dict]:
    """Crea el vector store y la herramienta retriever para un CV."""
    if not chunks:
        return None
    collection_name = "default_collection_name" # Valor inicial por si falla antes
    tool_name = f"buscar_cv_{db_suffix}_default_tool" # Valor inicial
    try:
        # --- Sanitización del nombre para la colección ---
        base_name_part_coll = person_name.lower().replace(' ', '_')
        sanitized_name_part_coll = re.sub(r'[^a-z0-9_-]', '', base_name_part_coll)
        sanitized_name_part_coll = sanitized_name_part_coll[:20] if sanitized_name_part_coll else "default"
        collection_name = f"cv_{db_suffix}_{sanitized_name_part_coll}_{os.urandom(4).hex()}"
        print(f"DEBUG: Nombre de colección generado: {collection_name}")

        # --- Sanitización del nombre para la herramienta ---
        base_name_part_tool = person_name.lower().replace(' ', '_').replace('.', '_')
        sanitized_name_part_tool = re.sub(r'\W|^(?=\d)', '_', base_name_part_tool)
        sanitized_name_part_tool = re.sub(r'_+', '_', sanitized_name_part_tool)
        sanitized_name_part_tool = sanitized_name_part_tool.strip('_')
        tool_name = f"buscar_cv_{sanitized_name_part_tool}" if sanitized_name_part_tool else f"buscar_cv_{db_suffix}_default"
        print(f"DEBUG: Nombre de herramienta generado: {tool_name}")

        # Añadir metadatos
        for chunk in chunks:
            chunk.metadata["person"] = person_name
            chunk.metadata["source"] = file_name

        # Crear VectorStore
        vector_store = Chroma.from_documents(
            chunks,
            embeddings_model,
            collection_name=collection_name
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Crear Herramienta
        tool_description = f"Busca información EXCLUSIVAMENTE en el CV de {person_name} (archivo: {file_name}). Úsala SOLO para preguntas específicas sobre {person_name}."
        cv_tool = create_retriever_tool(
            retriever,
            name=tool_name,
            description=tool_description,
        )
        print(f"VectorStore y Herramienta '{tool_name}' creados para {person_name}.")
        return {"tool": cv_tool, "retriever": retriever, "vector_store": vector_store, "tool_name": tool_name}

    except Exception as e:
        print(f"Error creando VectorStore/Herramienta para {person_name} ({file_name}). Nombre de colección intentado: {collection_name}. Nombre de herramienta intentado: {tool_name}. Error: {e}")
        st.error(f"Error crítico al crear recursos para {person_name}: {e}")
        return None

# --- 4. Función de Configuración Streamlit ---
def setup_cv_environment_streamlit(uploaded_file1, uploaded_file2) -> Tuple[Optional[str], Optional[str], List, Optional[str]]:
    """Orquesta la carga, extracción de nombre y creación de herramientas para dos CVs subidos."""
    print("--- Iniciando Configuración del Entorno de CVs (Streamlit) ---")
    all_tools = []
    person1_name, person2_name = None, None
    tool_name1, tool_name2 = None, None
    file_name1, file_name2 = uploaded_file1.name, uploaded_file2.name

    # Guardar archivos temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp1.write(uploaded_file1.getvalue())
        cv_path1 = tmp1.name
        tmp2.write(uploaded_file2.getvalue())
        cv_path2 = tmp2.name

    try:
        # Procesar CV 1
        chunks1, sample1 = load_and_split_cv(cv_path1)
        if chunks1:
            person1_name = extract_name_from_cv(sample1, file_name1)
            tool_info1 = create_cv_tools(chunks1, person1_name, file_name1, "cv1")
            if tool_info1:
                all_tools.append(tool_info1["tool"])
                tool_name1 = tool_info1["tool_name"]

        # Procesar CV 2
        chunks2, sample2 = load_and_split_cv(cv_path2)
        if chunks2:
            person2_name = extract_name_from_cv(sample2, file_name2)
            tool_info2 = create_cv_tools(chunks2, person2_name, file_name2, "cv2")
            if tool_info2:
                all_tools.append(tool_info2["tool"])
                tool_name2 = tool_info2["tool_name"]

    finally:
        # Limpiar archivos temporales
        if 'cv_path1' in locals() and os.path.exists(cv_path1): os.remove(cv_path1)
        if 'cv_path2' in locals() and os.path.exists(cv_path2): os.remove(cv_path2)

    # Verificar si TODO se procesó correctamente
    if not all_tools or not person1_name or not person2_name or not tool_name1 or not tool_name2 or len(all_tools) != 2:
        st.error("Falló la configuración. No se pudieron procesar ambos CVs o crear las herramientas necesarias. Verifique los archivos y los logs.")
        return None, None, [], None

    # Crear el Prompt del Sistema Dinámicamente
    system_prompt = f"""
Eres un asistente experto en comparar Curriculums Vitae (CVs). Tienes acceso a dos CVs:
1. El CV de {person1_name} (archivo: {file_name1}) al que accedes con la herramienta `{tool_name1}`.
2. El CV de {person2_name} (archivo: {file_name2}) al que accedes con la herramienta `{tool_name2}`.

Tu tarea es responder preguntas basándote ÚNICAMENTE en la información contenida en estos dos CVs.

Instrucciones IMPORTANTES para usar las herramientas:
- Analiza CUIDADOSAMENTE la pregunta del usuario para identificar si se refiere a una persona específica ({person1_name} o {person2_name}) o a ambos. Usa variantes del nombre si es necesario (ej. solo nombre de pila).
- **Si la pregunta menciona CLARAMENTE a '{person1_name}'** (o variantes como su nombre de pila si es distinguible) y NO a {person2_name}, usa EXCLUSIVAMENTE la herramienta `{tool_name1}`.
- **Si la pregunta menciona CLARAMENTE a '{person2_name}'** (o variantes) y NO a {person1_name}, usa EXCLUSIVAMENTE la herramienta `{tool_name2}`.
- **Si la pregunta es GENERAL, NO menciona un nombre específico, o pregunta por 'ambos'/'quién'/'alguno'**, DEBES USAR AMBAS herramientas: `{tool_name1}` Y `{tool_name2}`. Ejecútalas para obtener la información completa.
- **Si tienes dudas sobre a quién se refiere la pregunta, es MÁS SEGURO usar AMBAS herramientas.**

Instrucciones para la RESPUESTA FINAL:
- Sintetiza la respuesta final basándote ESTRICTAMENTE en los resultados obtenidos de las herramientas.
- Si consultaste ambos CVs, indica claramente qué información corresponde a {person1_name} y cuál a {person2_name}.
- Si una herramienta no devolvió información relevante, menciónalo si es pertinente.
- NO INVENTES INFORMACIÓN. Si la respuesta no está en los CVs consultados, indica claramente que no encontraste esa información.
- Sé objetivo y cíñete a los datos de los CVs.

Herramientas disponibles: {tool_name1}, {tool_name2}
"""
    print(f"--- Configuración Streamlit Completa. Nombres: '{person1_name}', '{person2_name}'. Herramientas: {[t.name for t in all_tools]} ---")
    return person1_name, person2_name, all_tools, system_prompt


# --- 5. Definición del Estado y Clase Agente (Reutilizados) ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class Agent:
    # La inicialización ya incluye el model.bind_tools(tools) crucial
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        # Vincular el modelo a las herramientas UNA VEZ aquí
        self.model = model.bind_tools(tools)
        print(f"Agente LangGraph inicializado con {len(tools)} herramientas y modelo {self.model.model_name}.")
        print(f"Herramientas disponibles para el agente: {list(self.tools.keys())}")


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        is_ai_message = isinstance(result, AIMessage)
        has_tool_calls = hasattr(result, "tool_calls") and result.tool_calls and len(result.tool_calls) > 0
        return is_ai_message and has_tool_calls

    def call_openai(self, state: AgentState):
        messages = state['messages']
        # Asegurarse de que el mensaje del sistema esté al principio y sea el correcto
        if not messages or not isinstance(messages[0], SystemMessage):
             messages = [SystemMessage(content=self.system)] + (messages if messages else [])
        elif messages[0].content != self.system:
             # Si el system prompt ha cambiado (ej, nuevos CVs), actualizarlo
             print("DEBUG: Actualizando mensaje del sistema en call_openai.")
             messages[0] = SystemMessage(content=self.system)

        # Usar el modelo ya vinculado en __init__
        message = self.model.invoke(messages)
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"DEBUG: LLM solicitó llamadas a herramientas: {[t.get('name', 'N/A') for t in message.tool_calls]}")
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
             print("ERROR: Se esperaba una AIMessage con tool_calls en take_action.")
             # Devolver un ToolMessage de error para que el LLM lo vea
             return {'messages': [ToolMessage(tool_call_id="error_state", name="error_state", content="Error interno: estado inesperado antes de la acción.")]}

        tool_calls = last_message.tool_calls
        results = []
        print(f"\nDEBUG: Ejecutando {len(tool_calls)} acciones solicitadas por el LLM...")
        for t in tool_calls:
            tool_name = t.get('name')
            tool_args = t.get('args', {})
            tool_id = t.get('id')

            if not tool_name or not tool_id:
                print(f"ADVERTENCIA: Llamada a herramienta inválida recibida del LLM: {t}")
                results.append(ToolMessage(tool_call_id=tool_id or "invalid_id", name=tool_name or "invalid_name", content="Error: Llamada a herramienta inválida del LLM."))
                continue

            if tool_name not in self.tools:
                print(f"ADVERTENCIA: LLM intentó llamar a una herramienta inexistente: '{tool_name}'. Herramientas disponibles: {list(self.tools.keys())}")
                results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=f"Error: La herramienta '{tool_name}' no existe. Usa solo: {list(self.tools.keys())}"))
            else:
                try:
                    # La query suele estar en args['query'] o args directamente si es un string
                    query_input = tool_args.get('query', tool_args if isinstance(tool_args, str) else str(tool_args))
                    if not isinstance(query_input, str): query_input = str(query_input) if query_input else "Consulta general CV"

                    print(f"DEBUG: Ejecutando herramienta '{tool_name}' con input: '{query_input[:50]}...'")
                    result = self.tools[tool_name].invoke(query_input) # Invocar la herramienta correcta

                    # Formatear resultado (lista de Documentos a string)
                    if isinstance(result, list) and all(hasattr(doc, 'page_content') for doc in result):
                         result_str = "\n\n".join([f"-- INICIO FRAGMENTO CV ({tool_name}) --\n{doc.page_content}\n-- FIN FRAGMENTO CV --" for doc in result])
                    else: result_str = str(result) # Fallback

                    # Devolver el resultado como ToolMessage
                    results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=result_str))

                except Exception as e:
                    print(f"ERROR: Falló la ejecución de la herramienta '{tool_name}' con args {tool_args}: {e}")
                    traceback.print_exc() # Imprimir stack trace en consola
                    st.error(f"Error ejecutando la herramienta {tool_name}: {e}") # Mostrar error en UI
                    # Devolver mensaje de error como resultado de la herramienta
                    results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=f"Error al ejecutar la herramienta '{tool_name}': {traceback.format_exc()}"))

        print(f"DEBUG: {len(results)} acciones ejecutadas. Volviendo al LLM.")
        return {'messages': results}


# --- 6. Interfaz de Streamlit ---
st.set_page_config(page_title="Comparador Inteligente de CVs", layout="wide")
st.title("📄🤖 Comparador Inteligente de CVs con Agentes LangGraph")
st.caption("Sube dos CVs en formato PDF para compararlos o hacer preguntas específicas.")

# --- Inicializar Session State ---
# Usar un diccionario para agrupar el estado del agente
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "setup_done": False,
        "agent_executor": None,
        "person1_name": "Persona 1",
        "person2_name": "Persona 2",
        "system_prompt": ""
    }
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [] # Historial solo para UI

# --- Sección de Carga de Archivos ---
with st.sidebar:
    st.header("Carga de Archivos")
    uploaded_file1 = st.file_uploader("Cargar CV 1 (PDF)", type="pdf", key="file1")
    uploaded_file2 = st.file_uploader("Cargar CV 2 (PDF)", type="pdf", key="file2")

    if uploaded_file1 and uploaded_file2:
        if st.button("🚀 Procesar CVs y Configurar Agente", key="process_btn"):
            with st.spinner("Analizando CVs, extrayendo nombres y preparando el agente... por favor espera."):
                # Limpiar estado anterior ANTES de procesar
                st.session_state.chat_messages = []
                st.session_state.agent_state = {
                    "setup_done": False, "agent_executor": None,
                    "person1_name": "Persona 1", "person2_name": "Persona 2",
                    "system_prompt": ""
                }

                # Ejecutar la configuración
                person1, person2, tools, sys_prompt = setup_cv_environment_streamlit(uploaded_file1, uploaded_file2)

                # Si la configuración fue exitosa, guardar estado y crear agente
                if person1 and person2 and tools and sys_prompt:
                    st.session_state.agent_state["person1_name"] = person1
                    st.session_state.agent_state["person2_name"] = person2
                    st.session_state.agent_state["system_prompt"] = sys_prompt

                    try:
                         # Crear e instanciar el agente AHORA con todo lo necesario
                         agent_instance = Agent(llm, tools, system=sys_prompt)
                         # Guardar la instancia COMPLETA del agente en el estado
                         st.session_state.agent_state["agent_executor"] = agent_instance
                         st.session_state.agent_state["setup_done"] = True
                         # Mensaje inicial para el chat de la UI
                         st.session_state.chat_messages = [
                             AIMessage(content=f"¡Listo! Puedes preguntar sobre los CVs de {person1} y {person2}.")
                         ]
                         st.success(f"Agente configurado para {person1} y {person2}. ¡Puedes empezar a preguntar!")
                         st.rerun() # Forzar rerun para actualizar UI principal inmediatamente
                    except Exception as e:
                         st.error(f"Error al instanciar el agente LangGraph: {e}")
                         # Asegurar que el estado refleje el fallo
                         st.session_state.agent_state["setup_done"] = False
                         st.session_state.agent_state["agent_executor"] = None
                else:
                    # El error ya se mostró en setup_cv_environment_streamlit
                    st.session_state.agent_state["setup_done"] = False
                    st.session_state.agent_state["agent_executor"] = None
    elif st.session_state.agent_state["setup_done"]:
         # Si ya está configurado, mostrar info
         st.info(f"Agente configurado para {st.session_state.agent_state['person1_name']} y {st.session_state.agent_state['person2_name']}. Puedes preguntar o cargar nuevos CVs.")
    else:
         # Estado inicial o después de un fallo
         st.info("Por favor, carga ambos archivos CV en formato PDF y presiona 'Procesar'.")

# --- Sección de Chat ---
st.header("Chat de Consulta de CVs")

if not st.session_state.agent_state["setup_done"]:
    st.warning("⚠️ Por favor, carga ambos CVs y presiona 'Procesar CVs y Configurar Agente' en la barra lateral para comenzar.")
else:
    # Mostrar historial de mensajes (de la UI)
    for message in st.session_state.chat_messages:
        role = "assistant" if isinstance(message, (AIMessage, SystemMessage)) else "user"
        if isinstance(message, SystemMessage): role="system"

        with st.chat_message(role):
             content_to_display = ""
             if hasattr(message, 'content'):
                 content_to_display = message.content
             elif isinstance(message, dict) and 'content' in message:
                 content_to_display = message.get('content', '')

             if content_to_display:
                 st.markdown(content_to_display)

    # Input del usuario
    prompt_placeholder = f"Pregunta sobre {st.session_state.agent_state['person1_name']} o {st.session_state.agent_state['person2_name']}..."
    if prompt := st.chat_input(prompt_placeholder, key="chat_input"):
        # Añadir mensaje del usuario al historial de UI
        st.session_state.chat_messages.append(HumanMessage(content=prompt))
        # Mostrar mensaje del usuario inmediatamente (se redibujará abajo)
        # st.rerun() # No es necesario aquí, se hace después de la respuesta AI

        # Ejecutar el agente si está listo
        agent_exec_instance = st.session_state.agent_state.get("agent_executor")
        if agent_exec_instance:
            with st.chat_message("user"): # Mostrar prompt mientras procesa
                 st.markdown(prompt)
            with st.spinner("Pensando y buscando en los CVs..."):
                try:
                    # Preparar input para el grafo (solo el mensaje nuevo)
                    invoke_input = {"messages": [HumanMessage(content=prompt)]}

                    # Asegurar que el agente use el system_prompt actual del estado
                    agent_exec_instance.system = st.session_state.agent_state["system_prompt"]

                    print(f"DEBUG: Invocando grafo con input: {invoke_input}")
                    print(f"DEBUG: System prompt usado: {agent_exec_instance.system[:150]}...")

                    # Invocar el grafo del agente
                    # El grafo maneja internamente el paso de mensajes y resultados de tools
                    result = agent_exec_instance.graph.invoke(invoke_input)

                    # La respuesta final debería estar en el último mensaje de la salida
                    if result and 'messages' in result and result['messages']:
                        ai_response = result['messages'][-1]
                        if isinstance(ai_response, AIMessage):
                             # Añadir respuesta AI al historial de UI
                             st.session_state.chat_messages.append(ai_response)
                        else:
                             # Si el último mensaje no es AI, algo raro pasó
                             st.error("El agente no devolvió una respuesta final clara.")
                             print("Resultado inesperado del agente:", result['messages'])
                             error_msg = AIMessage(content="Hubo un problema interno al procesar la respuesta.")
                             st.session_state.chat_messages.append(error_msg)
                    else:
                        st.error("El agente no devolvió ningún mensaje.")
                        print("Resultado vacío o inválido del agente:", result)
                        error_msg = AIMessage(content="El agente no produjo una respuesta.")
                        st.session_state.chat_messages.append(error_msg)

                    # Forzar re-run para mostrar la nueva respuesta AI en la UI
                    st.rerun()

                except Exception as e:
                    st.error(f"Error al ejecutar la consulta: {e}")
                    traceback.print_exc() # Imprimir traceback completo en consola/logs
                    # Añadir mensaje de error al historial de UI
                    error_message = AIMessage(content=f"Lo siento, ocurrió un error al procesar tu pregunta. Detalles técnicos: {e}")
                    st.session_state.chat_messages.append(error_message)
                    st.rerun() # Forzar re-run para mostrar el mensaje de error
        else:
            st.error("El agente no está configurado correctamente. Intenta procesar los CVs de nuevo.")
            # Limpiar el input para evitar reenvío accidental
            # st.session_state.chat_input = "" # No funciona directamente así con st.chat_input