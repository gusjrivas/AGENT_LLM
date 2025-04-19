# -*- coding: utf-8 -*-
"""
app.py: Solución RAG genérica con LangGraph y Streamlit
para comparar dos CVs cualesquiera.
REFACTORIZADO para importación y con corrección de flujo de ejecución en Streamlit.
"""

# --- 1. Importaciones Necesarias ---
import os
import re
import operator
import tempfile
from typing import TypedDict, Annotated, List, Tuple, Optional, Dict
import traceback # Para imprimir errores detallados

import streamlit as st

# LangChain/LangGraph imports
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END

# --- 2. Configuración Inicial y Modelos ---
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = None
embeddings_model = None

def initialize_models():
    """Inicializa los modelos LLM y Embeddings."""
    global llm, embeddings_model, openai_api_key
    if not openai_api_key:
        print("ERROR: OpenAI API Key no encontrada en variables de entorno.")
        raise ValueError("OpenAI API Key no configurada.")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        print("Modelos de OpenAI inicializados correctamente.")
    except Exception as e:
        print(f"Error al inicializar los modelos de OpenAI: {e}")
        raise e

try:
    initialize_models()
except Exception as e:
    # En modo importación, solo imprimir. Streamlit lo manejará después si falla.
    print(f"Advertencia durante inicialización de modelos (puede ser normal si se importa): {e}")


# --- 3. Funciones de Procesamiento ---

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
        # En Streamlit, es mejor mostrar advertencia que detener todo si un archivo falla
        if 'st' in globals(): # Comprobar si st está disponible (evita error al importar)
            st.warning(f"Error procesando {os.path.basename(file_path)}: {e}. Verifique el archivo.")
        return None, None

def extract_name_from_cv(cv_sample_text: str, file_name: str) -> str:
    """Usa un LLM para extraer el nombre completo del propietario del CV."""
    default_name = f"Persona_{os.path.basename(file_name).split('.')[0]}"
    if not cv_sample_text or not llm: # Verificar si llm está inicializado
        print(f"Advertencia: No se puede extraer nombre de {file_name} (sin texto o sin LLM).")
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
        if not extracted_name or "Nombre Desconocido" in extracted_name or len(extracted_name) > 60:
             print(f"Advertencia: No se pudo extraer un nombre claro de {file_name}. Usando nombre por defecto.")
             if 'st' in globals(): st.info(f"No se pudo extraer un nombre claro de {file_name}, se usará '{default_name}'.")
             return default_name
        print(f"Nombre extraído de '{file_name}': {extracted_name}")
        return extracted_name
    except Exception as e:
        print(f"Error al extraer nombre de {file_name} usando LLM: {e}")
        if 'st' in globals(): st.warning(f"Error al intentar extraer el nombre de {file_name}. Se usará '{default_name}'.")
        return default_name

def create_cv_tools(chunks: List, person_name: str, file_name: str, db_suffix: str) -> Optional[Dict]:
    """Crea el vector store y la herramienta retriever para un CV, con sanitización estricta de nombres."""
    if not chunks or not embeddings_model: return None
    collection_name = f"coll_{db_suffix}_default"
    tool_name = f"tool_{db_suffix}_default"
    try:
        # Sanitización Colección
        coll_base = person_name.lower().replace(' ', '_')
        coll_sanitized = re.sub(r'[^a-z0-9_-]', '', coll_base)[:20] or "default"
        collection_name = f"cv_{db_suffix}_{coll_sanitized}_{os.urandom(4).hex()}"
        print(f"DEBUG: Nombre de colección generado: {collection_name}")

        # Sanitización Herramienta
        tool_base = f"cv_{person_name}"
        tool_base = re.sub(r'[\s.]+', '_', tool_base.lower())
        tool_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', tool_base)
        tool_sanitized = re.sub(r'[_]+', '_', tool_sanitized)
        tool_sanitized = re.sub(r'[-]+', '-', tool_sanitized)
        tool_sanitized = tool_sanitized.strip('_-')
        tool_sanitized = tool_sanitized[:64]
        if not tool_sanitized or not re.match(r'^[a-zA-Z0-9_-]+$', tool_sanitized):
             tool_name = f"buscar_cv_{db_suffix}_fallback_{os.urandom(2).hex()}"
        else: tool_name = tool_sanitized
        if not re.match(r'^[a-zA-Z0-9_-]+$', tool_name):
             tool_name = f"tool_cv_lookup_{db_suffix}_{os.urandom(4).hex()}"
             print(f"ERROR CRITICO: Usando nombre de herramienta forzado: {tool_name}")
        print(f"DEBUG: Nombre de herramienta FINAL generado: {tool_name}")

        # Metadatos y VectorStore
        for chunk in chunks: chunk.metadata["person"] = person_name; chunk.metadata["source"] = file_name
        vector_store = Chroma.from_documents(chunks, embeddings_model, collection_name=collection_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Crear Herramienta
        tool_description = f"Busca info en CV de {person_name} ({file_name}). SOLO para preguntas sobre {person_name}."
        cv_tool = create_retriever_tool(retriever, name=tool_name, description=tool_description)
        print(f"Tool '{tool_name}' creada para {person_name}.")
        return {"tool": cv_tool, "retriever": retriever, "vector_store": vector_store, "tool_name": tool_name}

    except Exception as e:
        print(f"Error creando tools para {person_name}: {e}")
        if 'st' in globals(): st.error(f"Error crítico al crear recursos para {person_name}: {e}")
        return None

# --- 4. Definición del Agente ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        if not isinstance(tools, list) or not all(hasattr(t, 'name') for t in tools):
             raise ValueError("Se esperaba una lista de objetos Tool válidos en Agent.__init__")
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        print(f"DEBUG [Agent.__init__]: Intentando vincular: {[t.name for t in tools]}")
        try:
            self.model = model.bind_tools(tools) # Vincular UNA VEZ aquí
            print(f"Agente inicializado. Modelo vinculado a: {list(self.tools.keys())}")
        except Exception as bind_error:
            print(f"--- ERROR CRÍTICO en model.bind_tools ---"); print(f"Herramientas: {[t.name for t in tools]}"); print(f"Error: {bind_error}"); traceback.print_exc()
            raise bind_error

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        is_ai_message = isinstance(result, AIMessage)
        has_tool_calls = hasattr(result, "tool_calls") and result.tool_calls and len(result.tool_calls) > 0
        return is_ai_message and has_tool_calls

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if not messages or not isinstance(messages[0], SystemMessage):
             messages = [SystemMessage(content=self.system)] + (messages if messages else [])
        elif messages[0].content != self.system:
             messages[0] = SystemMessage(content=self.system)
        try:
             message = self.model.invoke(messages)
             if hasattr(message, "tool_calls") and message.tool_calls:
                 print(f"DEBUG: LLM solicitó: {[t.get('name', 'N/A') for t in message.tool_calls]}")
             return {'messages': [message]}
        except Exception as invoke_error:
             print(f"ERROR en self.model.invoke: {invoke_error}")
             error_content = f"Error interno al procesar con el modelo: {invoke_error}"
             if hasattr(invoke_error, 'response'): error_content += f"\nDetalles: {invoke_error.response.text[:200]}"
             return {'messages': [AIMessage(content=error_content)]}

    def take_action(self, state: AgentState):
        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
             return {'messages': [ToolMessage(tool_call_id="error_state", name="error_state", content="Error interno.")]}
        tool_calls = last_message.tool_calls; results = []
        print(f"\nDEBUG: Ejecutando {len(tool_calls)} acciones...")
        for t in tool_calls:
            tool_name = t.get('name'); tool_args = t.get('args', {}); tool_id = t.get('id')
            if not tool_name or not tool_id: results.append(ToolMessage(tool_call_id=tool_id or "invalid", name=tool_name or "invalid", content="Error: Llamada inválida.")); continue
            if tool_name not in self.tools: results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=f"Error: Herramienta '{tool_name}' no encontrada.")); continue
            try:
                query_input = tool_args.get('query', tool_args if isinstance(tool_args, str) else str(tool_args))
                if not isinstance(query_input, str): query_input = str(query_input) if query_input else "Consulta general CV"
                print(f"DEBUG: Ejecutando '{tool_name}'...")
                result = self.tools[tool_name].invoke(query_input)
                if isinstance(result, list): result_str = "\n\n".join([f"Frag.:\n{doc.page_content}" for doc in result]) if result else f"No se encontraron fragmentos relevantes ({tool_name})."
                else: result_str = str(result)
                results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=result_str))
            except Exception as e: print(f"ERROR Ejec. tool '{tool_name}': {e}"); traceback.print_exc(); results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=f"Error ejecución: {traceback.format_exc()}"))
        return {'messages': results}

# --- 5. Funciones Específicas de la UI Streamlit ---

def setup_cv_environment_streamlit(uploaded_file1, uploaded_file2) -> Tuple[Optional[str], Optional[str], List, Optional[str]]:
    """Orquesta la carga, extracción de nombre y creación de herramientas para dos CVs subidos."""
    print("--- Iniciando Configuración del Entorno de CVs (Streamlit) ---")
    all_tools = []
    person1_name, person2_name = None, None
    tool_name1, tool_name2 = None, None
    file_name1, file_name2 = uploaded_file1.name, uploaded_file2.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp1.write(uploaded_file1.getvalue())
        cv_path1 = tmp1.name
        tmp2.write(uploaded_file2.getvalue())
        cv_path2 = tmp2.name

    tool_info1, tool_info2 = None, None # Inicializar
    try:
        # Procesar CV 1
        chunks1, sample1 = load_and_split_cv(cv_path1)
        if chunks1:
            person1_name = extract_name_from_cv(sample1, file_name1)
            tool_info1 = create_cv_tools(chunks1, person1_name, file_name1, "cv1") # Usa llm y embeddings globales
            if tool_info1:
                all_tools.append(tool_info1["tool"])
                tool_name1 = tool_info1["tool_name"] # Nombre sanitizado

        # Procesar CV 2
        chunks2, sample2 = load_and_split_cv(cv_path2)
        if chunks2:
            person2_name = extract_name_from_cv(sample2, file_name2)
            tool_info2 = create_cv_tools(chunks2, person2_name, file_name2, "cv2") # Usa llm y embeddings globales
            if tool_info2:
                all_tools.append(tool_info2["tool"])
                tool_name2 = tool_info2["tool_name"] # Nombre sanitizado

    finally:
        if 'cv_path1' in locals() and os.path.exists(cv_path1): os.remove(cv_path1)
        if 'cv_path2' in locals() and os.path.exists(cv_path2): os.remove(cv_path2)

    # Verificar si AMBOS se procesaron y generaron herramientas
    if not person1_name or not person2_name or not tool_info1 or not tool_info2 or len(all_tools) != 2:
        st.error("Falló la configuración. No se pudieron procesar ambos CVs o crear sus herramientas. Verifique los archivos/logs.")
        return None, None, [], None

    # Crear Prompt del Sistema con nombres de herramientas VALIDADOS
    system_prompt = f"""
Eres un asistente experto en comparar Curriculums Vitae (CVs). Tienes acceso a dos CVs:
1. El CV de {person1_name} (archivo: {file_name1}) al que accedes con la herramienta `{tool_name1}`.
2. El CV de {person2_name} (archivo: {file_name2}) al que accedes con la herramienta `{tool_name2}`.
Tu tarea es responder preguntas basándote ÚNICAMENTE en la información contenida en estos dos CVs.
Instrucciones IMPORTANTES para usar las herramientas:
- Analiza CUIDADOSAMENTE la pregunta del usuario.
- **Si la pregunta menciona CLARAMENTE a '{person1_name}'** (o variantes) y NO a {person2_name}, usa EXCLUSIVAMENTE la herramienta `{tool_name1}`.
- **Si la pregunta menciona CLARAMENTE a '{person2_name}'** (o variantes) y NO a {person1_name}, usa EXCLUSIVAMENTE la herramienta `{tool_name2}`.
- **Si la pregunta es GENERAL, NO menciona un nombre específico, o pregunta por 'ambos'/'quién'/'alguno'**, DEBES USAR AMBAS herramientas: `{tool_name1}` Y `{tool_name2}`.
- **Si tienes dudas sobre a quién se refiere la pregunta, es MÁS SEGURO usar AMBAS herramientas.**
Instrucciones para la RESPUESTA FINAL:
- Sintetiza la respuesta final basándote ESTRICTAMENTE en los resultados de las herramientas.
- Si consultaste ambos CVs, indica claramente qué información corresponde a {person1_name} y cuál a {person2_name}.
- Si una herramienta no devolvió información relevante, menciónalo si es pertinente.
- NO INVENTES INFORMACIÓN. Si la respuesta no está en los CVs, indica que no encontraste esa información.
- Sé objetivo.
Herramientas disponibles: {tool_name1}, {tool_name2}
"""
    print(f"--- Configuración Streamlit Completa. Nombres: '{person1_name}', '{person2_name}'. Herramientas: {[t.name for t in all_tools]} ---")
    return person1_name, person2_name, all_tools, system_prompt


# --- 6. Función Principal de la Aplicación Streamlit ---
def run_streamlit_app():
    """Ejecuta la interfaz y lógica de la aplicación Streamlit."""

    # Verificar inicialización de modelos (importante en Streamlit)
    if not llm or not embeddings_model:
         st.error("Error: Modelos LLM/Embeddings no inicializados. Verifica API Key.")
         st.stop()

    st.set_page_config(page_title="Comparador Inteligente de CVs", layout="wide")
    st.title("📄🤖 Comparador Inteligente de CVs con Agentes LangGraph")
    st.caption("Sube dos CVs en formato PDF para compararlos o hacer preguntas específicas.")

    # Inicializar Session State
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {"setup_done": False, "agent_executor": None, "person1_name": "P1", "person2_name": "P2", "system_prompt": ""}
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
    if "processing_lock" not in st.session_state: st.session_state.processing_lock = False

    # Barra Lateral
    with st.sidebar:
        st.header("Carga de Archivos")
        uploaded_file1 = st.file_uploader("Cargar CV 1 (PDF)", type="pdf", key="file1")
        uploaded_file2 = st.file_uploader("Cargar CV 2 (PDF)", type="pdf", key="file2")

        if uploaded_file1 and uploaded_file2:
            if st.button("🚀 Procesar CVs y Configurar Agente", key="process_btn", disabled=st.session_state.processing_lock):
                with st.spinner("Analizando CVs, extrayendo nombres y preparando el agente..."):
                    st.session_state.chat_messages = []
                    st.session_state.agent_state = {"setup_done": False, "agent_executor": None, "person1_name": "P1", "person2_name": "P2", "system_prompt": ""}
                    st.session_state.processing_lock = True # Bloquear durante procesamiento

                    person1, person2, tools, sys_prompt = setup_cv_environment_streamlit(uploaded_file1, uploaded_file2)

                    if person1 and person2 and tools and sys_prompt:
                        st.session_state.agent_state["person1_name"] = person1
                        st.session_state.agent_state["person2_name"] = person2
                        st.session_state.agent_state["system_prompt"] = sys_prompt
                        try:
                             # *** Instanciar Agente ***
                             agent_instance = Agent(llm, tools, system=sys_prompt)
                             st.session_state.agent_state["agent_executor"] = agent_instance
                             st.session_state.agent_state["setup_done"] = True
                             st.session_state.chat_messages = [AIMessage(content=f"¡Listo! CVs de {person1} y {person2} procesados.")]
                             st.success(f"Agente configurado para {person1} y {person2}.")
                        except Exception as e:
                             st.error(f"Error CRÍTICO al instanciar/vincular el agente: {e}")
                             st.session_state.agent_state["setup_done"] = False
                    else:
                        st.session_state.agent_state["setup_done"] = False
                    st.session_state.processing_lock = False # Desbloquear al final
                    st.rerun() # Refrescar UI después de procesar (éxito o fallo)
        elif st.session_state.agent_state["setup_done"]:
             st.info(f"Agente listo para {st.session_state.agent_state['person1_name']} y {st.session_state.agent_state['person2_name']}.")
        else:
             st.info("Carga ambos CVs y presiona 'Procesar'.")

    # Sección de Chat
    st.header("Chat de Consulta de CVs")

    if not st.session_state.agent_state["setup_done"]:
        st.warning("⚠️ Carga los CVs y procesa para empezar.")
    else:
        # Mostrar historial
        for message in st.session_state.chat_messages:
            role = "assistant" if isinstance(message, (AIMessage, SystemMessage)) else "user"
            if isinstance(message, SystemMessage): role="system"
            with st.chat_message(role):
                 if hasattr(message, 'content'): st.markdown(message.content)

        # Input del usuario
        prompt_placeholder = f"Pregunta sobre {st.session_state.agent_state['person1_name']} o {st.session_state.agent_state['person2_name']}..."
        if prompt := st.chat_input(prompt_placeholder, key="chat_input_main", disabled=st.session_state.processing_lock):
            st.session_state.chat_messages.append(HumanMessage(content=prompt))
            st.rerun() # Mostrar pregunta inmediatamente

    # Procesar última pregunta si es humana y no se está procesando ya
    if st.session_state.chat_messages and \
       isinstance(st.session_state.chat_messages[-1], HumanMessage) and \
       st.session_state.agent_state["setup_done"] and \
       not st.session_state.processing_lock:

        st.session_state.processing_lock = True # Bloquear antes de procesar
        agent_exec_instance = st.session_state.agent_state.get("agent_executor")
        last_prompt = st.session_state.chat_messages[-1].content

        if agent_exec_instance:
            # Usar st.chat_message para el spinner, así se muestra en el lugar correcto
            with st.chat_message("assistant"):
                with st.spinner("Pensando y buscando en los CVs..."):
                    ai_response_message = None # Variable para guardar la respuesta
                    try:
                        invoke_input = {"messages": [HumanMessage(content=last_prompt)]}
                        # Asegurar el prompt correcto
                        agent_exec_instance.system = st.session_state.agent_state["system_prompt"]

                        print(f"DEBUG: Invocando grafo...")
                        result = agent_exec_instance.graph.invoke(invoke_input)

                        if result and 'messages' in result and result['messages']:
                            ai_response = result['messages'][-1]
                            if isinstance(ai_response, AIMessage):
                                 ai_response_message = ai_response
                            else:
                                 print("Respuesta inesperada:", result['messages'])
                                 ai_response_message = AIMessage(content="Problema interno al generar respuesta.")
                        else:
                            print("Resultado vacío:", result)
                            ai_response_message = AIMessage(content="El agente no produjo una respuesta.")

                    except Exception as e:
                        print(f"Error en ejecución consulta: {e}")
                        traceback.print_exc()
                        ai_response_message = AIMessage(content=f"Lo siento, error procesando: {e}")
                    finally:
                        if ai_response_message:
                             st.session_state.chat_messages.append(ai_response_message)
                        st.session_state.processing_lock = False # Desbloquear SIEMPRE
                        st.rerun() # Mostrar respuesta o error final
        else:
            st.error("Error interno: Agente no encontrado en estado.")
            st.session_state.processing_lock = False # Desbloquear

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    run_streamlit_app()