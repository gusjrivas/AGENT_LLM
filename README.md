# Comparador Inteligente de CVs con Agentes LangChain/LangGraph y Streamlit

Este proyecto implementa una aplicación web interactiva que permite a los usuarios comparar dos Currículums Vitae (CVs) en formato PDF. Utiliza un agente inteligente construido con LangChain y LangGraph para responder preguntas sobre el contenido de los CVs, dirigiendo las consultas al documento correcto según se mencione a una persona específica o se realice una pregunta general sobre ambos.

**Video Demostrativo:** 

[https://github.com/gusjrivas/AGENT_LLM/blob/main/video/2025-04-19_agent_cv.mp4](https://github.com/gusjrivas/AGENT_LLM/blob/main/video/2025-04-19_agente_cv.mp4)


## Características Principales

*   **Interfaz Web Sencilla:** Construida con Streamlit para facilitar la interacción.
*   **Carga de Archivos:** Permite subir dos CVs en formato PDF.
*   **Extracción Automática de Nombres:** Utiliza un LLM (Modelo de Lenguaje Grande) para identificar automáticamente el nombre de la persona en cada CV.
*   **Procesamiento RAG (Retrieval-Augmented Generation):**
    *   Divide los CVs en fragmentos (chunks).
    *   Genera representaciones vectoriales (embeddings) para cada fragmento usando modelos de OpenAI.
    *   Almacena los embeddings en una base de datos vectorial en memoria (ChromaDB).
*   **Agente Inteligente (LangGraph):**
    *   Un único agente orquestador gestiona las consultas.
    *   Analiza la pregunta del usuario para determinar la intención (¿pregunta sobre Persona 1, Persona 2 o ambos?).
    *   Utiliza herramientas (`RetrieverTool`) específicas para buscar información relevante en el vector store del CV correspondiente.
    *   Sintetiza una respuesta coherente basada únicamente en la información recuperada de los CVs.
*   **Manejo Dinámico:** Se adapta a cualquier par de CVs subidos, generando dinámicamente las herramientas y el prompt del sistema para el agente.

## Tecnologías Utilizadas

*   **Python:** Lenguaje de programación principal.
*   **Streamlit:** Framework para la interfaz web.
*   **LangChain:** Framework para construir aplicaciones LLM.
    *   `PyPDFLoader`: Para cargar documentos PDF.
    *   `RecursiveCharacterTextSplitter`: Para dividir texto en chunks.
    *   `OpenAIEmbeddings`: Para generar vectores de texto.
    *   `Chroma`: Base de datos vectorial en memoria.
    *   `RetrieverTool`: Para crear herramientas de búsqueda RAG.
    *   `ChatOpenAI`: Para interactuar con modelos LLM de OpenAI (ej. GPT-4o mini).
*   **LangGraph:** Extensión de LangChain para crear agentes con estado y ciclos de decisión (grafos).
*   **OpenAI API:** Se requiere una clave API para usar los modelos de LLM y embeddings.

## Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd <nombre-del-directorio>
    ```

2.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Activar el entorno:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Si no tienes un `requirements.txt`, puedes crearlo con `pip freeze > requirements.txt` después de instalar manualmente, o listar las dependencias aquí):*
    ```bash
    pip install streamlit langchain langgraph langchain-openai langchain-community chromadb pypdf tiktoken python-dotenv
    ```

4.  **(Opcional, para visualizar el grafo) Instalar `pygraphviz` y `Graphviz`:**
    *   Instalar `pygraphviz`:
        ```bash
        pip install pygraphviz
        ```
    *   Instalar `Graphviz` (dependencia del sistema operativo):
        *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install graphviz graphviz-dev`
        *   **macOS (Homebrew):** `brew install graphviz`
        *   **Windows:** Descargar desde [graphviz.org/download/](https://graphviz.org/download/) y añadir a PATH.

## Configuración

1.  **Clave API de OpenAI:** Necesitas una clave API válida de OpenAI.
2.  **Variable de Entorno:** Configura la clave API como una variable de entorno llamada `OPENAI_API_KEY`. Puedes:
    *   **Definirla en tu sistema operativo:**
        *   Linux/macOS: `export OPENAI_API_KEY="tu_sk-..."`
        *   Windows (cmd): `set OPENAI_API_KEY=tu_sk-...`
        *   Windows (PowerShell): `$env:OPENAI_API_KEY="tu_sk-..."`
    *   **(Alternativa para desarrollo local) Crear un archivo `.env`:** Crea un archivo llamado `.env` en la raíz del proyecto y añade la línea:
        ```
        OPENAI_API_KEY="tu_sk-..."
        ```
        *(El código actual prioriza la variable de entorno del SO, pero puedes descomentar `load_dotenv()` en `app.py` si prefieres usar `.env`)*.

## Uso

1.  **Navega a la Carpeta:** Abre tu terminal y ve al directorio donde se encuentra el archivo `app.py`.
2.  **Activa el Entorno Virtual** (si creaste uno).
3.  **Ejecuta la Aplicación Streamlit:**
    ```bash
    streamlit run app.py
    ```
4.  **Interactúa con la Aplicación:**
    *   Se abrirá una pestaña en tu navegador.
    *   Usa la **barra lateral** para **cargar los dos CVs** en formato PDF.
    *   Haz clic en el botón "**Procesar CVs y Configurar Agente**". Espera a que termine el procesamiento (verás un spinner).
    *   Una vez configurado, aparecerá un mensaje de éxito y podrás usar el **área de chat principal**.
    *   Escribe tus preguntas sobre los CVs (ej: "¿Qué experiencia tiene [Nombre extraído 1] en Python?", "¿Quién tiene un título universitario?", "Compara sus habilidades de liderazgo").
    *   El agente responderá buscando la información en los documentos cargados.

## ¿Cómo Funciona Internamente?

1.  **Carga y Procesamiento:** Al pulsar "Procesar", la función `setup_cv_environment_streamlit` coordina:
    *   `load_and_split_cv`: Lee los PDFs y los divide en chunks.
    *   `extract_name_from_cv`: Usa el LLM para obtener los nombres.
    *   `create_cv_tools`: Genera embeddings para los chunks, los guarda en ChromaDB (en memoria) y crea una `RetrieverTool` específica para cada CV, sanitizando los nombres para que sean válidos.
2.  **Configuración del Agente:** Se genera un `system_prompt` dinámico que instruye al agente sobre cómo usar las herramientas creadas (`tool_name1`, `tool_name2`) basándose en los nombres extraídos (`person1_name`, `person2_name`). Se instancia la clase `Agent` (definida con LangGraph).
3.  **Ciclo de Consulta (LangGraph):**
    *   El usuario envía una pregunta (prompt).
    *   El nodo `llm` del agente recibe la pregunta y el `system_prompt`. Decide si necesita usar herramientas y cuáles.
    *   Si se eligen herramientas, el nodo `action` las ejecuta (llama a la `RetrieverTool` correspondiente, que busca en ChromaDB).
    *   Los resultados (texto recuperado) vuelven al nodo `llm`.
    *   El nodo `llm` sintetiza la respuesta final basándose en la pregunta original y los resultados de las herramientas.
    *   La respuesta se muestra al usuario.

## Visualización del Agente (Opcional)

Puedes generar un diagrama del flujo del agente (nodos `llm`, `action` y sus conexiones) ejecutando el script `visualize_agent.py` (asegúrate de tener `pygraphviz` y `Graphviz` instalados):

```bash
python visualize_agent.py
