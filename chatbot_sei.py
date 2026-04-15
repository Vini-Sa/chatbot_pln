
# ************ IMPORTAÇÕES PRINCIPAIS **************************

import streamlit as st
import os
import gc
import time
import base64
import fitz  # pymupdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader


# ***************CONFIGURAÇÕES DO STREAMLIT E INTERFACE*********************

st.set_page_config(
    page_title="Assistente SEI Julgar",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary: #1a56db;
  --primary-light: #e8f0fe;
  --accent: #16a34a;
  --accent-light: #dcfce7;
  --surface: #f8fafc;
  --border: #e2e8f0;
  --text: #0f172a;
  --muted: #64748b;
  --user-bubble: #1a56db;
  --ai-bubble: #ffffff;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

* { box-sizing: border-box; }

.stApp {
  background: linear-gradient(135deg, #f0f4ff 0%, #fafafa 100%);
  font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.hero-banner {
  background: linear-gradient(135deg, #1a56db 0%, #1e40af 50%, #1d4ed8 100%);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 28px;
  display: flex;
  align-items: center;
  gap: 20px;
  box-shadow: 0 8px 32px rgba(26,86,219,0.25);
}
.hero-banner h1 {
  color: #ffffff !important;
  font-size: 1.8rem !important;
  font-weight: 700 !important;
  margin: 0 !important;
  line-height: 1.2;
}
.hero-banner p {
  color: rgba(255,255,255,0.8);
  margin: 4px 0 0 0;
  font-size: 0.9rem;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid #bbf7d0;
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 12px;
}
.status-dot {
  width: 7px; height: 7px;
  background: var(--accent);
  border-radius: 50%;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.chat-area {
  background: #ffffff;
  border-radius: 16px;
  border: 1px solid var(--border);
  padding: 24px;
  min-height: 420px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 16px;
}

.msg-row {
  display: flex;
  gap: 12px;
  margin: 14px 0;
  animation: slideIn 0.2s ease-out;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  box-shadow: var(--shadow-sm);
}
.avatar.ai  { background: var(--primary-light); }
.avatar.usr { background: var(--user-bubble); }

.bubble {
  max-width: 78%;
  padding: 12px 16px;
  border-radius: 16px;
  font-size: 14.5px;
  line-height: 1.6;
  box-shadow: var(--shadow-sm);
}
.bubble.ai {
  background: var(--ai-bubble);
  color: var(--text);
  border: 1px solid var(--border);
  border-top-left-radius: 4px;
}
.bubble.user {
  background: var(--user-bubble);
  color: #ffffff;
  border-top-right-radius: 4px;
}

.sources-wrapper { margin-top: 10px; }
.source-tag {
  display: inline-block;
  border-radius: 6px;
  padding: 2px 8px;
  font-size: 11.5px;
  margin: 3px 3px 0 0;
  font-weight: 500;
}
.source-text {
  background: var(--primary-light);
  color: var(--primary);
  border: 1px solid #bfdbfe;
}
.source-image {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}

[data-testid="stChatInput"] > div {
  border: 2px solid var(--border) !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: var(--shadow-sm) !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}
[data-testid="stChatInput"] button {
  background: var(--primary) !important;
  border-radius: 10px !important;
}

.stButton > button {
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.info-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.7;
}
.info-card strong { color: var(--text); }

@keyframes slideIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ******************** SIDEBAR **********************************

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 16px 0;">
          <div style="font-size:2.4rem;">⚖️</div>
          <div style="font-weight:700; font-size:1.05rem; color:#0f172a;">SEI Julgar</div>
          <div style="font-size:12px; color:#64748b;">Assistente Virtual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Modelo Gemini**")

    id_model = st.selectbox(
        "Selecione o modelo:",
        ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        label_visibility="collapsed",
        help="gemini-2.0-flash: 200 req/dia (free tier) | gemini-2.5-flash: 20 req/dia",
    )

    temperature = st.slider(
        "Temperatura", 0.0, 1.0, 0.3, 0.05,
        help="Valores mais baixos = respostas mais precisas e determinísticas.",
    )

    k_docs = st.slider(
        "Trechos por consulta", 1, 6, 3,
        help="Quantidade de trechos do manual recuperados para cada resposta.",
    )

    st.markdown("---")
    st.markdown("**Base de conhecimento**")

    if st.button("🔄 Reindexar base", use_container_width=True,
                 help="Apaga e reconstrói o índice incluindo texto e imagens do PDF."):
        # Passo 1: libera conexões do ChromaDB limpando o cache
        st.cache_resource.clear()
        st.session_state["pending_reindex"] = True
        st.rerun()

    # Indicador de chunks indexados
    if "index_stats" in st.session_state:
        txt, img = st.session_state["index_stats"]
        st.markdown(
            f'<div style="font-size:12px;color:#64748b;margin-top:4px;">'
            f'📄 {txt} texto &nbsp;|&nbsp; 🖼️ {img} imagem</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    with st.expander("ℹ️ Sobre o assistente"):
        st.markdown(
            """
            <div class="info-card">
            <strong>SEI Julgar Assistente v2.0</strong><br>
            Especializado no módulo <strong>SEI Julgar</strong>.<br><br>
            • Texto e imagens do manual indexados<br>
            • Busca semântica com ChromaDB<br>
            • Powered by Google Gemini
            </div>
            """,
            unsafe_allow_html=True,
        )


# ************************ CABEÇALHO ****************************************************

st.markdown(
    """
<div class="hero-banner">
  <div style="font-size:3rem; line-height:1;">⚖️</div>
  <div>
    <h1>Assistente Virtual do SEI Julgar</h1>
    <p>Tire dúvidas sobre o módulo SEI Julgar com base na documentação oficial.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ************************** FUNÇÕES DE INDEXAÇÃO ******************************************

def describe_pdf_pages(pdf_path: str, api_key: str) -> list[Document]:
    """Renderiza cada página do PDF como screenshot e descreve o conteúdo visual com Gemini."""
    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=api_key,
    )

    docs = []
    pdf_doc = fitz.open(pdf_path)
    total = len(pdf_doc)

    progress = st.progress(0, text="Analisando páginas com Gemini Vision...")

    for page_num, page in enumerate(pdf_doc):
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()

        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Esta é a página {page_num + 1} do manual do sistema SEI Julgar. "
                    "Se a página mostrar telas do sistema, botões, formulários, menus, "
                    "tabelas ou qualquer elemento de interface: descreva tudo detalhadamente "
                    "para que um usuário consiga reproduzir os passos. "
                    "Se a página for apenas texto corrido sem elementos visuais do sistema, "
                    "responda somente com: SKIP"
                ),
            },
        ])

        # Retry com backoff em caso de rate limit (429)
        for attempt in range(4):
            try:
                response = vision_llm.invoke([message])
                description = response.content.strip()
                if description.upper() != "SKIP":
                    docs.append(Document(
                        page_content=description,
                        metadata={"source": "image", "page": page_num + 1},
                    ))
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 60 * (attempt + 1)  # 60s, 120s, 180s
                    progress.progress(
                        (page_num + 1) / total,
                        text=f"Rate limit — aguardando {wait}s... (página {page_num + 1}/{total})",
                    )
                    time.sleep(wait)
                else:
                    break  # ignora outros erros ou esgotou tentativas

        progress.progress(
            (page_num + 1) / total,
            text=f"Gemini Vision: página {page_num + 1}/{total}...",
        )

    progress.empty()
    pdf_doc.close()
    return docs


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_vectorstore():
    """Carrega (ou cria) o vectorstore — cacheado independente do valor de k."""
    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = load_embeddings()

    # Abre (ou cria) a coleção — não deleta o diretório, apenas verifica se tem dados
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    try:
        doc_count = vectorstore._collection.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        pdf_files = ["Documentação SEI Julgar - Secretaria.pdf"]
        all_chunks: list[Document] = []

        for pdf_path in pdf_files:
            if not os.path.exists(pdf_path):
                continue

            # 1. Extrai texto
            reader = PdfReader(pdf_path)
            text_pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)

            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
            text_chunks = splitter.create_documents(
                text_pages,
                metadatas=[{"source": "text"} for _ in text_pages],
            )
            all_chunks.extend(text_chunks)

            # 2. Renderiza páginas e descreve conteúdo visual
            st.info("Analisando páginas com Gemini Vision... isso pode levar alguns minutos na primeira vez.")
            image_docs = describe_pdf_pages(pdf_path, os.getenv("GOOGLE_API_KEY"))
            all_chunks.extend(image_docs)

        if all_chunks:
            vectorstore.add_documents(all_chunks)

    # Atualiza estatísticas na sidebar
    try:
        data = vectorstore.get(include=["metadatas"])
        metas = [m or {} for m in data["metadatas"]]
        txt_n = sum(1 for m in metas if m.get("source") == "text")
        img_n = sum(1 for m in metas if m.get("source") == "image")
        st.session_state["index_stats"] = (txt_n, img_n)
    except Exception:
        pass

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2, "lambda_mult": 0.7},
    )


@st.cache_resource
def load_llm(model_id: str, temp: float):
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=temp,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def config_rag_chain(llm, retriever):
    context_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Dado o histórico de conversa e a última pergunta do usuário, "
            "reformule a pergunta de forma autônoma sem respondê-la. "
            "Se já for clara, retorne-a sem alteração.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um assistente virtual especializado **exclusivamente** no módulo **SEI Julgar**.
Responda com base apenas na documentação oficial fornecida no contexto.
O contexto pode incluir descrições de imagens/telas do sistema — use essas informações para detalhar melhor os passos.

REGRAS OBRIGATÓRIAS:
1. Responda somente sobre o SEI Julgar. Para outros assuntos diga: "Posso responder apenas sobre o funcionamento do SEI Julgar."
2. Nunca invente informações. Se não souber, diga claramente.
3. Use linguagem formal, clara e objetiva.
4. Quando relevante, estruture a resposta com passos numerados ou listas.
5. Contexto disponível:\n\n{context}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)


# ******************* INTERFACE DO CHAT *******************************

def render_message(content: str, is_user: bool, sources: list = None):
    avatar = "🧑" if is_user else "🤖"
    role = "user" if is_user else "assistant"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

        if sources:
            tags = ""
            for doc in sources:
                src = doc.metadata.get("source", "text")
                page = doc.metadata.get("page", "")
                if src == "image":
                    tags += f'<span class="source-tag source-image">🖼️ Página {page}</span>'
                else:
                    tags += f'<span class="source-tag source-text">📄 Texto</span>'
            st.markdown(
                f'<div class="sources-wrapper">{tags}</div>',
                unsafe_allow_html=True,
            )


def chat_response(rag_chain, user_input: str):
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history[:-1],
        })
        answer = response.get("answer", "Erro ao gerar a resposta.")
        docs = response.get("context", [])
    except Exception as e:
        err = str(e)
        if "429" in err:
            import re
            wait_match = re.search(r"retry.*?(\d+)s", err, re.IGNORECASE)
            wait = wait_match.group(1) if wait_match else "alguns minutos"
            answer = (
                f"⚠️ **Limite de requisições atingido** (cota gratuita da API Gemini).\n\n"
                f"Aguarde **{wait} segundos** e tente novamente.\n\n"
                f"💡 Dica: use o modelo **gemini-2.0-flash** na sidebar — ele tem cota maior (200 req/dia)."
            )
        else:
            answer = f"Erro ao processar sua pergunta: {err}"
        docs = []

    st.session_state.chat_history.append(AIMessage(content=answer))
    return answer, docs


# ************************* EXECUÇÃO PRINCIPAL ************************************

# Reindex: deleta a coleção via API do ChromaDB — sem tocar no sistema de arquivos
if st.session_state.get("pending_reindex"):
    gc.collect()
    try:
        import chromadb
        _client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(allow_reset=True),
        )
        for col in _client.list_collections():
            _client.delete_collection(col.name)
        del _client
        gc.collect()
    except Exception:
        pass

    st.session_state["pending_reindex"] = False
    st.session_state["chat_history"] = [
        AIMessage(content="Base apagada! Aguarde enquanto reindexo o manual...")
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou o assistente do SEI Julgar. Como posso ajudar?")
    ]

if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ **GOOGLE_API_KEY** não encontrada. Verifique o arquivo `.env`.")
    st.stop()

try:
    with st.spinner("Carregando base de conhecimento..."):
        llm = load_llm(id_model, temperature)
        retriever = config_retriever(k_docs)
        rag_chain = config_rag_chain(llm, retriever)

    col_status, col_clear = st.columns([4, 1])
    with col_status:
        st.markdown(
            '<div class="status-badge"><span class="status-dot"></span> Sistema online</div>',
            unsafe_allow_html=True,
        )
    with col_clear:
        if st.button("🗑️ Limpar chat", use_container_width=True):
            st.session_state.chat_history = [
                AIMessage(content="Histórico limpo. Como posso ajudar?")
            ]
            st.rerun()

    chat_container = st.container(border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            render_message(msg.content, isinstance(msg, HumanMessage))

    user_input = st.chat_input("Digite sua pergunta sobre o SEI Julgar...")

    if user_input:
        render_message(user_input, is_user=True)

        with st.spinner("Consultando a documentação..."):
            answer, docs = chat_response(rag_chain, user_input)

        render_message(answer, is_user=False, sources=docs)
        st.rerun()

except Exception as e:
    st.error(f"Erro ao inicializar o chatbot: {str(e)}")
