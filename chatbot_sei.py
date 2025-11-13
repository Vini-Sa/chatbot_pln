# ************ IMPORTAÇÕES PRINCIPAIS **************************

import streamlit as st
from pathlib import Path
import requests, os, time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader


# *********** MÓDULO DE PLN – Pré-processamento e enriquecimento semântico ******************

import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Tenta carregar o modelo SpaCy em português (usa versão menor se a maior não estiver disponível)
try:
    nlp = spacy.load("pt_core_news_md")
except Exception:
    try:
        nlp = spacy.load("pt_core_news_sm")
    except Exception:
        st.error("Erro: modelo SpaCy não encontrado. Execute: !python -m spacy download pt_core_news_md")
        st.stop()

# Carrega o modelo de embeddings para cálculo de similaridade semântica
embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Carrega stopwords da língua portuguesa (ou cria lista vazia em caso de falha)
try:
    stop_words = set(stopwords.words("portuguese"))
except Exception:
    stop_words = set()

# Função para normalizar e remover stopwords - Normaliza o texto removendo pontuação, acentuação e stopwords.
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    try:
        tokens = [t for t in word_tokenize(text, language="portuguese") if t not in stop_words]
    except Exception:
        tokens = text.split()
    return " ".join(tokens).strip()

# Função para analizar a intenção do usuário - Identifica a intenção geral do usuário com base em palavras-chave simples.
def analyze_intent(text: str) -> str:
    if any(x in text for x in ["como", "de que forma", "procedimento", "passo", "como faço", "instrução"]):
        return "pedido de instrução"
    if any(x in text for x in ["por que", "motivo", "razão", "porquê"]):
        return "pedido de justificativa"
    if any(x in text for x in ["quem", "órgão", "setor", "quem é", "responsável"]):
        return "pedido de identificação"
    return "outros"

# Função para analisar o sentimento da pergunta - Analisa o sentimento geral da frase com base em palavras positivas e negativas.
def analyze_sentiment(text: str) -> str:
    try:
        doc = nlp(text)
        positives = sum(1 for tok in doc if tok.lemma_.lower() in {"bom", "ótimo", "excelente", "favorável", "positivo", "conforme"})
        negatives = sum(1 for tok in doc if tok.lemma_.lower() in {"ruim", "erro", "problema", "falha", "negativo", "indevido"})
        if positives > negatives:
            return "positivo"
        if negatives > positives:
            return "negativo"
        return "neutro"
    except Exception:
        return "neutro"

# Função para extrair as entidades da pergunta - Extrai entidades nomeadas do texto (pessoas, órgãos, locais, etc).
def extract_entities(text: str) -> dict:
    doc = nlp(text)
    ents = {}
    for ent in doc.ents:
        ents.setdefault(ent.label_, []).append(ent.text)
    return ents

#Função para expandir semânticamente a pergunta - Adiciona termos semanticamente próximos para melhorar a recuperação de contexto.
def semantic_expansion(text: str) -> str:
    base_terms = ["processo", "documento", "protocolo", "julgamento", "tramitação", "autuação", "andamento", "petição", "decisão", "parecer"]
    try:
        query_emb = embedder.encode(text, convert_to_tensor=True)
        base_embs = embedder.encode(base_terms, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, base_embs)[0]
        best_terms = [base_terms[i] for i, s in enumerate(scores) if float(s) > 0.40]
        additions = " ".join([t for t in best_terms if t not in text])
        return (text + " " + additions).strip()
    except Exception:
        return text

# Pipeline completo das funções
def preprocess_input(text: str) -> str:
    normalized = normalize_text(text)
    intent = analyze_intent(normalized)
    sentiment = analyze_sentiment(normalized)
    entities = extract_entities(normalized)
    expanded = semantic_expansion(normalized)

    print("[PLN] Intenção:", intent)
    print("[PLN] Sentimento:", sentiment)
    print("[PLN] Entidades:", entities)
    print("[PLN] Pergunta expandida:", expanded)

    return expanded


# ***************CONFIGURAÇÕES DO STREAMLIT E INTERFACE*********************

st.set_page_config(
    page_title="Atendimento SEI Julgar",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_dotenv()

# CSS customizado: define cores, bolhas e estilo visual do chat
st.markdown(
    """
<style>
:root {
  --primary: #2493d1;
  --accent-green: #a0b83c;
  --accent-green-focus: #8da635;
  --text-default: #1b2b3a;
  --muted: #888;
  --bubble-ai-bg: #f9faf9;
  --bubble-user-bg: #2493d1;
  --sidebar-bg: #f5f6f7;
}

/* ====== ESTILO GERAL ====== */
.stApp { background-color: white; font-family: 'Inter', sans-serif; }
.main .block-container { color: var(--text-default); }

/* ====== SIDEBAR ====== */
section[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
  padding: 15px;
  border-right: 1px solid #e3e3e3;
}

/* ====== BOLHAS DO CHAT ====== */
.chat-bubble {
  padding: 0.9em 1.3em;
  border-radius: 1.2em;
  margin: 0.5em 0;
  max-width: 85%;
  font-size: 15px;
  line-height: 1.4;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  animation: fadeIn 0.25s ease-in-out;
}
.user-bubble {
  background-color: var(--bubble-user-bg);
  color: #fff;
  margin-left: auto;
}
.ai-bubble {
  background-color: var(--bubble-ai-bg);
  color: var(--primary);
  border: 1px solid #e5e5e5;
}

/* ====== CAMPO DE ENTRADA DO CHAT ====== */
[data-testid="stChatInput"] > div {
  display: flex !important;
  align-items: center !important;
  border: 2px solid var(--accent-green) !important;
  border-radius: 30px !important;
  background-color: #ffffff !important;
  box-shadow: 0 4px 12px rgba(160,184,60,0.15) !important;
  transition: all 0.3s ease;
  padding: 6px 10px !important;
}

[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--accent-green-focus) !important;
  box-shadow: 0 0 0 4px rgba(160,184,60,0.2) !important;
  transform: translateY(-2px);
}

[data-testid="stChatInput"] input {
  flex: 1 !important;
  border: none !important;
  outline: none !important;
  font-size: 1rem !important;
  padding: 10px 14px !important;
}

[data-testid="stChatInput"] button {
  background-color: var(--accent-green) !important;
  color: white !important;
  border-radius: 50% !important;
  width: 42px !important;
  height: 42px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin-left: 8px !important;
}


/* ====== BOTÕES ====== */
.stButton button {
  background-color: var(--accent-green) !important;
  color: white !important;
  border: none !important;
  padding: 8px 16px !important;
  border-radius: 20px !important;
  font-size: 14px !important;
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(6px);}
  to {opacity: 1; transform: translateY(0);}
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }

/* ====== BOTÃO DE MENU ====== */
.toggle-btn {
  position: fixed;
  top: 15px;
  left: 15px;
  background-color: var(--primary);
  color: white;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: bold;
  cursor: pointer;
  z-index: 1000;
  transition: background 0.3s;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.toggle-btn:hover { background-color: #1c6fa1; }
</style>

<!-- BOTÃO FLUTUANTE PARA REABRIR SIDEBAR -->
<div class="toggle-btn" onclick="window.parent.document.querySelector('button[kind=header]')?.click()">☰ Menu</div>
""",
    unsafe_allow_html=True,
)


# ******************** SIDEBAR - Barra lateral do chat, colocamos para mostrar o que é possível fazer no Streamlit **********************************

with st.sidebar:
    st.image(
        "https://www.gov.br/ans/pt-br/assuntos/noticias/sobre-ans/sistema-eletronico-de-informacoes-ficara-indisponivel-de-12-a-16-9/sei-logo.png",
        width=130,
    )

    st.markdown("---")
    st.title("⚙️ Configurações")
    st.markdown("---")

    id_model = st.selectbox(
        "Selecione o modelo:",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
    )

    temperature = st.slider(
        "Temperatura (criatividade):",
        0.0,
        1.0,
        0.7,
        0.1,
        help="Valores mais baixos produzem respostas mais determinísticas.",
    )

    st.markdown("---")
    st.subheader("Sobre")
    st.info(
        """
**SEI Julgar Assistente** v1.0
- Especializado no módulo SEI Julgar
- Suporte a secretarias e gabinetes
- Base de conhecimento local
"""
    )


# ************************ CABEÇALHO ****************************************************

st.markdown(
    """
<div style="display:flex; align-items:center; justify-content:center; gap:18px; margin:25px 0 40px 0;">
  <img src="https://www.gov.br/ans/pt-br/assuntos/noticias/sobre-ans/sistema-eletronico-de-informacoes-ficara-indisponivel-de-12-a-16-9/sei-logo.png"
       style="width:110px; height:auto;">
  <h1 style="color:#2c7ec8; margin:0; font-size:2.4rem; font-weight:700;">
    Assistente Virtual do SEI Julgar
  </h1>
</div>
""",
    unsafe_allow_html=True,
)


# ************************** FUNÇÕES DE INDEXAÇÃO ******************************************

@st.cache_resource
def load_llm():
    #Carrega o modelo de linguagem da API Groq
    return ChatGroq(model=id_model, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def config_retriever():
    """Cria ou carrega base vetorial (ChromaDB)."""

    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # Carrega o PDF e extrai o texto
        pdf_files = ["Documentação SEI Julgar - Secretaria.pdf"]
        texts = []
        for file in pdf_files:
            if os.path.exists(file):
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)

        # Divide em chunks
        """
        Chunk_size - Define o tamanho dos chunks
        Chunk_overlap - Define o quanto os chunks se sobrepõem
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.create_documents(texts)

        # Cria o banco vetorial com ChromaDB
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
        vectorstore.persist()

    # Retorna o retriever
    """
    O Maximal Marginal Relevance busca no banco os chunks mais relevantes e diversos para evitar a redundância de informações.
        fetch_k - Define a quantidade de chunks que serão recuperados do banco
        k - Define a quantidade de chunks que serão retornados a llm
        lambda_mult - Controla o peso de Relevância e Diversidade 0.7(70% foco na relevância e 30% na diversidade)
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4, "lambda_mult": 0.7},
    )


def config_rag_chain(llm, retriever):
    """Cria pipeline RAG: reformulação + busca + resposta final."""

    # Prompt que reformula perguntas do usuário
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformule perguntas do usuário sem respondê-las."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Recuperador com histórico
    history_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_prompt
    )

    # Prompt principal de QA
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente virtual especializado **exclusivamente** no módulo **SEI Julgar**.
        Forneça respostas **técnicas e objetivas**, baseadas apenas na documentação oficial.

        REGRAS:
        1. Não inventar informações.
        2. Não responder temas fora do SEI Julgar.
        3. Manter tom formal e instrutivo.
        4. Se o tema for irrelevante, diga: "Posso responder apenas sobre o funcionamento do SEI Julgar."
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\nContexto relevante do manual:\n{context}")
    ])

    #envia a llm o prompt acima mais os documentos retornados pelo retriever
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)



# ******************* INTERFACE DO CHAT *******************************

def show_message(content, is_user=False):
    """Renderiza mensagens no formato de bolhas."""
    bubble_class = "user-bubble" if is_user else "ai-bubble"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>{content}</div>", unsafe_allow_html=True)

def chat_response(rag_chain, user_input):
    """Processa entrada do usuário e retorna resposta do modelo."""
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
    answer = response.get("answer", "Erro ao gerar resposta.")
    st.session_state.chat_history.append(AIMessage(content=answer))
    return answer


# ************************* EXECUÇÃO PRINCIPAL ************************************

#Cria o histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá 👋, sou o assistente do SEI Julgar! Como posso ajudar?")]
# Validação da chave da API
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY não encontrada.")
    st.stop()
# Inicialização do modelo e da base vetorial
try:
    with st.spinner("Carregando modelo e base vetorial..."):
        llm = load_llm()
        retriever = config_retriever()
        rag_chain = config_rag_chain(llm, retriever)
    st.success("✅ Sistema carregado com sucesso!")

    # Botão de "limpar chat"
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🧹 Limpar chat", key="clear", use_container_width=True):
            st.session_state.chat_history = [AIMessage(content="Chat limpo! 😊 Como posso ajudar agora?")]

    # Exibição do histórico de mensagens
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            show_message(msg.content, isinstance(msg, HumanMessage))

    # Captura a entrada do usuário
    user_input = st.chat_input("Digite sua mensagem...")

    # Processamento da resposta
    if user_input:
        with chat_container:
            show_message(user_input, is_user=True)
        placeholder = st.empty()
        placeholder.markdown("<div class='ai-bubble'><em>Digitando...</em></div>", unsafe_allow_html=True)
        with st.spinner("Processando..."):
            answer = chat_response(rag_chain, preprocess_input(user_input))
        placeholder.empty()
        with chat_container:
            show_message(answer, is_user=False)

# Tratamento de erros
except Exception as e:
    st.error(f"Erro ao inicializar o chatbot: {str(e)}")
