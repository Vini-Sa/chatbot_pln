

# ************ IMPORTAÇÕES PRINCIPAIS **************************

import streamlit as st
from pathlib import Path
import requests
import os
import time
import shutil
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

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

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

</style>

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
def load_llm(model_id, temp):
    # Carrega o modelo de linguagem da API Groq
    return ChatGroq(model=model_id, temperature=temp, api_key=os.getenv("GROQ_API_KEY"))


def delete_vector_database():
    """Deleta completamente a base de dados vetorial existente."""
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Erro ao deletar base: {e}")


def create_vector_database_from_pdfs():
    """Cria a base de dados vetorial a partir dos PDFs na pasta do projeto."""
    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Carrega PDFs da pasta do projeto
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("Aviso: Nenhum arquivo PDF encontrado na pasta do projeto")
        return None

    texts = []
    for file in pdf_files:
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")

    if not texts:
        print("Aviso: Nenhum texto foi extraído dos PDFs")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents(texts)

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    
    return vectorstore


@st.cache_resource
def config_retriever():
    persist_dir = "./chroma_db"
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Verifica se a base existe
    if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        # Base existe, carrega
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # Base não existe, cria a partir dos PDFs
        vectorstore = create_vector_database_from_pdfs()
        if vectorstore is None:
            st.error("❌ Erro: Nenhum arquivo PDF encontrado na pasta do projeto!")
            st.stop()

    # Retorna o retriever
    # O Maximal Marginal Relevance busca no banco os chunks mais relevantes e diversos para evitar a redundância de informações.
    # fetch_k - Define a quantidade de chunks que serão recuperados do banco
    # k - Define a quantidade de chunks que serão retornados a llm
    # lambda_mult - Controla o peso de Relevância e Diversidade 0.7(70% foco na relevância e 30% na diversidade)

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4, "lambda_mult": 0.7},
    )


def is_question_or_statement(text):
    """
    Detecta se a entrada é uma pergunta ou uma assertiva/afirmação.
    Retorna True se for pergunta, False se for assertiva/informação.
    """
    text_clean = text.strip().lower()
    
    # Indicadores de pergunta
    question_indicators = ["?", "qual", "quais", "como", "por que", "porquê", "quando", "onde", 
                          "o que", "pode me", "poderia", "consegue", "é possível", "sabe"]
    
    is_question = (
        text.endswith("?") or 
        any(text_clean.startswith(word) for word in question_indicators)
    )
    return is_question


def config_rag_chain(llm, retriever):
    # Cria pipeline RAG: reformulação + busca + resposta final.

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

    # Prompt principal de QA - TOTALMENTE BASEADO NO MANUAL
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Você é um assistente virtual especializado **exclusivamente** no módulo **SEI Julgar**.
Seu ÚNICO propósito é responder dúvidas com base na documentação oficial do manual.

========== REGRAS FUNDAMENTAIS ==========
1. RESPONDA APENAS COM BASE NO MANUAL: use exclusivamente as informações do contexto fornecido.
2. NÃO USE INFORMAÇÕES DO USUÁRIO COMO VERDADE: o usuário pode fornecer informações incorretas no chat - ignore-as.
3. NÃO INVENTE RESPOSTAS: se a informação não estiver no manual, diga que não há informação suficiente.
4. SE TIVER DÚVIDA: responda "De acordo com o manual disponível, não há informações suficientes para responder essa questão com precisão."
5. PODE CORRIGIR ERROS: se o usuário disser algo que contradiz o manual, corrija com gentileza, citando o manual.
6. MANTENHA A HUMILDADE: sua fonte de verdade é APENAS o manual fornecido.

========== FORMATO DE RESPOSTA ==========
- Respostas técnicas, objetivas e diretas
- Cite a documentação quando apropriado
- Se for uma assertiva (não-pergunta), verifique se a acertiva é correta de acordo com o manual e corrija se necessário, caso não seja possível aferir a assertiva com o manual, responda que não há informação suficiente para confirmar ou negar a afirmação, mas sugira onde o usuário pode procurar aprofundar suas informações caso tenha isso na base de conhecimento.
- Se for uma pergunta fora do escopo do manual ou do SEI Julgar, responda: "Entendo, mas minha função é esclarecer dúvidas sobre o SEI Julgar. Tem alguma pergunta sobre o funcionamento?"
- Mantenha tom formal e profissional

========== CONTEXTO DISPONÍVEL DO MANUAL ==========
{context}
"""
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "{input}"
        )
    ])

    # envia a llm o prompt acima mais os documentos retornados pelo retriever
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)


# ******************* INTERFACE DO CHAT *******************************

def show_message(content, is_user=False):
    # Renderiza mensagens no formato de bolhas.
    bubble_class = "user-bubble" if is_user else "ai-bubble"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>{content}</div>", unsafe_allow_html=True)


def chat_response(rag_chain, user_input):
    # Processa a entrada do usuário e retorna a resposta do modelo.
    
    # Detecta se é pergunta ou assertiva
    is_question = is_question_or_statement(user_input)

    # 1. Adiciona a mensagem do usuário ao histórico
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    # 2. Se for assertiva/informação, retorna resposta padrão
    if not is_question:
        response_text = (
            "Entendo o que você disse, mas minha função é esclarecer dúvidas sobre o funcionamento do SEI Julgar "
            "com base na documentação oficial. Tem alguma pergunta sobre o sistema?"
        )
        st.session_state.chat_history.append(AIMessage(content=response_text))
        return response_text

    # 3. Chama a RAG Chain (history-aware + retriever + QA)
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    # 4. Extrai resposta do modelo
    answer = response.get("answer", "Erro ao gerar a resposta.")

    # 5. Adiciona a resposta ao histórico
    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

    return answer


# ************************* EXECUÇÃO PRINCIPAL ************************************

# Cria o histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá 👋, sou o assistente do SEI Julgar! Como posso ajudar?")]

if "rebuild_db" not in st.session_state:
    st.session_state.rebuild_db = False

if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = None

# ================== SIDEBAR: GERENCIAMENTO DE BASE DE CONHECIMENTO ==================

with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Gerenciar Base de Conhecimento")
    
    # Upload de documentos
    uploaded_files = st.file_uploader(
        "Adicionar/Atualizar documentos (PDF):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Faça upload de PDFs para atualizar a base de conhecimento. Se houver PDF com mesmo nome, será sobrescrito."
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"📄 {len(uploaded_files)} arquivo(s)")
        with col2:
            if st.button("🔄 Salvar e Reconstruir", key="rebuild_btn", use_container_width=True):
                st.session_state.rebuild_db = True
                st.session_state.uploaded_files_cache = uploaded_files
    
    # Botão de resetar base
    if st.button("🗑️ Resetar Base", key="reset_btn", use_container_width=True, help="Deleta a base atual e recria do zero"):
        delete_vector_database()
        st.cache_resource.clear()
        st.success("✅ Base deletada!")
        st.rerun()
    
    # Informações sobre a base
    st.markdown("---")
    st.caption("ℹ️ Status da Base")
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        st.success(f"✅ Base de conhecimento ativa")
        # Conta os PDFs na pasta do projeto
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        if pdf_files:
            st.info(f"📄 {len(pdf_files)} PDF(s) carregado(s)")
            for pdf in pdf_files:
                st.caption(f"  • {pdf}")
    else:
        st.warning("⚠️ Base não inicializada")

# ============================================================================

# Validação da chave da API
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY não encontrada.")
    st.stop()

# Função auxiliar para reconstruir a base
def rebuild_vector_database(uploaded_files):
    """
    Reconstrói a base vetorial com novos documentos.
    1. Salva os PDFs na pasta do projeto (sobrescrevendo se necessário)
    2. Deleta a base anterior
    3. Cria a nova base a partir dos PDFs
    """
    try:
        with st.spinner("💾 Salvando PDFs na pasta do projeto..."):
            # Salva os PDFs uploadados na pasta do projeto
            for uploaded_file in uploaded_files:
                try:
                    file_path = uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ {uploaded_file.name} salvo com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar {uploaded_file.name}: {e}")
                    return False
        
        with st.spinner("🔄 Deletando base anterior..."):
            # Deleta a base anterior
            delete_vector_database()
        
        with st.spinner("🔨 Reconstruindo base vetorial..."):
            # Cria nova base
            vectorstore = create_vector_database_from_pdfs()
            if vectorstore is None:
                st.error("❌ Erro ao criar nova base!")
                return False
            
            # Conta chunks
            all_texts = vectorstore.similarity_search("")
            st.success(f"✅ Base vetorial reconstruída com sucesso!")
            return True
            
    except Exception as e:
        st.error(f"❌ Erro ao reconstruir base: {str(e)}")
        return False

# Reconstruir base se necessário
if st.session_state.rebuild_db and "uploaded_files_cache" in st.session_state:
    if rebuild_vector_database(st.session_state.uploaded_files_cache):
        st.session_state.rebuild_db = False
        st.session_state.uploaded_files_cache = None
        # Limpa cache de config_retriever para forçar recarga
        st.cache_resource.clear()
        # Força recarga do Streamlit para carregar a nova base
        st.rerun()

# Inicialização do modelo e da base vetorial
try:
    with st.spinner("Carregando modelo e base vetorial..."):
        llm = load_llm(id_model, temperature)
        retriever = config_retriever()
        rag_chain = config_rag_chain(llm, retriever)
    #st.success("✅ Sistema carregado com sucesso!")

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

        with st.spinner("Processando..."):
            answer = chat_response(rag_chain, user_input)

        with chat_container:
            show_message(answer, is_user=False)


# Tratamento de erros
except Exception as e:
    st.error(f"Erro ao inicializar o chatbot: {str(e)}")
