
#  Assistente Virtual do SEI Julgar

Este projeto implementa um chatbot especializado no módulo SEI
Julgar, utilizando **RAG (Retrieval-Augmented Generation)**,
**LangChain**, **ChromaDB** e modelos **Groq**.\
Ele auxilia usuários de secretarias e gabinetes a localizar informações
do manual oficial de forma rápida e objetiva.

------------------------------------------------------------------------

##  Tecnologias Utilizadas

-   **Python 3.11.9**
-   **Streamlit** --- Interface do chatbot
-   **LangChain** --- Pipeline de consultas
-   **ChromaDB** --- Armazenamento vetorial local
-   **HuggingFace Embeddings (bge-m3)**
-   **PDF Loader (PyPDF2)**
-   **Groq LLMs** (LLaMA 3.x, Mixtral)

------------------------------------------------------------------------

##  Estrutura do Projeto

    chatbot_sei/
    ├── chatbot_sei.py                     # Código principal
    ├── Documentação SEI Julgar - Secretaria.pdf
    ├── chroma_db/                         # Base vetorial (gerada automaticamente)
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

##  Instalação

### 1️ Clonar o repositório

``` bash
git clone https://github.com/seu_usuario/chatbot_pln.git
cd chatbot_pln
```

### 2️ Criar o ambiente virtual

**Windows**

``` bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3️ Instalar dependências

``` bash
pip install -r requirements.txt
```
A instalação das dependências demora um tempo consideravél, dado a instalação de várias bibliotecas.

### 4️ Criar arquivo `.env`

``` env
GROQ_API_KEY=suachaveaqui
```

Obtenha sua chave em: https://console.groq.com

------------------------------------------------------------------------

##  Executando o Chatbot

``` bash
streamlit run chatbot_sei.py
```

A interface abrirá automaticamente no navegador.

------------------------------------------------------------------------

##  Como funciona?

1.  O PDF da Documentação do SEI Julgar é carregado.
2.  O texto é dividido em *chunks*.
3.  Os embeddings são gerados usando bge-m3.
4.  A ChromaDB indexa e armazena localmente.
5.  O chatbot usa RAG para:
    -   reformular perguntas
    -   buscar trechos relevantes
    -   responder apenas sobre o SEI Julgar

------------------------------------------------------------------------

##  Limpando a base vetorial

Se quiser reconstruir a indexação:

``` bash
rmdir /s /q chroma_db  # Windows
rm -rf chroma_db       # Linux/Mac
```

------------------------------------------------------------------------

## Para rodar o projeto no notebook Jupyter 
1) Baixe o arquivo ProjetoFinal_PLN.ipynb.
2) Baixe Documentação SEI Julgar - Secretaria.pdf
3) Baixe o arquivo chatbot_sei.py
4) Irá precisar da chave API Groq https://groq.com/
5) Siga os passos presentes no notebook e importe os arquivos.
   
##  Autores

**Vinicius Sá e André Cacau**\
Assistente Virtual do SEI Julgar.
