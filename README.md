[README.md](https://github.com/user-attachments/files/23529484/README.md)
# Chatbot SEI Julgar

Este projeto apresenta um assistente virtual desenvolvido para apoiar o uso do módulo SEI Julgar, oferecendo respostas rápidas e contextualizadas com base no manual oficial.

O chatbot utiliza técnicas de Processamento de Linguagem Natural (PLN) em português, LangChain, modelos da Groq e a arquitetura RAG (Retrieval-Augmented Generation), permitindo buscar e interpretar trechos relevantes da documentação para orientar o usuário de forma precisa.

---

## Requisitos do Sistema

- **Python 3.11.9**  
  (Versões mais novas como 3.12 ou 3.13 podem causar erros de compatibilidade com dependências.)

- **VS Code** instalado com a extensão **Python**.
- **Git** instalado (para controle de versão e envio ao GitHub).

---

## Passo a passo de instalação

### 1. Clonar o repositório
Abra o terminal do VS Code e execute:
```bash
git clone https://github.com/seu-usuario/chatbot_sei_julgar.git
cd chatbot_sei_julgar
```

### 2. Criar o ambiente virtual

#### No **Windows (PowerShell ou Terminal do VS Code)**:
```bash
python -m venv venv
```

Ativar o ambiente:
```bash
venv\Scripts\activate
```

#### No **Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências
(Não esqueça de colocar o requirements.txt no projeto)
Com o ambiente virtual ativado:
```bash
pip install -r requirements.txt
```

### 4. Instalar o modelo SpaCy (português)

O modelo `pt_core_news_md` é necessário para o módulo de PLN:
```bash
python -m spacy download pt_core_news_md
```

> Caso sua rede bloqueie downloads diretos (como em redes institucionais), baixe manualmente via [https://github.com/explosion/spacy-models](https://github.com/explosion/spacy-models) e instale com:
> ```bash
> pip install caminho/para/pt_core_news_md.tar.gz
> ```

### 5. Configurar variáveis de ambiente

Crie um arquivo chamado `.env` na raiz do projeto e adicione:
```
GROQ_API_KEY="sua_chave_aqui"
```
Iremos deixar nossa chave para testes da disciplina:
```
GROQ_API_KEY="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"
```
> A chave é necessária para acessar o modelo da **Groq**.  
> Você pode obtê-la em: [https://console.groq.com](https://console.groq.com)

### 6. Executar o chatbot

No terminal (com o ambiente virtual ativo):
```bash
streamlit run chatbot_sei.py
```
---
O navegador abrirá automaticamente na interface do chatbot.
Primeiro carregamento do chatbot irá demorar um pouco, pois serão baixados
os modelos e criado a base do Chroma.

## Estrutura do Projeto

```
chatbot_sei_julgar/
├── chatbot_sei.py          # Código principal do assistente
├── requirements.txt        # Dependências do projeto
├── .env                    # Chave da API Groq
├── chroma_db/              # Banco vetorial local (criado automaticamente)
├── Documentação SEI Julgar - Secretaria.pdf  # Fonte de conhecimento
└── README.md               # Este arquivo
```

---

## Funcionalidades

- Processamento de linguagem natural (SpaCy + NLTK)
- Expansão semântica e normalização de texto
- Recuperação de contexto (RAG) com **ChromaDB**
- Integração com **LLM Groq**
- Interface interativa via **Streamlit**

---

## Dicas

- Se ocorrer erro ao carregar o modelo SpaCy, verifique a conexão de rede.
- Para limpar a base vetorial e recarregar os embeddings:
  ```bash
  rmdir /s /q chroma_db     # PowerShell
  rm -rf chroma_db          # Linux/Mac
  ```
- Evite fazer commit do arquivo `.env`.

---


## Autor

Desenvolvido por **Vinicius Sá** e **André Cacau**

Assistente Virtual do SEI Julgar
