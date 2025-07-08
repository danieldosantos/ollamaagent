import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

DOCUMENT_PATH = "documentacao.txt"

if not os.path.exists(DOCUMENT_PATH):
    raise FileNotFoundError(
        f"Arquivo '{DOCUMENT_PATH}' n\u00e3o encontrado. Certifique-se de que o arquivo est\u00e1 na raiz do projeto."
    )

# 1. Carregar documentos (pode ser .txt, .md, etc.)
loader = TextLoader("documentacao.txt", encoding='utf-8')
docs = loader.load()

# 2. Dividir em pedaços pequenos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Gerar embeddings com Ollama
embeddings = OllamaEmbeddings()

# 4. Indexar os documentos em Chroma
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
db.persist()  # salva localmente

# 5. Configurar o modelo via Ollama
llm = ChatOllama(model="deepseek-r1:8b", base_url="http://localhost:11434")

# 6. Criar cadeia de perguntas com recuperação (RAG)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 7. Teste prático
pergunta = "Como cadastrar um cliente na plataforma?"
resposta = qa_chain.run(pergunta)

print("\n🧠 Pergunta:", pergunta)
print("💬 Resposta do modelo:")
print(resposta)
