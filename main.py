from pathlib import Path
import argparse

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple RAG pipeline using Ollama and LangChain",
    )
    parser.add_argument(
        "--document",
        default="documentacao.txt",
        help="Path to the documentation file",
    )
    parser.add_argument(
        "--question",
        default="Como cadastrar um cliente na plataforma?",
        help="Question to ask about the documentation",
    )

    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        raise FileNotFoundError(
            f"Arquivo '{doc_path}' n\u00e3o encontrado. Certifique-se de que o arquivo est\u00e1 no caminho especificado."
        )

    # 1. Carregar documentos (pode ser .txt, .md, etc.)
    loader = TextLoader(str(doc_path), encoding="utf-8")
    docs = loader.load()

    # 2. Dividir em pedaços pequenos
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Gerar embeddings com Ollama
    embeddings = OllamaEmbeddings(
        model="deepseek-r1:8b", base_url="http://localhost:11434"
    )

    # 4. Indexar os documentos em Chroma
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    db.persist()  # salva localmente

    # 5. Configurar o modelo via Ollama
    llm = ChatOllama(model="deepseek-r1:8b", base_url="http://localhost:11434")

    # 6. Criar cadeia de perguntas com recuperação (RAG)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # 7. Teste prático
    pergunta = args.question
    resposta = qa_chain.run(pergunta)

    print("\n🧠 Pergunta:", pergunta)
    print("💬 Resposta do modelo:")
    print(resposta)


if __name__ == "__main__":
    main()
