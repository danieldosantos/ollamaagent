from pathlib import Path
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import RetrievalQA

app = Flask(__name__)


def load_document(path: Path):
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def build_chain(doc_path: Path) -> RetrievalQA:
    docs = load_document(doc_path)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(
        model="deepseek-r1:8b", base_url="http://localhost:11434"
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    llm = ChatOllama(model="deepseek-r1:8b", base_url="http://localhost:11434")
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())


qa_chain = build_chain(Path("documentacao.txt"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "")
    answer = qa_chain.invoke({"query": question})["result"]
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
