import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import main
import server


def test_load_document_text():
    docs = main.load_document(Path('documentacao.txt'))
    assert len(docs) > 0


def test_qa_chain_response(monkeypatch):
    """QA chain should return a non-empty string."""

    # dummy classes used to avoid network calls
    class DummyEmbeddings:
        def embed_documents(self, docs):
            return [[0.0] * 3 for _ in docs]
        def embed_query(self, text):
            return [0.0] * 3

    class DummyRetriever:
        def __init__(self, docs):
            self.docs = docs
        def __call__(self, query):
            return self.docs

    class DummyDB:
        def __init__(self, docs, embeddings, persist_directory=None):
            self.docs = docs
        def persist(self):
            pass
        def as_retriever(self):
            return DummyRetriever(self.docs)

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            pass

    class DummyQA:
        def __init__(self, llm=None, retriever=None):
            pass
        def invoke(self, input_):
            return {"result": "mocked answer"}

    monkeypatch.setattr(server, "OllamaEmbeddings", lambda *a, **k: DummyEmbeddings())
    monkeypatch.setattr(server.Chroma, "from_documents", lambda docs, embeddings, persist_directory: DummyDB(docs, embeddings, persist_directory))
    monkeypatch.setattr(server, "ChatOllama", lambda *a, **k: DummyLLM())
    monkeypatch.setattr(server.RetrievalQA, "from_chain_type", lambda llm, retriever: DummyQA(llm, retriever))

    chain = server.build_chain(Path('documentacao.txt'))
    result = chain.invoke({"query": "Test?"})["result"]
    assert result
