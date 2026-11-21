from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.schema import Document
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

embedding_model = OllamaEmbeddings(model="nomic-embed-text:137m-v1.5-fp16")


class VectorStore:
    def __init__(self):
        self.dir = "./chroma"
        self.embedding_model = embedding_model

    # collection name is related to user_id [one collection per user]
    def _load_store(self, user_id: str, persist_directory: str = "./chroma"):
        collection_name = f"user_{user_id}_collection"
        vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        return vector_store

    async def add_docs(
        self, user_id: str, docs: list[Document]
    ):  # these docs have metadata of a particular thread of a user
        vector_store = self._load_store(user_id)
        await vector_store.aadd_documents(docs)

    async def search_docs(self, query: str, user_id: str) -> list[Document]:
        """Search store by a collection name by default in most recently uploaded pdf"""
        store = self._load_store(user_id)
        results = await store.asimilarity_search(query, k=3)
        return results

    def fetch_all_docs(self, user_id: str) -> list[Document]:
        store = self._load_store(user_id)
        store_result = store._collection.get(include=["documents"])
        all_docs = store_result["documents"]
        return all_docs


def rerank_docs(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    _model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    _reranker = CrossEncoderReranker(_model, top_n)
    if not docs:
        return []
    return list(_reranker.compress_documents(documents=docs, query=query))
