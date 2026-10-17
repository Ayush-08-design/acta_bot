import time
from pathlib import Path

from langchain_classic.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=40
        )

    def chunk(self, docs: list[Document], thread_id: str) -> list[Document]:
        docs = self.splitter.split_documents(docs)
        for doc in docs:  ## Adding metadata
            doc.metadata["thread_id"] = thread_id
            expiration_time = int(time.time()) + (30 * 24 * 3600)
            doc.metadata["expiration_time"] = expiration_time
        return docs


class Loader:
    def __init__(self, file_path: Path):
        self.loader = PyPDFLoader(file_path=file_path)

    def load(self) -> list[Document]:
        doc = self.loader.load()
        return doc