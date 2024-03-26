from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import requests
from typing import List


class ColBERTRetriever(BaseRetriever):
    url: str
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        params = {"query": query, "k": self.k}
        response = requests.get(self.url, params=params)
        return [Document(page_content=doc['text']) for doc in response.json()["topk"]]