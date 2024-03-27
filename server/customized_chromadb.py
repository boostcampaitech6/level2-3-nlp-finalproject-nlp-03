import chromadb
from chromadb.config import Settings

from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

HOST = '175.45.203.113'
PORT = '1123'

class CustomizedChromaDB:

    class MyEmbeddingFunction(EmbeddingFunction):
        def __init__(self, embed_func):
            super().__init__()
            self.embed_func = embed_func
        
        def __call__(self, input: Documents) -> Embeddings:
            embedding_model = self.embed_func
            embeddings = embedding_model.embed_documents(input)
            return embeddings

    def __init__(self, langcahin_embed_func):
        """
            Example:
                HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large",
                    model_kwargs={"device": "cuda"},  # streamlit에서는 gpu 없음
                    encode_kwargs={"normalize_embeddings": True},
                )
        
        """
        self.lanchain_embed_func = langcahin_embed_func
        self.chroma_embed_func = self.MyEmbeddingFunction(langcahin_embed_func)
        
        self.client = chromadb.HttpClient(
            host=HOST,
            port=PORT,
            settings=Settings(allow_reset=True),
        )
        self.collection = None
        self.collection_name = None
        self.langchain_db = None

    # client 생성
    def get_client(self):
        return self.client
    
    # collection 생성 및 관리
    def get_collection(self, collection_name : str):
        if self.client is None:
            raise ValueError('You have to connect client first')
        
        self.collection = self.client.get_collection(name=collection_name)
        self.collection_name = collection_name
        return self.collection

    def create_collection(self, collection_name : str):
        if self.client is None:
            raise ValueError('You have to connect client first')
        
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.chroma_embed_func)
        self.collection_name = collection_name

        return self.collection
        

    # 데이터 추가
    def add_data(self, docs : Sequence[Document] = ()):
        if self.collection is None:
            raise ValueError('You have to create or get collection first by using "create_collection()" or "get_collection()"')
        
        ids = [str(i) for i in range(self.collection.count(), self.collection.count()+len(docs)+1)]

        for doc, id in zip(docs, ids):
            self.collection.add(
                ids=id,
                metadatas=doc.metadata,
                documents=doc.page_content
            )

    # langchain용 chromadb 생성. langchain 형식에 맞게 바꿔주는 객체 정도로 파악.
    def langchain_chroma(self):
        self.langchain_db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.lanchain_embed_func,
        )
        return self.langchain_db
        

    ## chroma DB 함수 모음

    # client.list_collections() : collection 리스트 목록 return
    # collection = client.create_collection("testname") : 새 collection 생성
    # collection = client.get_collection("testname") : 기존 collection 가져오기
    # collection = client.get_or_create_collection("testname") : 이미 존재하는 collection이면 생성, 아니면 가져오기
    # client.delete_collection("testname") : collection 삭제
    # client.heartbeat() : returns timestamp to check if service is up
    # client.reset() : resets entire database - this *cant* be undone!

    # collection.modify(name) : collection의 이름 변경
    
    
    ## collection 내 데이터 관리

    # collection.get() : collection 내의 모든 레코드 return
    # collection.peek() : dataframe.head() 와 같은 기능. 상위 4개 레코드 return
    # collection.count() : collection 내의 레코드 개수 return

    # collection.update(ids:list, documents=list, metadatas:list)
    # collection.upsert(ids:list, documents=list, metadatas:list) : 있는 값이면 update, 없는 값이면 add하는 기능
    # collection.delete(ids:list)
    

    # https://docs.trychroma.com/api-reference#methods-on-collection-1
    # https://docs.trychroma.com/reference/Client