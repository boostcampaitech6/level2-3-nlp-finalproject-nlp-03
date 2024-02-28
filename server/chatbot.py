import os
import tiktoken
from loguru import logger
import time

from langchain.chains import ConversationalRetrievalChain # 메모리를 가지고 있는 chain 사용
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.memory import ConversationBufferMemory # 메모리 구현
from langchain_community.vectorstores import FAISS # vector store 임시 구현
from langchain_community.vectorstores import Chroma

from langchain_community.callbacks import get_openai_callback # 메모리 구현을 위한 추가 라이브러리
from langchain.memory import StreamlitChatMessageHistory # 메모리 구현을 위한 추가 라이브러리

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from glob import glob

class Chatbot:
    def __init__(self):
        self.mode = "openai"
        self.llm = None
        self.conversation = None
        self.chat_history = None
        self.processComplete = False
        self.files_path = './files'
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.init_chatbot()

    def init_chatbot(self):
        self.files_text = self.get_text(self.files_path)
        self.text_chunks = self.get_text_chunks(self.files_text)
        self.vectorstore = self.get_vectorstore(self.text_chunks)
        self.llm = self.create_llm_chain(self.mode)
        self.conversation = self.get_conversation_chain(self.llm, self.vectorstore)

    def get_text(self, files_path):
        file_list = glob(files_path + '/*')
        doc_list = []

        for doc in file_list:
            if doc.endswith('.pdf'):
                loader = PyPDFLoader(doc)
                documents = loader.load_and_split()
            elif doc.endswith('.docx'):
                loader = Docx2txtLoader(doc)
                documents = loader.load_and_split()
            elif doc.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(doc)
                documents = loader.load_and_split()
            doc_list.extend(documents)

        return doc_list

    def create_llm_chain(self, mode):
        if mode == "openai":
            print('>>>>>>>>> openai mode')
            openai_api_key = os.getenv("OPENAI_API_KEY")
            print('>>>>>>>>>>>> ', openai_api_key)
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo', callbacks=[StreamingStdOutCallbackHandler()], temperature=0) # temperature로 일관성 유지, streaming 기능 (streamlit은 안됨)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return llm
    
    def get_conversation_chain(self, llm, vectorstore):
        conversation_chain = ConversationalRetrievalChain.from_llm( 
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # chat_history 키값을 가진 메모리에 저장하게 해줌, output_key에서 답변에 해당하는 것만 history에 담게 해줌
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )
        return conversation_chain
    
    def get_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cuda'}, # streamlit에서는 gpu 없음
                    encode_kwargs={'normalize_embeddings': True}
                )
        db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")
        return db
    
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_text_chunks(self, files_text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self.tiktoken_len # 토큰 개수 기준으로 나눔
        )
        chunks = text_splitter.split_documents(files_text)
        return chunks
    
    def get_response(self, query):
        response = self.conversation({"question": query})
        return response['answer'], response['source_documents']
    
    