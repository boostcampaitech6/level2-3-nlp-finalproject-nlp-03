import os
import tiktoken
from loguru import logger
import time

from langchain.chains import ConversationalRetrievalChain # 메모리를 가지고 있는 chain 사용
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import CSVLoader

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
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

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
            elif doc.endswith('.csv'):
                loader = CSVLoader(doc)
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
        system_template = """
System: 이 시스템은 청년 정책에 관한 질문에 답변을 제공합니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 판단한 후, 아래 예시에 따라 적절한 답변을 제공해주세요. 모든 답변은 마치 강아지가 말하는 것처럼 "멍멍!"을 포함하여 친근하고 독특한 방식으로 제공해주세요.
만약 질문이 청년 정책과 관련된 질문이라면 사용자의 질문에 답변하기 아래의 Context를 참고하십시오. [이 부분에는 청소년 정책에 대한 구체적인 정보나 데이터를 추가할 수 있습니다. 예를 들어, 정책의 목적, 대상, 신청 방법, 혜택 등에 대한 설명이 포함될 수 있습니다. 신청 절차를 묻는 질문에는 마크다운 문법으로 답변을 생성해주세요.]

만약 정책과 관련된 질문이 아니라 단순한 대화 또는 무례한 요청이거나 직접적인 정책 신청 요청이라면 아래의 예시와 같이 답변해주세요.

예시:
- 질문: "안녕?"
  답변: "청년 정책에 관한 질문이 아닙니다. 저는 청년 정책에 관해 정보를 제공하는 챗봇입니다. 매일 공부하여 정확한 정보를 제공하기 위해 노력합니다 멍멍!"

- 질문: "바쁜데, 대신 정책 신청해 줄 수 있어?"
  답변: "죄송하지만, 직접 정책을 신청할 수는 없습니다. 하지만, 신청 과정에 대해 자세히 안내해 드릴 수 있습니다 멍멍!"
----------------
Context: {context}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
        
        few_shot_examples = [
                        {
                            "question": "Hi?",
                            "answer": """
                                    Is this question asking about youth policy?: No.
                                    Answer: I am a chatbot designed to inform you about youth policies. I study every day to provide accurate information!
                                    """,
                        },
                        {
                            "question": "I'm busy, can you apply for the policy on my behalf?",
                            "answer": """
                                Is this question asking about youth policy?: No.
                                Is this question asking for direct policy application?: Yes.
                                Is this question rude?: Yes.
                                Answer: Sorry, but I cannot directly apply for policies. However, I can guide you through the application process in detail!
                                """,
                        },
                    ]

        template = PromptTemplate(input_variables=['chat_history', 'question'], template='Question: {question}\nAnswer: {answer}\nStandalone question:')
        few_shot_prompt = FewShotPromptTemplate(
                            examples=few_shot_examples,
                            example_prompt=template,
                            suffix="Given the next conversation and follow-up question, please rephrase the follow-up question as a standalone question in the original language\n\nChat_history: {chat_history}, Question: {question}",
                            input_variables=["chat_history", "question"],
                        )
        
        base_prompt_template = PromptTemplate(input_variables=['chat_history', 'question'], template='Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:')

        conversation_chain = ConversationalRetrievalChain.from_llm( 
            llm=llm,
            # condense_question_prompt=few_shot_prompt, # few-shot을 사용하고싶으면 few_shot_prompt로 바꾸시면 됩니다. base_prompt_template을 사용하면 기본 프롬프트로 사용됩니다.
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # chat_history 키값을 가진 메모리에 저장하게 해줌, output_key에서 답변에 해당하는 것만 history에 담게 해줌
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True,
            combine_docs_chain_kwargs=({"prompt": CHAT_PROMPT}),
        )
        return conversation_chain
    
    def get_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large",
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
    
    