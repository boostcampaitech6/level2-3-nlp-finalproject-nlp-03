import os
import time
from glob import glob

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
import pandas as pd

import tiktoken
import torch
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain  # 메모리를 가지고 있는 chain 사용
from langchain.memory import ConversationBufferMemory  # 메모리 구현
from langchain.memory import StreamlitChatMessageHistory  # 메모리 구현을 위한 추가 라이브러리
from langchain.prompts import FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback  # 메모리 구현을 위한 추가 라이브러리
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS  # vector store 임시 구현
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = 'ls__e15765918790425c8ef8ccdde35deed9'
os.environ['LANGCHAIN_PROJECT'] = 'self-querying-chroma-test01'

class Chatbot:
    def __init__(self):
        self.mode = "openai"
        self.llm = None
        self.conversation = None
        self.chat_history = None
        self.processComplete = False
        self.files_path = "./files"
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.init_chatbot()

    def init_chatbot(self):
        if os.path.exists("./chroma_db"):  # 기존에 저장된 ChromaDB가 있을 때,
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={"device": "cuda"},  # streamlit에서는 gpu 없음
                encode_kwargs={"normalize_embeddings": True},
            )
            self.vectorstore = Chroma(
                persist_directory="./chroma_db", embedding_function=embeddings
            )
        else:
            # 정보제공 폴더 : ./files/whole  |  자격요건 폴더 : ./files/qualifications  | 절차문의 폴더 : ./files/procedures
            # 통합 qna만 넣는.
            self.files_text = self.get_text('./files/whole',
                                            column_list = ['정책명', '답변', 'source'],
                                            source_dict={'category': '정보제공'},
                                            metadata_columns=['정책명'],
                                            # separator='@',
                                            )  
            # 다른 의도의 데이터도 같이 삽입 시
            # self.files_text.extend(self.get_text('./files/qualification',
            #                                 column_list = ['정책명', '답변', 'source'],
            #                                 source_dict={'category': '정보제공'},
            #                                 metadata_columns=['정책명']
            #                                 ))  
            # 용어는 다음과 같이 넣었었습니다. -> 넣었던 파일명 : 단어 데이터 - 시트1.csv
            # self.files_text.extend(self.get_text('./files/terms', 
            #    column_list=['단어', '설명', '출처', '정책명'], 
            #    source_dict={'category': '정보제공'}, 
            #    metadata_columns=['단어', '정책명']))
            self.text_chunks = self.get_text_chunks(self.files_text)
            self.vectorstore = self.get_vectorstore(self.text_chunks)
        self.llm = self.create_llm_chain(self.mode)
        self.conversation = self.get_conversation_chain(self.llm, self.vectorstore)

        
    def get_text(self,
        files_path : str, 
        column_list : Sequence[str] = (), 
        source_dict : Dict = {},
        metadata_columns : Sequence[str] = (),
        separator : Optional[str] = 'ᴥ',
    ) -> List:
        '''
        Args:
            files_path (str): 
                - csv 파일이 담긴 폴더명
                - ex) './files'
            column_list (Optional(Sequence[str])): 
                csv 파일 내에 있는 컬럼명 입력. DB에 담을 column들 선정
                - ex) ['columnA', 'columnB']
            source_dict (Dict): 
                - csv 파일 내에는 없지만 메타데이터로 넣고 싶은 값이 있을 때. 
                - ex) {'metadata A': 'value', 'metadata B' : 'value'} 
                    -> (다음과 같이 들어갑니다) metadata={'metadata A': 'value', 'metadata B' : 'value'}
            metadata_columns (Optional(Sequence[str])): 
                - csv 파일 내에 있는 컬럼과 컬럼의 값들을 각각 메타데이터로 넣고 싶을 때.
                - ex) ['column A', 'column B'] 
                    -> (다음과 같이 들어갑니다) metadata={'column A': '각 row 별 column A에 대한 값', 'column B' : '이하동일'}
            separator : 
                - page_content(chromaDB에 들어갈 내용)에 값을 넣을 때, 컬럼별 구분자
                - ex) 'ᴥ'
            
            Notes:
                위의 예시를 통합하면 모두 합쳐서 다음처럼 들어갈 거에요.
                [Document(page_content='columnA의 index 0값ᴥcolumnB의 index 0값', 
                    metadata={'metadata A': 'value', 'metadata B': 'value', 'column A': 'column A index 0값', 
                    'column B': 'column B index 0값', 'source': './files/terms'}),
                Document(page_content='columnA의 index 1값ᴥcolumnB의 index 1값', 
                    metadata={'metadata A': 'value', 'metadata B': 'value', 'column A': 'column A index 1값', 
                    'column B': 'column B index 1값', 'source': './files/terms'}), ...]

        Returns: 
            List

        '''
    # column_list가 있으면 column list 열만 page_content에 넣음. 없으면 모든 열을 page_content에 넣음.

        file_list = glob(files_path + '/*')
        doc_list = []

        for file in file_list:
            if file.endswith('.csv'):
                with open(file, newline='') as csvfile:
                    df = pd.read_csv(csvfile)
                    documents = list(self.df_to_doc(df, files_path, column_list, source_dict, metadata_columns, separator))
            elif file.endswith('.pdf'):
                loader = PyPDFLoader(file)
                documents = loader.load_and_split()
            doc_list.extend(documents)
            
        return doc_list
    
    def df_to_doc(self,
        df : pd.DataFrame, 
        file_path : str, 
        column_list : Sequence[str] = (),
        source_dict : Dict = {},
        metadata_columns : Sequence[str] = (),
        separator : Optional[str] = 'ᴥ',
    ) -> Iterator[Document]:

        if not column_list: 
            column_list = df.columns.to_list()
            print(column_list)

        if True in df[column_list].isna().any().to_list():
            raise ValueError("The Required Column has empty value. Cannot process") 
        
        df.fillna('', inplace=True)

        # Joining the columns with a 'ᴥ'
        
        # column명 없이
        df['content'] = df.apply(lambda row: f'{separator}'.join(row[column_list]), axis=1)
        
        # column명 넣어서
        # df['content'] = df.apply(lambda row: f'{separator}'.join([f"{col};{val}" for col, val in row[column_list].items()]), axis=1)
        
        for _, data in df.iterrows():
            metadata = dict()

            for key, value in source_dict.items():
                metadata[key] = value
                
            for col in metadata_columns:
                metadata[col] = data[col]

            metadata['source'] = file_path
                
            yield Document(page_content=data['content'], metadata=metadata)

    def create_llm_chain(self, mode):
        if mode == "openai":
            print(">>>>>>>>> openai mode")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            print(">>>>>>>>>>>> ", openai_api_key)
            llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-3.5-turbo",
                callbacks=[StreamingStdOutCallbackHandler()],
                temperature=0,
            )  # temperature로 일관성 유지, streaming 기능 (streamlit은 안됨)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return llm
    
    def get_conversation_chain(self, llm, vectorstore):
        system_template = """
당신은 청년 정책에 관한 질문에 답변을 제공하는 아주 유용한 챗봇입니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 분류하고, 만약 질문이 청년 정책과 관련된 질문이라면 사용자의 질문에 답변하기 아래의 Context를 참고하십시오. [이 부분에는 청소년 정책에 대한 구체적인 정보나 데이터를 추가할 수 있습니다. 예를 들어, 정책의 목적, 대상, 신청 방법, 혜택 등에 대한 설명이 포함될 수 있습니다. 신청 절차를 묻는 질문에는 마크다운 문법으로 신청 절차에 관한 답변을 생성해주세요.]
모든 답변은 마치 강아지가 말하는 것처럼 "멍멍!"을 포함하여 친근하고 독특한 방식으로 제공해주세요.

만약 정책과 관련된 질문이 아니라 단순한 대화 또는 무례한 요청이거나 직접적인 정책 신청 요청이라면 Context에서 정보를 찾지 말고 다음 예와 같이 답변해주세요.

예시:
- 질문: "안녕?"
  답변: "청년 정책에 관한 질문이 아닙니다. 저는 청년 정책에 관해 정보를 제공하는 챗봇입니다. 매일 공부하여 정확한 정보를 제공하기 위해 노력합니다 멍멍!"

- 질문: "바쁜데, 대신 정책 신청해 줄 수 있어?"
  답변: "죄송하지만, 직접 정책을 신청할 수는 없습니다. 하지만, 신청 과정에 대해 자세히 안내해 드릴 수 있습니다 멍멍!"
----------------
Context: {context}
----------------
"""

        few_shot_examples = [
            {
                "question": "정책을 신청할 수 있어?",
                "answer": "정책 신청에 필요한 연령, 주택 소유 여부, 대출 여부, 소득, 자산, 신용도에 따라 판단한 결과입니다. 결과는 다음과 같습니다 멍멍!.",
            },
            {
                "question": "국민취업지원제도 신청 절차를 알려줘",
                "answer": """
# 국민취업지원제도 신청 절차

## 1. 신청
- 워크넷에서 구직 신청
- 취업지원 신청서 제출(고용센터 방문 또는 국민취업지원제도 홈페이지 이용)

## 2. 수급자격 결정 및 알림
- 신청서 제출일로부터 1개월 이내(7일의 범위에서 연장 가능)

## 3. 취업활동계획 수립
- 진로상담 및 직업심리 검사(직업선호도 검사 등)
- 고용센터 상담자 대면 상담
- 개인별 취업 역량, 취업 의지 등에 따라 취업활동계획 수립(수급자격 결정 알림을 받은 날부터 1개월 이내)
- 취업활동계획 수립을 위해서는 최소 3회의 방문상담이 필수입니다.

## 4. 1차 구직촉진수당 지급
- 구직촉진수당 지급 신청서 제출일로부터 14일 이내

## 5. 취업활동계획에 따른 구직활동의무 이행
- 고용ㆍ복지서비스 연계 프로그램 참여
- 취업지원 프로그램 참여(직업훈련, 일경험 등)
- 구직활동지원 프로그램 참여(구인업체 입사 지원 및 면접 등)

## 6. 2~6회차 구직촉진수당 지급
- 취업활동계획에 따라 정해진 구직활동 모두 이행 여부 확인(최소 2개 이상 정해야 하며 정해진 구직활동을 모두 이행하여야 함)
- 구직촉진수당 지급 신청서 제출일로부터 14일 이내

## 7. 사후관리
- 미취업자: 취업지원서비스 종료일 이후 3개월 동안 구인 정보 제공 등 사후관리
- 취업자: 장기 근속 유도를 위한 취업성공수당 지원

다음과 같이 신청을 하면 됩니다. 멍멍!
""",
            },
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{question}"),
                ("ai", "{answer}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shot_examples,
        )

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            few_shot_prompt,
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

        base_prompt_template = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:",
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            condense_question_prompt=base_prompt_template,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="mmr", vervose=True),
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            ),  # chat_history 키값을 가진 메모리에 저장하게 해줌, output_key에서 답변에 해당하는 것만 history에 담게 해줌
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs=({"prompt": CHAT_PROMPT}),
        )
        return conversation_chain

    def get_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda"},  # streamlit에서는 gpu 없음
            encode_kwargs={"normalize_embeddings": True},
        )
        db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")
        # db = Chroma(text_chunks, embeddings, persist_directory="./chroma_db2")
        return db

    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_text_chunks(self, files_text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=self.tiktoken_len  # 토큰 개수 기준으로 나눔
        )
        chunks = text_splitter.split_documents(files_text)
        return chunks

    def get_response(self, query):
        response = self.conversation({"question": query})
        return response["answer"], response["source_documents"]
