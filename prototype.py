import os
import streamlit as st
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

mode = "solar"

def main():
    st.set_page_config(
    page_title="동행 챗봇", # 탭에 타이틀과 아이콘 설정
    page_icon="👨‍👩‍👧‍👦")

    st.title("👨‍👩‍👧‍👦 동행 챗봇 prototype")

    # session_state 초기화 - 애플리케이션 재실행 시 유지시키기 위해서
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)        
        process = st.button("Process")
    
    if process: # 버튼 누르면
        files_text = get_text(uploaded_files) # Documnet Loader 적용
        text_chunks = get_text_chunks(files_text) # Text Splitter 적용
        vetorestore = get_vectorstore(text_chunks) # Text Embedding 적용

        llm = create_llm_chain(mode)
        st.session_state.conversation = get_conversation_chain(llm, vetorestore) # Retrieval - chain 구성

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "청년 정책에 대해서 궁금한 사항을 질문해주세요!!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # chat message container 적용 -> 아이콘 적용
            st.write(message["content"]) # markdown?ㄹ

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    query = st.chat_input("질문을 입력해주세요.")
    if query:
        start = time.time()
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"): # user message 표시
            st.write(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): # 로딩할 때 돌아가는 부분 구현
                result = chain({"question": query}) # chain에 query 입력
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history'] # 채팅 history 저장
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response) # 결과 값 표시
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].page_content, help=source_documents[0].metadata['source'])

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        end = time.time()
        print(f">>>>>>>>>>>>>> {end - start:.5f} sec")
    

def create_llm_chain(mode):
    '''
    어떤 모델을 쓸 지 결정하는 함수
    '''
    if mode == "openai": 
        print('>>>>>>>>> openai mode')
        openai_api_key = os.getenv("OPENAI_API_KEY")
        print('>>>>>>>>>>>> ', openai_api_key)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo', callbacks=[StreamingStdOutCallbackHandler()], temperature=0) # temperature로 일관성 유지, streaming 기능 (streamlit은 안됨)
    elif mode == "solar": 
        print('>>>>>>>> solar mode')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_name = "LDCC/LDCC-SOLAR-10.7B"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        print('-----------------\n', model, '--------------------\n') # # Linear4bit로 양자화 된 것을 확인

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            return_full_text=True,
            max_new_tokens=500,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm

def tiktoken_len(text):
    '''
    token 개수를 세는 함수
    '''
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    '''
    업로드된 파일을 텍스트로 변환하는 함수
    각 파일 형식에 따라서 구분됨
    '''
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file: # streamlit의 내부 서버에서 임시적으로 사용하기 위해서 파일 생성
            file.write(doc.getvalue()) # 원래 doc 의 내용을 file에 적음
            logger.info(f"Uploaded {file_name}") # 로깅

        # document 객체 생성
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    '''
    RecursiveCharacterTextSplitter을 사용해서 document 객체를 chunk로 나누는 함수
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len # 토큰 개수 기준으로 나눔
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    '''
    만들어진 chunk를 벡터화
    '''
    embeddings = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'}, # streamlit에서는 gpu 없음
                    encode_kwargs={'normalize_embeddings': True}
                )  
    # db = FAISS.from_documents(text_chunks, embeddings)
    db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")
    # db2._collection.add(ids=["gyun"], embeddings=embeddings, documents=text_chunks) # error
    return db

def get_conversation_chain(llm, vetorestore):
    '''
    chain 생성
    '''
    conversation_chain = ConversationalRetrievalChain.from_llm( 
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # chat_history 키값을 가진 메모리에 저장하게 해줌, output_key에서 답변에 해당하는 것만 history에 담게 해줌
            get_chat_history=lambda h: h, # 메모리가 들어온 그대로 chat history에 넣음
            return_source_documents=True, # 참고 문서 출력
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
