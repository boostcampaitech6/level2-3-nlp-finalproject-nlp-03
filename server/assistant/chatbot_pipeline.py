import os
import sys
from assistant import AssistantChatbot
from pypdf import PdfMerger
from glob import glob

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)  
vectorstore = None


def query_classification(bot, query, document):
    thread_id = bot.create_new_thread()
    assistant_id = bot.assistant.id

    query = f"""
사용자의 질문은 다음과 같습니다 : {query}
문서의 내용은 다음과 같습니다:  {document}
당신은 사용자의 질문과 문서를 비교하여 사용자의 질문이 문서와 연관되어 청년 정책과 연관된 질문이면 True로 답변을 하고 문서와 연관되지 않고 청년 정책과 상관없는 일상 대화라면 False를 반환해주기 바랍니다. 
일상 대화의 예시는 다음과 같습니다.
예시:
    - 안녕?
    - 반가워
    - 너는 뭘 하는 챗봇이니?
다음과 같은 예시의 일상 대화는 False로 대답하세요.
긴 문장으로 대답하지 말고 오로지 True 또는 False로만 대답하세요.
"""

    bot.send_message(
        assistant_id=assistant_id, 
        thread_id=thread_id, 
        user_message=query, 
        # file_ids=[file_id]
    )
    query_class = bot.threads_dict[thread_id][0]
    return query_class

def make_answer(bot, query, doc):
    thread_id = bot.create_new_thread()
    assistant_id = bot.assistant.id
    query = f"""
제공된 문서를 바탕으로 질문 {query} 에 대한 답을 하기 바랍니다.
------
문서 : {doc}
"""
    bot.send_message(
        assistant_id=assistant_id, 
        thread_id=thread_id, 
        user_message=query, 
        # file_ids=[file_id]
    )
    answer = bot.threads_dict[thread_id][0]
    return answer

def get_text(files_path):
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
                loader = CSVLoader(doc, csv_args={
                        'delimiter': ',',
                        'quotechar': '"',}
                        )
                documents = loader.load_and_split()
            doc_list.extend(documents)

        return doc_list

def tiktoken_len(text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

def get_text_chunks(files_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len # 토큰 개수 기준으로 나눔
    )
    chunks = text_splitter.split_documents(files_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cuda'}, # streamlit에서는 gpu 없음
                encode_kwargs={'normalize_embeddings': True}
            )
    db = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_db")
    return db

def get_chromadb(query):
    files_path = './files'

    if os.path.exists("./chroma_db"):  # 기존에 저장된 ChromaDB가 있을 때,
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda"},  # streamlit에서는 gpu 없음
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db", embedding_function=embeddings
        )
    else:
        files_text = get_text(files_path)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.75}
    )

    document = retriever.get_relevant_documents(query)
    print(document)
    return document

def run(query: str):
    file_path = current_path + "/files/"
    files = os.listdir(file_path)
    files = [f"{file_path}{file}" for file in files]
    print(files[1])

    document = get_chromadb(query)

    donghangbot = AssistantChatbot()
    donghangbot.load_assistant(id='asst_1BCYQqLoUFtHriaXZXvz8b4X')
    # donghangbot.update_assistant(instructions=donghangbot.system_template)

    donghangbot.show_json(donghangbot.assistant)

    # file_id = donghangbot.upload_file(files[1])
    # policy_query = query_classification(donghangbot, query, document)
    policy_query = 'True'
    print(policy_query)

    if policy_query == 'True':
        # donghangbot.update_assistant(
        #     assistant_id='asst_1BCYQqLoUFtHriaXZXvz8b4X',
        #     tools=[{ "type": "retrieval"}],
        #     instructions=system_prompt
        # )
        # donghangbot.show_json(donghangbot.assistant)
        answer = make_answer(donghangbot, query, document)
        print(answer)
    else:
        answer = None
        
    source_document = None

    # donghangbot.delete_file(file_id)
    donghangbot.exit()

    return answer, source_document



if __name__=='__main__':
    run("국민취업지원제도가 뭐야?") 
    # run("안녕?") 