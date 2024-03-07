import os
import sys
from assistant import AssistantChatbot
from pypdf import PdfMerger

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)    

def query_classification(bot, query, file_id):
    thread_id = bot.create_new_thread()
    assistant_id = bot.assistant.id

    query = f"""
사용자의 질문은 다음과 같습니다 : {query}
문서의 내용은 다음과 같습니다:  {file_id}
당신은 사용자의 질문과 문서의 질문을 비교하여 사용자의 질문이 문서와 연관되어 청년 정책과 연관된 질문이면 True로 답변을 하고 문서와 연관되지 않고 청년 정책과 상관없는 일상 대화라면 False를 반환해주기 바랍니다. 
"""

    bot.send_message(
        assistant_id=assistant_id, 
        thread_id=thread_id, 
        user_message=query, 
        file_ids=[file_id]
    )
    query_class = bot.threads_dict[thread_id][0]
    return query_class

def run(query: str):
    file_path = current_path + "/files/"
    files = os.listdir(file_path)
    files = [f"{file_path}{file}" for file in files]
    print(files[1])

    donghangbot = AssistantChatbot()
    donghangbot.load_assistant(id='asst_1BCYQqLoUFtHriaXZXvz8b4X')
    file_id = donghangbot.upload_file(files[1])
    query_class = query_classification(donghangbot, query, file_id)
    print(query_class)
    source_document = None

    return query_class, source_document



if __name__=='__main__':
    run("안녕?") 