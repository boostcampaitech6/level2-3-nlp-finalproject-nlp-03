import os
import sys
from assistant import AssistantChatbot
from pypdf import PdfMerger

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)  

system_prompt = """
당신은 청년 정책에 관한 질문에 답변을 제공하는 아주 유용한 챗봇입니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 분류하고, 만약 질문이 청년 정책과 관련된 질문이라면 사용자의 질문에 답변하기 위해 아래의 Context를 참고하십시오. [이 부분에는 청소년 정책에 대한 구체적인 정보나 데이터를 추가할 수 있습니다. 예를 들어, 정책의 목적, 대상, 신청 방법, 혜택 등에 대한 설명이 포함될 수 있습니다. 신청 절차를 묻는 질문에는 마크다운 문법으로 신청 절차에 관한 답변을 생성해주세요.]
모든 답변은 마치 강아지가 말하는 것처럼 "멍멍!"을 포함하여 친근하고 독특한 방식으로 제공해주세요.

만약 정책과 관련된 질문이 아니라 단순한 대화 또는 무례한 요청이거나 직접적인 정책 신청 요청이라면 Context에서 정보를 찾지 말고 다음 예와 같이 답변해주세요.

예시:
- 질문: "안녕?"
  답변: "청년 정책에 관한 질문이 아닙니다. 저는 청년 정책에 관해 정보를 제공하는 챗봇입니다. 매일 공부하여 정확한 정보를 제공하기 위해 노력합니다 멍멍!"

- 질문: "바쁜데, 대신 정책 신청해 줄 수 있어?"
  답변: "죄송하지만, 직접 정책을 신청할 수는 없습니다. 하지만, 신청 과정에 대해 자세히 안내해 드릴 수 있습니다 멍멍!"
"""

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

def make_answer(bot, query, file_id):
    thread_id = bot.create_new_thread()
    assistant_id = bot.assistant.id
    query = f"""
당신은 제공된 문서 {file_id} 를 바탕으로 질문 {query} 에 대한 답을 하기 바합니다.
"""
    bot.send_message(
        assistant_id=assistant_id, 
        thread_id=thread_id, 
        user_message=query, 
        file_ids=[file_id]
    )
    answer = bot.threads_dict[thread_id][0]
    return answer

def run(query: str):
    file_path = current_path + "/files/"
    files = os.listdir(file_path)
    files = [f"{file_path}{file}" for file in files]
    print(files[1])

    donghangbot = AssistantChatbot()
    donghangbot.load_assistant(id='asst_1BCYQqLoUFtHriaXZXvz8b4X')

    donghangbot.show_json(donghangbot.assistant)

    file_id = donghangbot.upload_file(files[1])
    policy_query = query_classification(donghangbot, query, file_id)
    print(policy_query)

    if policy_query:
        donghangbot.update_assistant(
            assistant_id='asst_1BCYQqLoUFtHriaXZXvz8b4X',
            tools=[{ "type": "retrieval"}],
            instructions=system_prompt
        )
        donghangbot.show_json(donghangbot.assistant)
        answer = make_answer(donghangbot, query, file_id)
        print(answer)
        
    source_document = None

    donghangbot.delete_file(file_id)
    donghangbot.exit()

    return answer, source_document



if __name__=='__main__':
    run("국민취업지원제도가 뭐야?") 