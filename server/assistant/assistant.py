import os
from openai import OpenAI
import json
import time
from dotenv import load_dotenv
from pprint import pprint


class AssistantChatbot:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        self.assistant = None
        self.thread = None
        self.threads_dict = {}
#         self.system_template = """
# 당신은 청년 정책에 관한 질문에 답변을 제공하는 아주 유용한 챗봇입니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 분류하여 정책과 관련된 답변을 해주세요.
# """
        self.system_template = """
당신은 청년 정책에 관한 질문에 답변을 제공하는 아주 유용한 챗봇입니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 분류하고, 만약 질문이 청년 정책과 관련된 질문이라면 사용자의 질문에 답변하기 위해 아래의 Context를 참고하십시오. [이 부분에는 청소년 정책에 대한 구체적인 정보나 데이터를 추가할 수 있습니다. 예를 들어, 정책의 목적, 대상, 신청 방법, 혜택 등에 대한 설명이 포함될 수 있습니다. 신청 절차를 묻는 질문에는 마크다운 문법으로 신청 절차에 관한 답변을 생성해주세요.]
모든 답변은 마치 강아지가 말하는 것처럼 "멍멍!"을 포함하여 친근하고 독특한 방식으로 제공해주세요.

만약 정책과 관련된 질문이 아니라 단순한 대화 또는 무례한 요청이거나 직접적인 정책 신청 요청이라면 Context에서 정보를 찾지 말고 다음 예와 같이 답변해주세요.

예시:
- 질문: "안녕?"
  답변: "청년 정책에 관한 질문이 아닙니다. 저는 청년 정책에 관해 정보를 제공하는 챗봇입니다. 매일 공부하여 정확한 정보를 제공하기 위해 노력합니다 멍멍!"

- 질문: "바쁜데, 대신 정책 신청해 줄 수 있어?"
  답변: "죄송하지만, 직접 정책을 신청할 수는 없습니다. 하지만, 신청 과정에 대해 자세히 안내해 드릴 수 있습니다 멍멍!"
"""
    
    def create_assistant(
        self,
        name: str = '동행챗봇',
        instructions: str = None,
        tools: list[dict] = [{"type": "retrieval"}],
        model: str = "gpt-4-1106-preview"
    ) -> str:
        if instructions is None:
            instructions = self.system_template

        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model
        )
        self.show_json(assistant)
        return assistant.id
    
    def load_assistant(
        self,
        id: str = 'asst_1BCYQqLoUFtHriaXZXvz8b4X'
    ):
        self.assistant = self.client.beta.assistants.retrieve(id)
        return self.assistant.id

    def update_assistant(
        self,
        assistant_id: str = 'asst_1BCYQqLoUFtHriaXZXvz8b4X',
        tools: list[dict] = [
            # {"type": "code_interpreter"},
            {"type": "retrieval"},
            # {"type": "function", "function": lambda x: None},
        ],
        instructions: str = ""
    ) -> str:
        self.assistant = self.client.beta.assistants.update(
            assistant_id,
            tools=tools,
            instructions=instructions,
        )
        # self.show_json(self.assistant)
        return self.assistant.id
        
    
    def create_new_thread(self, id: str=None):
        # 새로운 스레드를 생성합니다.
        if id:
            thred = self.client.beta.thread.retrieve(id)
        else:
            thread = self.client.beta.threads.create()
        self.show_json(thread)
        self.thread = thread
        self.threads_dict[self.thread.id] = []
        return thread.id
    
    def upload_file(self,fp_path: str):
        # TODO: add a check here for filesize limit

        file = self.client.files.create(
            file=open(fp_path, 'rb'),
            purpose='assistants'
        )
        return file.id 

    def delete_file(self, id: str):
        self.client.files.delete(id)

    def show_json(self, obj):
        # obj를 JSON 형태로 변환한 후 들여쓰기를 적용하여 출력합니다.
        pprint(json.loads(obj.model_dump_json()))

    # 반복문에서 대기하는 함수
    def wait_on_run(self, run, thread_id):
        while run.status == "queued" or run.status == "in_progress":
            # 실행 상태를 최신 정보로 업데이트합니다.
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,
            )
            time.sleep(0.5)

        if run.status == 'completed':
            print('>>>>>>> assistant run status: ', run.status)
        else:
            raise "Error in assistant runnig"
        return run

    def send_message(
            self, 
            assistant_id: str, 
            thread_id: str, 
            user_message: str,
            file_ids: list[str] = None,
        ):
        if file_ids:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_message,
                file_ids=file_ids
            )
        else:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_message
            )

        # 스레드를 실행합니다.
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        # 실행이 완료될 때까지 대기합니다.
        run = self.wait_on_run(run, thread_id)
        # self.show_json(run)

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )

        msg_list = ['  \n'.join([content.text.value for content in msg.content]) for msg in messages.data]
        self.threads_dict[self.thread.id] = msg_list
        # print(msg_list)

        return msg_list
    
    def exit(self):
        for thread_id in self.threads_dict.keys():
            self.client.beta.threads.delete(thread_id)
    

if __name__=="__main__":
    chatbot = AssistantChatbot()
    chatbot.load_assistant()
    thread_id = chatbot.create_new_thread()
    assistant_id = chatbot.assistant.id
    chatbot.send_message(assistant_id, thread_id, "만나서 반가워")
    chatbot.exit()
