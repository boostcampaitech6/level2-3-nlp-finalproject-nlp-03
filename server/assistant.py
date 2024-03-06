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
        self.system_template = """
당신은 청년 정책에 관한 질문에 답변을 제공하는 아주 유용한 챗봇입니다. 질문을 분석하여, 질문이 청년 정책에 관한 것인지, 단순한 대화인지 분류하여 정책과 관련된 답변을 해주세요.
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
    
    def set_assistant(
        self,
        id: str = 'asst_1BCYQqLoUFtHriaXZXvz8b4X'
    ):
        self.assistant = self.client.beta.assistants.retrieve(id)
    
    def create_new_thread(self, id: str=None):
        # 새로운 스레드를 생성합니다.
        if id:
            thred = self.client.beta.thread.retrieve(id)
        else:
            thread = self.client.beta.threads.create()
        self.show_json(thread)
        self.thread = thread
        self.threads_dict[self.thread.id] = []
        return thread
    
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

    import time


    # 반복문에서 대기하는 함수
    def wait_on_run(self, run, thread_id):
        while run.status == "queued" or run.status == "in_progress":
            # 실행 상태를 최신 정보로 업데이트합니다.
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run


    def submit_message(self, assistant_id, thread_id, user_message):
        # 스레드에 종속된 메시지를 '추가' 합니다.
        self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=user_message
        )
        # 스레드를 실행합니다.
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        return run


    def get_answer(self, thread_id):
        # 스레드에 종속된 메시지를 '조회' 합니다.
        return self.client.beta.threads.messages.list(thread_id=thread_id, order="asc")


    def print_message(self, response):
        for res in response:
            print(f"[{res.role.upper()}]\n{res.content[0].text.value}\n")


    def ask(self, assistant_id, thread_id, user_message):
        run = self.submit_message(
            assistant_id,
            thread_id,
            user_message,
        )
        # 실행이 완료될 때까지 대기합니다.
        run = self.wait_on_run(run, thread_id)
        self.print_message(self.get_answer(thread_id).data[-2:])
        return run
    
    def exit(self):
        for thread_id in self.threads_dict.keys():
            self.client.beta.threads.delete(thread_id)

    def get_response(self, query: str):
        response = self.conversation({"question": query})
        return response['answer'], response['source_documents']
    

if __name__=="__main__":
    chatbot = AssistantChatbot()
    chatbot.set_assistant()
    thread_id = chatbot.create_new_thread().id
    assistant_id = chatbot.assistant.id
    chatbot.ask(assistant_id, thread_id, "만나서 반가워")
    chatbot.exit()
