from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from chatbot import Chatbot

app = FastAPI()

class Request(BaseModel):
    query: str
    intent: str

# CORS 미들웨어를 추가하여 모든 도메인에서의 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/query")
async def read_query(request: Request):
    query = request.query
    intent = request.intent
    chatbot.set_intent(intent)
    response, source_document = chatbot.get_response(query)
    return {"response": response,
            "source_document": source_document[0].page_content, 
            "help": source_document[0].metadata['source']}

chatbot = Chatbot()

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)