from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from chatbot import Chatbot

## ragas
import os, csv
import pandas as pd
from datetime import datetime
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_entity_recall,
    context_relevancy,
    answer_similarity, 
    answer_correctness,
)

app = FastAPI()

class Request(BaseModel):
    query: str

# CORS 미들웨어를 추가하여 모든 도메인에서의 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

async def evaluate_ragas(query, response, source_document):

    ## ragas 평가
    ymd = datetime.today().strftime("%y%m%d")
    PATH = "/home/ssg/level2-3-nlp-finalproject-nlp-03/server/evals"
    filename = f"eval_{ymd}.csv"
    full_path = os.path.join(PATH, filename)

    questions = [query]
    answers = [response]
    contexts = []

    contexts.append([source_document[i].page_content for i in range(len(source_document))])

    data = {"question": questions, "answer": answers, "contexts": contexts}

    # 데이터 타입 명시
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),  # 변경된 부분
        #"ground_truth": Value("string")
    })

    dataset = Dataset.from_dict(data, features=features)

    result = evaluate(
        dataset = dataset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            #context_recall, ## ground truth가 있어야 사용 가능함.
            #context_precision,
            #context_entity_recall,
            context_relevancy,
            #answer_similarity, 
            #answer_correctness,
            ],
        )
    
    df = result.to_pandas()

    # 파일이 존재하는지 확인
    if os.path.exists(full_path):
    # 파일이 존재하면, 기존 데이터를 불러오고 새로운 데이터를 추가
        existing_df = pd.read_csv(full_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(full_path, index=False)  # index=False로 설정하여 인덱스가 파일에 포함되지 않도록 함
    else:
    # 파일이 존재하지 않으면, 새로운 데이터를 파일로 생성
        df.to_csv(full_path, index=False)

@app.post("/query")
async def read_query(request: Request, background_tasks: BackgroundTasks):
    query = request.query
    response, source_document = chatbot.get_response(query)

    background_tasks.add_task(evaluate_ragas, query, response, source_document)

    return {"response": response,
            "source_document": source_document[0].page_content, 
            "help": source_document[0].metadata['source']}

chatbot = Chatbot()

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)