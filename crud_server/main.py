from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import engineconn
from models import Policy

app = FastAPI()

engine = engineconn()
session = engine.sessionmaker()

class RegistPolicyRequest(BaseModel):
    minAge: int
    maxAge: int
    residence: str
    gender: str

@app.get("/")
async def first_get():
    print("first get")
    example = session.query(Policy).all()
    return example

@app.post("/create_policy")
async def create_policy(request: RegistPolicyRequest):
    # 요청에서 받은 데이터를 사용하여 새로운 Policy 객체 생성
    new_policy = Policy(minAge=request.minAge, maxAge=request.maxAge, residence=request.residence, gender=request.gender)
    # 세션에 추가하고 커밋
    session.add(new_policy)
    session.commit()
    print(new_policy)
    # 새로 추가된 정책 반환
    return new_policy

# 8001번 포트로 서버를 실행합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
