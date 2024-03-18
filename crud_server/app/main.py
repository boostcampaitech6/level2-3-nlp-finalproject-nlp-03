from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .database import engineconn
from .models import Policy
from datetime import datetime
from sqlalchemy import case
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

app = FastAPI()

engine = engineconn()
session = engine.sessionmaker()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class RegistPolicyRequest(BaseModel):
    minAge: int
    maxAge: int
    residence: str
    gender: str

class PolicyInfoResponse(BaseModel):
    id: int
    PolicyName: str
    d_day: int
    policy_type: str
    org_name: str

class CustomerInfo(BaseModel):
    birthDate: str
    householdIncomeRange: str
    occupation: str
    personalIncomeRange: str
    preferredPolicyArea: str


def calculate_age(birth_date: str) -> int:
    """
    Calculate age based on the given birth date.

    Parameters:
        birth_date (str): A string representing the birth date in the format 'YYYY-MM-DD'.

    Returns:
        int: The calculated age.
    """
    # Convert birth date string to a datetime object
    birth_date_obj = datetime.strptime(birth_date, "%Y-%m-%d")
    
    # Get the current date
    current_date = datetime.now()
    
    # Calculate age
    age = current_date.year - birth_date_obj.year
    
    # Adjust age if the birth month and day have not occurred yet this year
    if current_date.month < birth_date_obj.month or \
            (current_date.month == birth_date_obj.month and current_date.day < birth_date_obj.day):
        age -= 1
    
    return age

def map_policy_to_response(policy: Policy) -> PolicyInfoResponse:
    """
    Map a Policy object to a PolicyInfoResponse object.

    Parameters:
        policy (Policy): A Policy object.

    Returns:
        PolicyInfoResponse: A PolicyInfoResponse object mapped from the provided Policy object.
    """
    end_date = datetime.strptime(policy.end_date, "%Y-%m-%d")  # 문자열을 datetime으로 변환
    d_day = (end_date - datetime.now()).days  # d_day 계산

    return PolicyInfoResponse(
        id=policy.PolicyID,
        PolicyName=policy.PolicyName,
        d_day=d_day,
        policy_type=policy.policyType,
        org_name=policy.OrgName
    )


@app.post("/filter_policy")
async def filter_policy(customer_info: CustomerInfo):
    try:
        birth_date = customer_info.birthDate
        occupation = customer_info.occupation
        today_date = datetime.now()
        preferred_policy_area = customer_info.preferredPolicyArea
        personal_income_range = customer_info.personalIncomeRange
        household_income_range = customer_info.householdIncomeRange
        
        # 나이를 계산할 수 있는 경우에만 계산
        if birth_date:
            age = calculate_age(birth_date)
            # Filter policies based on customer's information and order by preferred policy area
            filtered_policies = session.query(Policy).filter(
                Policy.MinAge <= age,
                Policy.MaxAge >= age, 
                today_date >= Policy.start_date,
            ).order_by(
                case({Policy.policyType: 0 for Policy in session.query(Policy).filter(Policy.policyType == preferred_policy_area)}, value=Policy.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                Policy.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()
        else:
            # birth_date가 비어 있는 경우에는 다른 조건에 따라 필터링된 정책 반환
            print(preferred_policy_area)
            filtered_policies = session.query(Policy).filter(
                today_date >= Policy.start_date,
            ).order_by(
                case({Policy.policyType: 0 for Policy in session.query(Policy).filter(Policy.policyType == preferred_policy_area)}, value=Policy.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                Policy.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()

        mapped_policies = [map_policy_to_response(policy) for policy in filtered_policies]
        return mapped_policies
    except SQLAlchemyError as e:
        session.rollback()  # 트랜잭션 롤백
        return {"error": "Database Error", "details": str(e)}
    except Exception as e:
        session.rollback()  # 트랜잭션 롤백
        return {"error": "Internal Server Error", "details": str(e)}


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
