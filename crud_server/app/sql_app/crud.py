from sqlalchemy.orm import Session
from datetime import datetime
from .utils import calculate_age, map_policy_to_response, map_policyV2_to_response
from sqlalchemy import case
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from . import models, schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_policies(db: Session):
    today_date = datetime.now().strftime('%Y%m%d')[2:]
    data_path = f'./sql_app/data/card_{today_date}.csv'
    df = pd.read_csv(data_path)

    for i in range(len(df)):
        if not df['MaxAge'][i] < 201.0:
            df['MaxAge'][i] = 200.0

        policy = models.PolicyV2(
            PolicyName=df['PolicyName'][i],
            D_day=str(df['D-day'][i]),
            OrgName=df['OrgName'][i],
            policyType=df['PolicyType'][i][:-2],
            Progress=df['Progress'][i],
            MinAge=int(df['MinAge'][i]),
            MaxAge=int(df['MaxAge'][i])
        )
        db.add(policy)

    db.commit()
    return {"message": "Data updated"}

def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def get_filtered_policies3(db: Session, customer_info: schemas.CustomerInfo):
    try:
        print(customer_info)
        birth_date = customer_info.birthDate
        residence = customer_info.residence
        preferred_policy_area = customer_info.preferredPolicyArea

        filtered_policies = db.query(models.PolicyV2).filter(
            ((birth_date == '') or (models.PolicyV2.MinAge <= calculate_age(birth_date) and models.PolicyV2.MaxAge >= calculate_age(birth_date))),
            ((residence == 'skip') or (models.PolicyV2.OrgName == residence)) or (models.PolicyV2.OrgName == '중앙부처'),
            ((preferred_policy_area == 'skip') or (models.PolicyV2.policyType == preferred_policy_area)),
        ).all()

        mapped_policies = [map_policyV2_to_response(policy) for policy in filtered_policies]
        return mapped_policies
    except SQLAlchemyError as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Database Error", "details": str(e)}
    except Exception as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Internal Server Error", "details": str(e)}


def get_filtered_policies2(db: Session, customer_info: schemas.CustomerInfo):
    try:
        print(customer_info)
        birth_date = customer_info.birthDate
        residence = customer_info.residence
        occupation = customer_info.occupation
        today_date = datetime.now()
        preferred_policy_area = customer_info.preferredPolicyArea
        personal_income_range = customer_info.personalIncomeRange
        household_income_range = customer_info.householdIncomeRange
        
        # 나이를 계산할 수 있는 경우에만 계산
        if birth_date:
            age = calculate_age(birth_date)
            # Filter policies based on customer's information and order by preferred policy area
            filtered_policies = db.query(models.PolicyV2).filter(
                models.PolicyV2.MinAge <= age,
                models.PolicyV2.MaxAge >= age, 
            ).order_by(
                case({PolicyV2.policyType: 0 for PolicyV2 in db.query(models.PolicyV2).filter(models.PolicyV2.policyType == preferred_policy_area)}, value=models.PolicyV2.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                models.PolicyV2.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()
        else:
            # birth_date가 비어 있는 경우에는 다른 조건에 따라 필터링된 정책 반환
            filtered_policies = db.query(models.PolicyV2).order_by(
                case({PolicyV2.policyType: 0 for PolicyV2 in db.query(models.PolicyV2).filter(models.PolicyV2.policyType == preferred_policy_area)}, value=models.PolicyV2.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                models.PolicyV2.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()

        mapped_policies = [map_policyV2_to_response(policy) for policy in filtered_policies]
        return mapped_policies
    except SQLAlchemyError as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Database Error", "details": str(e)}
    except Exception as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Internal Server Error", "details": str(e)}
    


def get_filtered_policies(db: Session, customer_info: schemas.CustomerInfo):
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
            filtered_policies = db.query(models.Policy).filter(
                models.Policy.MinAge <= age,
                models.Policy.MaxAge >= age, 
                today_date >= models.Policy.start_date,
            ).order_by(
                case({Policy.policyType: 0 for Policy in db.query(models.Policy).filter(models.Policy.policyType == preferred_policy_area)}, value=models.Policy.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                models.Policy.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()
        else:
            # birth_date가 비어 있는 경우에는 다른 조건에 따라 필터링된 정책 반환
            filtered_policies = db.query(models.Policy).filter(
                today_date >= models.Policy.start_date,
            ).order_by(
                case({Policy.policyType: 0 for Policy in db.query(models.Policy).filter(models.Policy.policyType == preferred_policy_area)}, value=models.Policy.policyType, else_=1),  # preferred_policy_area와 일치하는 경우를 우선으로 정렬
                models.Policy.policyType  # 그 다음으로 policyType에 따라 정렬
            ).all()

        mapped_policies = [map_policy_to_response(policy) for policy in filtered_policies]
        return mapped_policies
    except SQLAlchemyError as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Database Error", "details": str(e)}
    except Exception as e:
        db.rollback()  # 트랜잭션 롤백
        return {"error": "Internal Server Error", "details": str(e)}