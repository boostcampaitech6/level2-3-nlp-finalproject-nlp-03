from sqlalchemy.orm import Session
from datetime import datetime
from .utils import calculate_age, map_policy_to_response
from sqlalchemy import case
from sqlalchemy.exc import SQLAlchemyError

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


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def get_filtered_policies(db: Session, customer_info: schemas.CustomerInfo):
    try:
        birth_date = customer_info.birthDate
        occupation = customer_info.occupation
        today_date = datetime.now()
        preferred_policy_area = customer_info.preferredPolicyArea
        personal_income_range = customer_info.personalIncomeRange
        household_income_range = customer_info.householdIncomeRange
        print(customer_info)
        
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
            print(preferred_policy_area)
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