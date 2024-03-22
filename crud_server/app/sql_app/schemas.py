from typing import Union

from pydantic import BaseModel


class ItemBase(BaseModel):
    title: str
    description: Union[str, None] = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    items: list[Item] = []

    class Config:
        orm_mode = True


class PolicyBase(BaseModel):
    PolicyName: str
    MinAge: int
    MaxAge: int
    policyType: str
    start_date: str
    end_date: str
    OrgName: str

class Policy(PolicyBase):
    PolicyID: int

    class Config:
        orm_mode = True

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