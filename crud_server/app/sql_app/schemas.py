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


class PolicyCreate(BaseModel):
    PolicyName: str
    D_day: str
    OrgName: str
    PolicyType: str
    Progress: str
    MinAge: int
    MaxAge: int
    
class PolicyV2Base(BaseModel):
    PolicyName: str
    MinAge: int
    MaxAge: int
    policyType: str
    D_day: str
    OrgName: str

class PolicyV2(PolicyV2Base):
    PolicyID: int

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

class PolicyV2InfoResponse(BaseModel):
    id: int
    PolicyName: str
    d_day: str
    policy_type: str
    Progress: str
    org_name: str

class CustomerInfo(BaseModel):
    birthDate: str
    residence: str
    householdIncomeRange: str
    occupation: str
    personalIncomeRange: str
    preferredPolicyArea: str