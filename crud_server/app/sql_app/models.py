from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, TEXT
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(256), unique=True, index=True)
    hashed_password = Column(String(256))
    is_active = Column(Boolean, default=True)

    items = relationship("Item", back_populates="owner")

class Policy(Base):
    __tablename__ = "policies"

    PolicyID = Column(Integer, primary_key=True, autoincrement=True)
    PolicyName = Column(TEXT)
    
    # PolicyIntroduction = Column(TEXT)
    # SupportContent = Column(TEXT)
    MinAge = Column(Integer)
    MaxAge = Column(Integer)
    policyType = Column(TEXT)
    start_date = Column(TEXT)
    end_date = Column(TEXT)
    OrgName = Column(TEXT)

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    title = Column(String(256), index=True)
    description = Column(String(256), index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")