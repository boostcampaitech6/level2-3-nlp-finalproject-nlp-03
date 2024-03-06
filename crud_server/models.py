from sqlalchemy import Column, TEXT, INT, BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Policy(Base):
    __tablename__ = "policy_table"

    id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    minAge = Column(INT, nullable=False)
    maxAge = Column(INT, nullable=False)
    residence = Column(TEXT, nullable=False)
    gender = Column(TEXT, nullable=False)


class Test(Base):
    __tablename__ = "test"

    id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    name = Column(TEXT, nullable=False)
    number = Column(INT, nullable=False)