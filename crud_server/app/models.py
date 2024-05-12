from sqlalchemy import Column, TEXT, INT, BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Policy(Base):
    __tablename__ = "policy_table"
    
    PolicyID = Column(INT, primary_key=True, autoincrement=True)
    PolicyName = Column(TEXT)
    
    # PolicyIntroduction = Column(TEXT)
    # SupportContent = Column(TEXT)
    MinAge = Column(INT)
    MaxAge = Column(INT)
    policyType = Column(TEXT)
    start_date = Column(TEXT)
    end_date = Column(TEXT)
    OrgName = Column(TEXT)

    # Major = Column(TEXT)
    # EmploymentStatus = Column(TEXT)
    # Education = Column(TEXT)
    # IncomeRange = Column(INT)
    # Residence = Column(TEXT)
    # Specialization = Column(TEXT)
    # EligibilityRestriction = Column(TEXT)


class Test(Base):
    __tablename__ = "test"

    id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    name = Column(TEXT, nullable=False)
    number = Column(INT, nullable=False)