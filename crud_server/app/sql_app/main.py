from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/filter_policy", response_model=list[schemas.PolicyInfoResponse])
async def filter_policy(customer_info: schemas.CustomerInfo, db: Session = Depends(get_db)):
    policies = crud.get_filtered_policies(db=db, customer_info=customer_info)
    return policies


@app.post("/filter_policy_v2", response_model=list[schemas.PolicyV2InfoResponse])
async def filter_policy(customer_info: schemas.CustomerInfo, db: Session = Depends(get_db)):
    policies = crud.get_filtered_policies2(db=db, customer_info=customer_info)
    return policies


@app.post("/filter_policy_v3", response_model=list[schemas.PolicyV2InfoResponse])
async def filter_policy(customer_info: schemas.CustomerInfo, db: Session = Depends(get_db)):
    policies = crud.get_filtered_policies3(db=db, customer_info=customer_info)
    return policies


@app.post("/update_policies")
async def update_policies(db: Session = Depends(get_db)):
    return crud.update_policies(db=db)