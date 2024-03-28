from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME")
SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}'\
    .format(USERNAME=USERNAME,
            PASSWORD=PASSWORD,
            HOST=HOST,
            PORT=PORT,
            DBNAME=DBNAME)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
