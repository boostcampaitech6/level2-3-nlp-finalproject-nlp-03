from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database
import os
from dotenv import load_dotenv
from .models import Policy

Base = declarative_base()

class engineconn:
    def __init__(self):
        load_dotenv()
        self.USERNAME = os.getenv("USERNAME")
        self.PASSWORD = os.getenv("PASSWORD")
        self.HOST = os.getenv("HOST")
        self.PORT = os.getenv("PORT")
        self.DBNAME = os.getenv("DBNAME")
        DB_URL = 'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}'\
            .format(USERNAME=self.USERNAME,
                    PASSWORD=self.PASSWORD,
                    HOST=self.HOST,
                    PORT=self.PORT,
                    DBNAME=self.DBNAME)

        self.engine = create_engine(DB_URL, pool_recycle=500)

        # Check if the database exists, if not, create it
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        # Create tables based on models if they don't exist
        Policy.metadata.create_all(self.engine)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn
