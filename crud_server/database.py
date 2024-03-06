from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import models

class engineconn:

    def __init__(self):
        load_dotenv()
        self.USERNAME = os.getenv("USERNAME")
        self.PASSWORD = os.getenv("PASSWORD")
        self.HOST = os.getenv("HOST")
        self.PORT = os.getenv("PORT")
        self.DBNAME = os.getenv("DBNAME")
        DB_URL = 'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}'.format(
            USERNAME=self.USERNAME,
            PASSWORD=self.PASSWORD,
            HOST=self.HOST,
            PORT=self.PORT,
            DBNAME=self.DBNAME
        )
        self.engine = create_engine(DB_URL, pool_recycle = 500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn