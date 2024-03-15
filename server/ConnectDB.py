import chromadb
from chromadb.config import Settings

host = '175.45.203.113'
port = '1123'

class connectDB:
    def __init__(self, allow_reset):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            setting=Settings(allow_reset=allow_reset),
        )
        if allow_reset:
            self.client.reset() # resets the database
        self.collection = self.client.create_collection()
        
        

    def getClient(self):
        pass
