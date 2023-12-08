from chromadb.config import Settings
import chromadb
from langchain.vectorstores import Chroma

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class ChromaClass:
    def __init__(self,persist_directory,embeding_model,collection_name):
        self.persist_directory = persist_directory
        self.CHROMA_SETTINGS = Settings(self.persist_directory)
        self.persistent_client = chromadb.PersistentClient(path=self.persist_directory,settings=self.CHROMA_SETTINGS)

        self.CLIENT = Chroma(
                        client=self.persistent_client,
                        collection_name=collection_name,
                        persist_directory = self.persist_directory,
                        #client_settings=CHROMA_SETTINGS,
                        embedding_function=embeding_model
        )