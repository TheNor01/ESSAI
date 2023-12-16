
from langchain.indexes import SQLRecordManager, index

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance



@singleton
class Indexer():
    def __init__(self,collection_name,vectorstore):
        index_name_local = collection_name + "_index"
        namespace = f"chromadb/{index_name_local}"
        print("CREATED INDEX NAMESPACE -> "+namespace)
        
        self.vectorstore = vectorstore

        self.record_manager = SQLRecordManager(
            namespace, db_url="./keywords_suggester/storage/Indexer/sqlite:///record_manager_cache.sql"
        )
        

        self.record_manager.create_schema()

    def IndexIncremental(self,docs):
        info = index(docs, self.record_manager, self.vectorstore, cleanup="incremental", source_id_key="source")
        return info
    
    
    def Index(self,docs):
        index(docs, self.record_manager, self.vectorstore, cleanup=None, source_id_key="source")