
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
    def __init__(self,collection_name):
        index_name_local = collection_name + "_index"
        namespace = f"chromadb/{index_name_local}"
        print("CREATED INDEX NAMESPACE -> "+namespace)

        self.record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )

        self.record_manager.create_schema()

    def Index(self,docs,vectorstore):
        index(docs, self.record_manager, vectorstore, cleanup="incremental", source_id_key="source")