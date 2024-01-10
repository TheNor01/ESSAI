from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from essai.config import settings
from essai.bin.modules.ChromaSingle import ChromaClass


#https://python.langchain.com/docs/modules/data_connection/indexing

def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, ChromaDB.CLIENT, cleanup="full", source_id_key="source")

if __name__ == '__main__':
    collection_name_local = "TestCollection"
    index_name_local = collection_name_local + "_index"

    settings.init()
    embed_model = settings.embed_model
    persist_directory = settings.persist_directory+"init_dataset_small"+"/"

    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

    namespace = f"chromadb/{index_name_local}"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )

    record_manager.create_schema()