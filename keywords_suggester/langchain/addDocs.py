
from pydoc import doc
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.vectorstores import Chroma
from regex import P
from modules.LoaderEmbeddings import ProcessChunksFromLocal

from chromadb.config import Settings


persist_directory = "keywords_suggester/index_storage_lang"
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

CHROMA_SETTINGS = Settings(
        persist_directory="keywords_suggester/index_storage_lang"
)

#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)


persistent_client = chromadb.PersistentClient(
    path=persist_directory,
    settings=CHROMA_SETTINGS
)

print(persistent_client.get_settings())


langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="langchain",
    persist_directory = persist_directory,
    #client_settings=CHROMA_SETTINGS,
    embedding_function=embed_model,
)

print(langchain_chroma._persist_directory)

docs = langchain_chroma.get(where={"user": "d80b7"},limit=1)

"""

https://stackoverflow.com/questions/76482987/chroma-database-embeddings-none-when-using-get

{'ids': ['46c66dae-594c-11ee-92b8-0a925a2cd92a'], 
'embeddings': None, 
'metadatas': [{'category': 'sciences', 'row': 0, 'source': 'keywords_suggester/data_transformed/dataset/sciences/18.csv', 
'start_index': 0, 'user': 'd80b7'}], 
'documents': ['Video of Crazy-faced cats don’t win the adoption game\tCrazy-faced cats don’t win the adoption']}

"""


#collection = persistent_client.get_collection("langchain")
collection = langchain_chroma._collection
print(collection.count())


chunks = ProcessChunksFromLocal("keywords_suggester/data_transformed_upload/dataset/dummy")

langchain_chroma.add_documents(documents=chunks)
print(langchain_chroma._collection.count())


print(langchain_chroma._persist_directory)
langchain_chroma.persist()




exit()


for chunk in chunks[0:1]:
    print(chunk.page_content)
    query_result = embed_model.embed_query(chunk.page_content)
    print(query_result[0:3])
    print(len(query_result))




    """
    collection.add(
        #embeddings=query_result,
        documents= [chunk.page_content],
        metadatas= [chunk.metadata],
        ids = ["111111"]
    )
    """







print("There are", langchain_chroma._collection.count(), "in the collection")

result = collection.get(
    ids=["111111"],
)

print(result)