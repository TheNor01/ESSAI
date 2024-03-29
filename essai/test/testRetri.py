

import sys
sys.path.append('essai/langchain')


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from loaders.DIRLoader import DIRLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
import chromadb
from datetime import datetime, timedelta
from langchain.schema import Document
from chromadb.config import Settings


loader = DIRLoader('essai/data_transformed/dataset/',metadata_columns=["user","category"],content_column="content")
docs = loader.load()


persist_directory = "essai/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)


#https://python.langchain.com/docs/modules/data_connection/retrievers/time_weighted_vectorstore

persist_directory = "essai/index_storage_lang"
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

CHROMA_SETTINGS = Settings(
        persist_directory="essai/index_storage_lang"
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

retriever = TimeWeightedVectorStoreRetriever(vectorstore=langchain_chroma, decay_rate=.000001, k=1)

yesterday = datetime.now() - timedelta(days=1)

retriever.add_documents([Document(page_content="hello world", metadata={"last_accessed_at": yesterday})])
exit()


retriever.add_documents([Document(page_content="hello foo")])


print(retriever.get_relevant_documents("hello world"))




#1 type of retriver, VectorStores
#similarity, mmr, similarity treshold

#https://www.reddit.com/r/LangChain/comments/12qn2qi/filter_with_retriever/
croma_retriever = vectordb.as_retriever(search_kwargs={"k": 2}) #or K
#croma_retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})


#2 dense retriver
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2


#3 parent retriever

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=embed_model
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, 
    docstore=store, 
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)
retrieved_docs = retriever.get_relevant_documents("car dealers")

print(len(retrieved_docs))
print(retrieved_docs[0].page_content)


#4 

retrieverTF = TFIDFRetriever.from_documents(docs)
result = retrieverTF.get_relevant_documents("tesla")




#Assemble
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, croma_retriever], weights=[0.5, 0.5])

docs = ensemble_retriever.get_relevant_documents("cars")


print(docs)