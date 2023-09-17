from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.storage import  LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore





fs = LocalFileStore("./caching/") 


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]


def initDbCroma(docs,embed_model):

    print("INIT CROMA DB")

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 50,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
    )

    all_splits = text_splitter.split_documents(docs)
    split_docs_chunked = split_list(all_splits, 120)

    
    for split_docs_chunk in split_docs_chunked:
        vectordb = Chroma.from_documents(
            documents=split_docs_chunk,
            embedding=embed_model,
            persist_directory="keywords_suggester/index_storage_lang"
        )

    vectordb.persist()
    print("DONE")

    return vectordb


#Populate croma collections
"""

https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever#retrieving-full-documents




"""


"""

for split_docs_chunk in split_docs_chunked:
    vectordb = Chroma.from_documents(
        documents=split_docs_chunk,
        embedding=embed_model,
        persist_directory="keywords_suggester/index_storage_lang"
    )
    vectordb.persist()

print("DONE")

collection = vectordb.get()

print(collection) #ids, metadata, documents
print(collection.keys())

"""

# The storage layer for the parent documents

