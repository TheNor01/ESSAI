import imp
from tkinter import NO
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.modules.LoaderEmbeddings import InitChromaDocsFromPath

import chromadb


#SingleTon Chroma cross interface

"""
loader = DIRLoader('keywords_suggester/data_transformed/dataset/food',metadata_columns=["user","category"],content_column="content")
docs = loader.load()

#print(docs[0])
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

all_splits = text_splitter.split_documents(docs)


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        
split_docs_chunked = split_list(all_splits, 166)
"""


split_docs_chunked = InitChromaDocsFromPath('keywords_suggester/data_transformed/dataset/food')

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Populate croma collections
"""

https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever#retrieving-full-documents



"""

vectordb = None

for split_docs_chunk in split_docs_chunked:
    vectordb = Chroma.from_documents(
        documents=split_docs_chunk,
        #collection_name=
        embedding=embed_model,
        persist_directory="keywords_suggester/index_storage_lang"
    )
    
vectordb.persist()

print("DONE")

collection = vectordb.get()

print(collection) #ids, metadata, documents
print(collection.keys())
