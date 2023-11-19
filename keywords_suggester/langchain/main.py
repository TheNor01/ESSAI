from loaders.DIRLoader import DIRLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb



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

#embed_model = HuggingFaceEmbeddings()
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)



#Populate croma collections
"""

https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever#retrieving-full-documents




"""


vectordb=None

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
