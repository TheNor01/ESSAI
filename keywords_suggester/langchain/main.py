from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

loader = DirectoryLoader('keywords_suggester/data/dataset/automotive', glob="**/*.txt")
docs = loader.load()




text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

all_splits = text_splitter.split_documents(docs)


print(len(all_splits))
print("SAMPLE DOC")
print(all_splits[0])


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        
split_docs_chunked = split_list(all_splits, 166)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)


for split_docs_chunk in split_docs_chunked:
    vectordb = Chroma.from_documents(
        documents=split_docs_chunk,
        embedding=embed_model,
        persist_directory="keywords_suggester/index_storage_lang"
    )
    vectordb.persist()

print("DONE")

query = "What cars are fashion?"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)