

import sys
sys.path.append('keywords_suggester/langchain')


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from loaders.DIRLoader import DIRLoader



loader = DIRLoader('keywords_suggester/data_transformed/dataset/',metadata_columns=["user","category"],content_column="content")
docs = loader.load()


persist_directory = "keywords_suggester/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

croma_retriever = vectordb.as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, croma_retriever], weights=[0.5, 0.5])

docs = ensemble_retriever.get_relevant_documents("cars")


print(docs)