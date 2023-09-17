
import sys
sys.path.append('keywords_suggester/langchain')

from loaders.DIRLoader import DIRLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb



from setup.buildDb import *


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


loader = DIRLoader('keywords_suggester/data_transformed/dataset/',metadata_columns=["user","category"],content_column="content")
docs = loader.load()

db = initDbCroma(docs,embed_model)

retriever = db.as_retriever()

#assemble retriver



