
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


loader = DIRLoader('keywords_suggester/data_transformed/dataset',metadata_columns=["user","category"],content_column="content")
docs = loader.load()

db = initDbCroma(docs,embed_model)


exit()
retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'k' : 10, 'score_threshold': 0.3, 'filter': {'content':'automotive'}}
            )


response = retriever.get_relevant_documents("I want to buy a red cars")
print(response)


#assemble retriver



