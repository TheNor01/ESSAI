
from langchain.embeddings import HuggingFaceEmbeddings

def init():

    global persist_directory
    persist_directory="keywords_suggester/index_storage_lang/"


    global embed_model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #spostare su global