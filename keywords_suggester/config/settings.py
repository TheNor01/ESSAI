
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def init():

    global persist_directory
    persist_directory="keywords_suggester/index_storage_lang/"

    global source_directory
    source_directory="keywords_suggester/dataset_source/"

    global upload_directory
    upload_directory="keywords_suggester/dataset_source/data_upload"

    global embed_model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #spostare su global

    global bert_embeded
    bert_embeded = SentenceTransformer('all-mpnet-base-v2')
