
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def init():

    global persist_directory
    persist_directory="keywords_suggester/index_storage_lang/"

    global collection_name
    collection_name="default"

    global bert_name
    bert_name = "bert"+"_"+collection_name

    global init_dataset
    init_dataset="init_dataset"

    global source_directory
    source_directory="keywords_suggester/dataset_source/"

    global upload_directory
    upload_directory="keywords_suggester/dataset_source/data_upload"

    global embed_name
    embed_name="all-mpnet-base-v2"

    global embed_model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers"+embed_name)

    global bert_embeded
    bert_embeded = SentenceTransformer(embed_name)
    
