
from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
from keywords_suggester.config import settings

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings

if __name__ == "__main__":

    SAVE=0

    settings.init()
    persist_directory = settings.persist_directory+"init_dataset_small"+"/"
    embed_model = settings.embed_model
    collection_name_local = "TestCollection"
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)


    embeded_model = SentenceTransformer('all-mpnet-base-v2')
    collection=ChromaDB.CLIENT.get(include=["documents","embeddings","metadatas"])

    computed_document = collection["documents"]
    computed_embeddings = collection["embeddings"]

    print(ChromaDB.CLIENT._collection.name)
    print(len(computed_embeddings))


    computed_document_array=np.array([np.array(xi) for xi in computed_embeddings])
    print(computed_document_array.shape)

    BERT = None
    if(SAVE==1):
        #main_model = BERTopic(embedding_model=embeded_model,verbose=True)
        BERT = BertTopicClass()
        topics, prob = BERT.main_model.fit_transform(documents=computed_document,embeddings=computed_document_array)#,embeddings=computed_document_array)
        
        BERT.PersistModel()
        #BERT.save("./keywords_suggester/models_checkpoint/bert", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)
    else:
        print("RESTORING MODEL BERT ...")
        BERT = BertTopicClass(restore=1)


    print(BERT.main_model.get_topic_info())