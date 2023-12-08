from chromadb.config import Settings
import chromadb
import os
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from keywords_suggester.config import settings
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class BertTopicClass:
    def __init__(self,restore=0):

        self.storageModel = "keywords_suggester/models_checkpoint/bert"
        self.embeded_model = SentenceTransformer('all-mpnet-base-v2')
        
        if(restore==0):
            hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            vectorizer_model = CountVectorizer(stop_words="english",min_df=2, ngram_range=(1, 2))
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)


            self.main_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=self.embeded_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                top_n_words=5,
                language='english',
                calculate_probabilities=True,
                verbose=True
            )
        else:
            self.main_model = BERTopic.load(self.storageModel, embedding_model=self.embeded_model)
    
    def PersistModel(self):
        self.main_model.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embeded_model)
        


