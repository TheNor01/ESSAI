
from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass


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

    exit()

    computed_document_array=np.array([np.array(xi) for xi in computed_embeddings])
    print(computed_document_array.shape)



    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english",min_df=2, ngram_range=(1, 2))
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)


    main_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        top_n_words=5,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )

    if(SAVE==1):
        main_model = BERTopic(embedding_model=embeded_model,verbose=True)
        topics, prob = main_model.fit_transform(documents=computed_document,embeddings=computed_document_array)#,embeddings=computed_document_array)
        main_model.save("./keywords_suggester/models_checkpoint/bert", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)
    else:
        main_model = BERTopic.load("./keywords_suggester/models_checkpoint/bert", embedding_model=embeded_model)