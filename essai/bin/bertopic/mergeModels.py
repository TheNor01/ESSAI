
from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset

from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer



# fare setting e singleton


if __name__ == "__main__":

    embeded_model = SentenceTransformer('all-mpnet-base-v2')
    main_model = BERTopic.load("./essai/models_checkpoint/bert", embedding_model=embeded_model)


    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
    abstracts_1 = dataset["abstract"][:1_500] #according min topic size == K hdbscan

    print(len(abstracts_1))
    print(type(abstracts_1))


   
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    topic_model_1 = BERTopic(umap_model=umap_model, embedding_model=embeded_model,min_topic_size=20).fit(abstracts_1)

    merged_model = BERTopic.merge_models([main_model, topic_model_1], min_similarity=0.6) #Increasing this value will increase the change of adding new topics

    print(len(main_model.get_topic_info()))
    print(len(merged_model.get_topic_info()))

    print(merged_model.get_topic_info().tail(5))

    exit()

    #da lanciare settimana in settimana

    persist_directory = "essai/index_storage_lang"

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)


    embeded_model = SentenceTransformer('all-mpnet-base-v2')
    collection=vectordb.get(include=["documents","embeddings","metadatas"])

    computed_document = collection["documents"]

    computed_document.extend(abstracts_1)


    topics, prob = merged_model.fit_transform(documents=computed_document)
    merged_model.save("./essai/models_checkpoint/bert_merged", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)
    # it should be main LOCATION

