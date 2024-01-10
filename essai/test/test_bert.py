from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset

from essai.bin.modules.BertSingle import BertTopicClass

#https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#langchain

#https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#language TUNING PARAM


SAVE=0
persist_directory = "essai/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)



embeded_model = SentenceTransformer('all-mpnet-base-v2')
collection=vectordb.get(include=["documents","embeddings","metadatas"])

computed_document = collection["documents"]
computed_embeddings = collection["embeddings"]


computed_document_array=np.array([np.array(xi) for xi in computed_embeddings])
print(computed_document_array.shape)


#print(type(computed_document_array))

#SUPERVISED
#https://maartengr.github.io/BERTopic/getting_started/supervised/supervised.html# --> it needs all sample already with a topic


#vediamo https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

"""
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english",min_df=2, ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)


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
"""



#print(computed_document)
if(SAVE==1):
    topic_model = BERTopic(embedding_model=embeded_model,verbose=True)
    topics, prob = topic_model.fit_transform(documents=computed_document,embeddings=computed_document_array)#,embeddings=computed_document_array)
    topic_model.save("./essai/models_checkpoint/bert", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)
else:
    topic_model = BERTopic.load("./essai/models_checkpoint/bert", embedding_model=embeded_model)

similar_topics, similarity = topic_model.find_topics("food", top_n=5)

#print(similar_topics)
#print(topic_model.get_topic(5)) #topic_model.get_topic(1, full=True)
#topic_model.visualize_barchart()


#print(topic_model.get_topic_info()) # we can access the frequent topics that were generated:
#print(topic_model.get_document_info(computed_document))



#print(topic_model.get_topic(similar_topics[0]))

#topic_model.visualize_topics().show() #show need


#https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html#saveload-bertopic-model
#https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#finding-similar-topics-between-models
#https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#multimodal-data
#https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html
#https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html
#https://maartengr.github.io/BERTopic/faq.html




"""
To continuously update the topic model as new data comes in
To continuously find new topics as new data comes in.

#how my topic change over time

#https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html

https://maartengr.github.io/BERTopic/getting_started/online/online.html #---> streaming 

oppure https://maartengr.github.io/BERTopic/getting_started/merge/merge.html


"""

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
abstracts_1 = dataset["abstract"][:2_000]

#prendere embeding e fare extend di questi

topic_model_1 = BERTopic(umap_model=umap_model, embedding_model=embeded_model,min_topic_size=20).fit(abstracts_1)
merged_model = BERTopic.merge_models([topic_model, topic_model_1], min_similarity=0.6) #Increasing this value will increase the change of adding new topics


print(len(topic_model.get_topic_info()))
print(len(merged_model.get_topic_info()))

print(merged_model.get_topic_info().tail(5))

#da lanciare settimana in settimana


computed_document.extend(abstracts_1)


topics, prob = merged_model.fit_transform(documents=computed_document)
merged_model.save("./essai/models_checkpoint/bert_merged", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)

