from chromadb.config import Settings
import chromadb
import os
import re
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from keywords_suggester.config import settings
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import traceback
from transformers import pipeline


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class BertTopicClass:
    def __init__(self,BERT_NAME,restore=0):

        self.storageModel = "keywords_suggester/models_checkpoint/bert"+"/"+BERT_NAME
        self.embeded_model = SentenceTransformer('all-mpnet-base-v2')



        self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=2,ngram_range=(1, 2)) # min_df changed to 1
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        #min samples 15  #min_cluster_size

        if(restore==0):
            self.main_model = BERTopic(
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                embedding_model=self.embeded_model,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                top_n_words=5,
                language='english',
                calculate_probabilities=True,
                verbose=True
            )
        else:
            print("LOADING MODEL FROM "+self.storageModel)
            self.main_model = BERTopic.load(self.storageModel, embedding_model=self.embeded_model)
    
    def PersistModel(self):
        print("SAVING BERT INTO "+self.storageModel)
        self.main_model.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embeded_model)


    def GenereateTopicLabels(self):
        topic_labels =  self.main_model.generate_topic_labels(nr_words=3,
                                                 topic_prefix=False,
                                                 word_length=10,
                                                 separator=", ")
        print(topic_labels)

    def ChangeLabelMeaning(self,dictToChange:dict):
        #{1: "Space Travel", 7: "Religion"}
        self.main_model.set_topic_labels(dictToChange)
        print("SAVING CHANGES..")
        self.PersistModel()


    def SuggestLabels(text,candidate_labels):
        classifier_EXT = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        output_labels = classifier_EXT(text, candidate_labels)

        return output_labels
    
   
    def TopicOverTime(self,docs,timestamps):

        print("LOADING TOPIC OVER TIME...")

        #trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
        #trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
        #trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
        #trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
        #trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
        # timestamps = trump.date.to_list()[0:657]
        # tweets = trump.text.to_list()

        #print(len(timestamps))
        #print(len(docs))

        #TODO FIX not same size

        topics_over_time = self.main_model.topics_over_time(docs,timestamps,datetime_format=None)
        self.main_model.visualize_topics_over_time(topics_over_time).show()

    def PreviewMerge(self,text : list[str],min_similarity_topics):
        
    
        #IDEA slide parameters in order to preview different result

        local_hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        local_vectorizer_model = CountVectorizer(stop_words="english",ngram_range=(1, 2)) # min_df changed to 1
        local_ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        local_umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        try:
            topic_model_1 = BERTopic(
                umap_model=local_umap_model,
                hdbscan_model=local_hdbscan_model,
                embedding_model=self.embeded_model,
                vectorizer_model=local_vectorizer_model,
                ctfidf_model=local_ctfidf_model,
                top_n_words=5,
                language='english',
                calculate_probabilities=True,
                verbose=True).fit(text)
            
            #da testare con tanti topics
            merged_model = BERTopic.merge_models([self.main_model, topic_model_1], min_similarity=min_similarity_topics) #Increasing this value will increase the change of adding new topics

            sizeTopicMain = len(self.main_model.get_topic_info())
            sizeTopicNew = len(merged_model.get_topic_info())


            print(sizeTopicMain)
            print(sizeTopicNew)

            discoveredTopics = sizeTopicNew-sizeTopicMain

            print("NEW TOPICS DISCOVERED --> Do you like them")
            print(merged_model.get_topic_info().tail(discoveredTopics))

            #TODO update merged models, is it right?


        except ValueError:
            traceback.print_exc() 
            print("ERROR OCCURED , try change your max_df, min_df value")
        except IndexError:
            traceback.print_exc() 
            print("ERROR OCCURED , increase number samples")

