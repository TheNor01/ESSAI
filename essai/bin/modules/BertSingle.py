import chromadb
import os
import pickle
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from essai.config import settings
from bertopic import BERTopic
from bertopic.backend._utils import select_backend
from datasets import load_dataset
import traceback
from transformers import pipeline
from bertopic.representation import MaximalMarginalRelevance
from wordcloud import WordCloud
from matplotlib import pyplot as plt




def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class BertTopicClass:
    def __init__(self,BERT_NAME,restore=0,nr_topics='auto'):

        settings.init()
        self.storageModel = "essai/models_checkpoint/bert"+"/"+BERT_NAME
        self.embed_name = settings.embed_name
        self.embeded_model = settings.bert_embeded

        self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=2,ngram_range=(1, 2)) # min_df changed to 1
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.umap_model = UMAP(n_neighbors=15, n_components=15, min_dist=0.1, metric='cosine', random_state=42)

        #NEW 
        self.nr_topics = nr_topics

        #NEW 
        self.representation_model = MaximalMarginalRelevance(diversity=0.6,top_n_words=15)

        self.init_documents = [] 
        self.main_model = None
        if(restore==0):
            self.main_model = BERTopic(
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                embedding_model=self.embeded_model,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                representation_model=self.representation_model,
                nr_topics=self.nr_topics,
                top_n_words=15,
                language='english',
                calculate_probabilities=True,
                verbose=True
            )

        else:
            print("LOADING MODEL FROM "+self.storageModel)
            #self.main_model = BERTopic.load(self.storageModel, embedding_model=self.embeded_model)
            self.main_model = BERTopic.load(self.storageModel, embedding_model=self.embed_name)
            #self.main_modell.embedding_model = select_backend(my_embedding_model)
    
    def PersistModel(self,modelToSave=None):

        modelToStore = None
        if not modelToSave is None:
            modelToStore=modelToSave
            print("Merged model saving")
        else:
            modelToStore=self.main_model

            print("old model saving")

        if(os.path.exists(self.storageModel)):
            print("FOUND A SAVED MODEL")
            print("WOULD YOU LIKE OVERWRITE IT?")

            user_input = input('(y/n): ')
            if user_input.lower() == 'y':
                print("SAVING BERT INTO "+self.storageModel)
                #self.main_model.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embeded_model)
                modelToStore.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embed_name)
            elif user_input.lower() == 'n':
                return
            else:
                print('Type y or n')
        else:
            print("SAVING BERT INTO "+self.storageModel)
            #self.main_model.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embeded_model)
            modelToStore.save(self.storageModel, serialization="pytorch", save_ctfidf=True, save_embedding_model=self.embed_name)

        

    def UpdateDocuments(self,docs): #TODO aggiungere alla lista esistente quelli passati nella creazione
        docs_to_save=None
        if(not self.init_documents):
            docs_to_save = self.init_documents 
        else: #extend documents to save
            docs_to_save = docs.extend(self.init_documents)
        self.__storeDocumentsPickle__(docs_to_save)

    
    def __storeDocumentsPickle__(self,docs_to_save):
        path="./essai/storage/documents_sync"
        if(not os.path.exists(path)):
            os.makedirs(path)
        with open('./essai/storage/documents_sync/documents.pkl', 'wb') as f:
            pickle.dump(docs_to_save, f)


    def GenereateTopicLabels(self):
        topic_labels =  self.main_model.generate_topic_labels(nr_words=3,
                                                 topic_prefix=False,
                                                 word_length=10,
                                                 separator=", ")

    def ChangeLabelMeaning(self,dictToChange:dict):
        #{1: "Space Travel", 7: "Religion"}
        self.main_model.set_topic_labels(dictToChange)
        print("SAVING CHANGES..")
        self.PersistModel()

        print("CUSTOM LABELS :->")
        print(self.main_model.custom_labels_)


    def __loadDocumentsSync__(self):
        mynewlist=None
        with open('./essai/storage/documents_sync.pkl', 'rb') as f:  #TODO we should query collection documents
            mynewlist = pickle.load(f)

        if(not mynewlist):
            print("INIT DOCUMENTS ARE EMPTY")
            return

        return mynewlist


    def ReduceTopics(self,number_topics_to_obtain='auto'):


        mynewlist = self.__loadDocumentsSync__()
        
        self.main_model.reduce_topics(mynewlist, nr_topics=number_topics_to_obtain)

        print(self.main_model.get_topic_info())
        self.PersistModel()


    def ManualMergeTopics(self,list_to_merge):
        #[[1, 2][3, 4]]
        mynewlist = self.__loadDocumentsSync__()

        for couple in list_to_merge:
            print("MERGING ",self.main_model.get_topic_info(couple[0])["Name"].item()," - " ,self.main_model.get_topic_info(couple[1])["Name"].item())
            print("======")
        self.main_model.merge_topics(mynewlist, list_to_merge)
        self.PersistModel()

    def VisualizeTopics(self):
        self.main_model.visualize_topics().show()

    def SuggestLabels(self,text,candidate_labels):
        classifier_EXT = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("CANDIDATED LABELS ARE:")
        print(*candidate_labels, sep='\n')
        print("======")
        print("using zero shot classification bert")
        output_labels = classifier_EXT(text, candidate_labels)

        return output_labels
    
    def TopicInfo(self,topic=None):
        topic_df =  self.main_model.get_topic_info(topic)
        return topic_df
    
    def FindSimilarTopics(self,topic,top_n=5):

        similar_topics, similarity = self.main_model.find_topics(topic, top_n)
        return similar_topics, similarity
    
    def Genereate_WC(self, topic):
        text = {word: value for word, value in self.main_model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        plt.savefig("./templates/static/images/WC.png")
        

    def GenerateTopicChart(self):
        self.main_model.visualize_barchart().show()
   
    def TopicOverTime(self,collectionChroma):

        #IT will be a report

        print("LOADING TOPIC OVER TIME...")
        docs = collectionChroma["documents"][0:8000]
        timestamps = [element["created_at"] for element in collectionChroma["metadatas"]][0:8000]

        topic_list = []
        for doc in docs:
            topics, _ = self.main_model.transform(doc)
            max_topic = topics[0]
            topic_list.append(max_topic)
           
        if len(timestamps) != len(docs):
            print("NOT EQUAL SIZE TIMESTAMPS AND DOCS")
            return

        print(len(timestamps))
        print(len(docs))

        topics_over_time = self.main_model.topics_over_time(docs,timestamps,topics=topic_list,datetime_format="%Y-%m-%d")

        #print(topics_over_time)

        fig = self.main_model.visualize_topics_over_time(topics_over_time,top_n_topics=6).show()
        fig.write_html("./DYM.html")

    def __documentsPerTopic__(self,merged_model,new_docs):

        #old_docs are trained documents, updated sync document
        old_docs = self.__loadDocumentsSync__()
        documents = pd.DataFrame(
            {
                "Document": old_docs + new_docs,
                "ID": range(len(old_docs)+len(new_docs)),
                "Topic": merged_model.topics_,
                "Image": None
            })
        
        self.UpdateDocuments(old_docs.extend(new_docs)) #store new documents

        return documents


    def PreviewMerge(self,text : list[str],min_similarity_topics):
        
        local_umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        try:
            topic_model_1 = BERTopic(
                umap_model=local_umap_model,
                embedding_model=self.embeded_model,
                top_n_words=10,
                nr_topics='auto',
                language='english',
                calculate_probabilities=True,
                verbose=True).fit(text)
            

            merged_model = BERTopic.merge_models([self.main_model, topic_model_1], min_similarity=min_similarity_topics) #Increasing this value will increase the change of adding new topics

            sizeTopicMain = len(self.main_model.get_topic_info())
            sizeTopicNew = len(merged_model.get_topic_info())

            print(sizeTopicMain)
            print(sizeTopicNew)

            discoveredTopics = sizeTopicNew-sizeTopicMain

            print("NEW TOPICS DISCOVERED -->")
            print(merged_model.get_topic_info().tail(discoveredTopics))

            #SAVE WEIGHTS

            # Assign CountVectorizer to merged model
            merged_model.vectorizer_model = topic_model_1.vectorizer_model
            documents_per_topic = self.__documentsPerTopic__(merged_model,text)


            # Re-calculate c-TF-IDF
            c_tf_idf, _ = merged_model._c_tf_idf(documents_per_topic)
            merged_model.c_tf_idf_ = c_tf_idf
            print("UPDATED WEIGHTS")
            print("SAVING MERGED MODELS")

            print(merged_model.get_topic_info())

            self.PersistModel(merged_model)


        except ValueError:
            traceback.print_exc() 
            print("ERROR OCCURED , try change your max_df, min_df value")
        except IndexError:
            traceback.print_exc() 
            print("ERROR OCCURED , increase number samples")

