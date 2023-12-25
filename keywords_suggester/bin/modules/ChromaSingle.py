from chromadb.config import Settings
import chromadb
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from keywords_suggester.bin.modules.Indexer import Indexer
import pandas as pd
from matplotlib import pyplot as plt
import mplcursors


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class ChromaClass:
    def __init__(self,persist_directory,embeding_model,collection_name):
        self.storagePath = "keywords_suggester/storage" + "/"
        self.persist_directory = persist_directory
        self.CHROMA_SETTINGS = Settings(self.persist_directory)
        self.persistent_client = chromadb.PersistentClient(path=self.persist_directory,settings=self.CHROMA_SETTINGS)
        self.CLIENT = Chroma(
                        client=self.persistent_client,
                        collection_name=collection_name,
                        persist_directory = self.persist_directory,
                        #client_settings=CHROMA_SETTINGS,
                        embedding_function=embeding_model
        )
        
        self.indexer = Indexer(collection_name,self.CLIENT)

        self.remove_keys = {"start_index","row"}

        self.__storeMetadataFile__()
        

    def GetListOfUsers(self):
        collection = self.CLIENT.get()
        metadata_collection = collection["metadatas"]
        users = set([ element['user'] for element in metadata_collection ])
        self.__storeUsersFile__(users)
        return users
    

    def HistogramUsersTopics(self,target_user = None): #extract a histogram of all users with their main topic
        collection = self.CLIENT.get()
        metadata_collection = collection["metadatas"]
        user_list = []
        category_list = []
        for element in metadata_collection:
             user_list.append(element["user"])
             category_list.append(element["category"])


        hist_df = pd.DataFrame(list(zip(user_list, category_list)),columns=['user','category'])
        #print(hist_df)

        #target_user = '95d12'
        # Query the DataFrame for information related to the target user
        if(target_user):
            hist_df = hist_df[hist_df['user'] == target_user]

        print(hist_df)

        threshold_percentage = 5.0
        category_counts = hist_df['category'].value_counts()
        low_percentage_categories = category_counts[category_counts / category_counts.sum() * 100 < threshold_percentage].index
        
        category_counts['Other'] = category_counts[low_percentage_categories].sum()
        category_counts = category_counts.drop(low_percentage_categories)

        # Filter out categories with 0 percentage values
        category_counts = category_counts[category_counts > 0]

        # Plotting the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        pie = ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)

        #TODO Add hover labels
        #TODO PLot better
        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"{sel.artist.get_label()}\nUsers: {category_counts[sel.artist.get_label()]}"))

        plt.show()
        pass
    

    def GetListOfDocs(self):
        collection = self.CLIENT.get()
        doc_collection = collection["documents"]
        return doc_collection
    
    def AddSpecialDocs(self,documents):
        
        #INSER HERE INDEXER
        
        #we load default collection, as a startup phase, then every special document will be tracked
        #We mark them as mutated documents. As a special info bio for users. We can update them every time
        
        info = self.indexer.IndexIncremental(documents)
        
        print("TOTAL IDS ADDED: -> "+str((info["num_added"])))
        print("TOTAL IDS UPDATED: -> "+str((info["num_updated"])))
        print("TOTAL IDS DELETED: -> "+str((info["num_deleted"])))
    
    def AddDocsToCollection(self,documents):
        self.persistent_client.heartbeat()

    
        #TODO Farli passare da indexer

        #ids_added = self.CLIENT.add_documents(documents) #automatic persist IS DONE HERE
        #print("TOTAL IDS ADDED: -> "+str(len(ids_added)))
        
    
        print("TOTAL COLLECTION: -->"+ str(self.CLIENT._collection.count()))
        pass
    
    def __storeUsersFile__ (self,users): #da spostare magari su INIT

        tmp_dir=self.storagePath + self.CLIENT._collection.name
        if(not os.path.exists(tmp_dir)):
            os.makedirs(tmp_dir)
            print("CREATED USER DIR:"+tmp_dir)
        
        with open(os.path.join(tmp_dir,"users.txt"), "w+") as output:
            output.write(str(users))


    def __storeMetadataFile__(self):
        collection = self.CLIENT.get()
        metadata_collection = collection["metadatas"]
        uniques_keys = set().union(*(d.keys() for d in metadata_collection)).difference(self.remove_keys)
        tmp_dir=self.storagePath + self.CLIENT._collection.name
        if(not os.path.exists(tmp_dir)):
            os.makedirs(tmp_dir)
            print("CREATED METADATA DIR:"+tmp_dir)


        with open(os.path.join(tmp_dir,"metadata.txt"), "w+") as output:
            output.write(str(uniques_keys))
