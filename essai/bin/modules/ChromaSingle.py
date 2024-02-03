from chromadb.config import Settings
import chromadb
from langchain.vectorstores import Chroma
import os
from essai.bin.modules.Indexer import Indexer
import pandas as pd
import plotly.express as px

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
        self.storagePath = "essai/storage" + "/"
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
        source_list=[]
        for element in metadata_collection:
             user_list.append(element["user"])
             category_list.append(element["category"])
             try:
                source_list.append(element["source"])
             except:
                source_list.append("None")


        hist_df = pd.DataFrame(list(zip(user_list, category_list,source_list)),columns=['user','category','source'])
        #print(hist_df)

        #target_user = '95d12'
        # Query the DataFrame for information related to the target user
        if(target_user):
            hist_df = hist_df[hist_df['user'] == target_user]

        inner_title=None
        if(target_user==None):
            inner_title = "Topic Distribution By Population"
        else:
            inner_title = "Topic Distribution Of: "+target_user


        #print(hist_df)

        threshold_percentage = 5.0
        category_counts = hist_df['category'].value_counts()

        hist_df['counts'] = hist_df.groupby(['category','source'])['user'].transform('count')

        result = hist_df[["category","source","counts"]].drop_duplicates()
        result['counts_pct'] = result.counts / result.counts.sum()

        print(result)

        result.loc[result['counts'] < 15, 'category'] = 'Other category' # Represent only large countries
        result = result.rename(columns={'counts': 'users'})
        fig = px.pie(result, values='users', names='category', title=inner_title)
        fig.show()

        return

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

    


        ids_added = self.CLIENT.add_documents(documents) #automatic persist IS DONE HERE
        print("TOTAL IDS ADDED: -> "+str(len(ids_added)))
        print("TOTAL COLLECTION: -->"+ str(self.CLIENT._collection.count()))
        pass
    
    def __storeUsersFile__ (self,users): #da spostare magari su INIT

        print("STORING USERS TO FILE")
        tmp_dir=self.storagePath + self.CLIENT._collection.name
        if(not os.path.exists(tmp_dir)):
            os.makedirs(tmp_dir)
            print("CREATED USER DIR:"+tmp_dir)
        
        with open(os.path.join(tmp_dir,"users.txt"), "w+") as output:
            output.write(str(users))
            
            print("USERS STORED")


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
