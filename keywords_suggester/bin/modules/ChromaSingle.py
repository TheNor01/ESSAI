from chromadb.config import Settings
import chromadb
from langchain.schema import Document
from langchain.vectorstores import Chroma
import os
from keywords_suggester.bin.modules.Indexer import Indexer

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
        

    def GetListOfUsers(self):
        collection = self.CLIENT.get()
        metadata_collection = collection["metadatas"]
        users = set([ element['user'] for element in metadata_collection ])
        self.__storeUsersFile__(users)
        return users
    
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
