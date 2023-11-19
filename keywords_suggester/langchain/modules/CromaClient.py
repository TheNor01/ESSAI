#https://docs.trychroma.com/reference/Client

"""
import chromadb




class CromaClient():
    def __init__(self): 
        self.client = chromadb.PersistentClient(path="resources/")
    
    def CreateCollection(self,collectionName):
        collection = self.client.create_collection(collectionName)

    def GetCollection(self,collectionName):
        collection = self.client.get_collection(collectionName)
        return collection
    
    def ResetCroma(self):
        print("RESETTING")
        self.client.reset()
"""