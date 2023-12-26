
from langchain.retrievers import BM25Retriever, EnsembleRetriever,KNNRetriever,TFIDFRetriever
import os
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance



@singleton
class RetrieverSingle():
    def __init__(self,ChromaSingle):
        self.chroma = ChromaSingle

    def AssembleRetriever(self,queryToText,*args):

        docs = self.chroma.GetListOfDocs()
        if(len(docs)==0):
            print("Empty list documents")
            exit()
        if(args[0]=="BM25"):
            _retriever = BM25Retriever.from_texts(docs)
        elif(args[0]=="KNN"):
            _retriever = KNNRetriever.from_texts(docs) #serve embeddings
        elif(args[0]=="TFF"):
            _retriever = TFIDFRetriever.from_texts(docs)
        else:
            print("NO SUPPORTED ASSEMBLER")
            exit()

        ensemble_retriever = EnsembleRetriever(retrievers=[_retriever, self.chroma.CLIENT.as_retriever()], weights=[0.5, 0.5])
        docs = ensemble_retriever.get_relevant_documents(queryToText)

        return docs


    
    #WHERE FILTER, in order to retrieve documents passing metadatas
    
    #Where filters only search embeddings where the key exists. If you search collection.get(where={"version": {"$ne": 1}}). 
    #Metadata that does not have the key version will not be returned.
    # where: A Where type dict used to filter results by. E.g. `{"$and": ["color" : "red", "price": {"$gte": 4.20}]}`. Optional.
    # where_document: A WhereDocument type dict used to filter by the documents. E.g. `{$contains: {"text": "hello"}}`. Optional.
    
    def ComposedQuery(self,dictQuery):
        #ChromaDB.CLIENT.get(where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
        #ChromaDB.CLIENT.get(where={"status": "read"}, where_document={"$contains": "affairs"})
        #ChromaDB.CLIENT.get(where_document={"$or": [{"$contains": "global affairs"}, {"$contains": "domestic policy"}]})
        #ChromaDB.CLIENT.get(where={"$or": [{"author": "john"}, {"author": "jack"}]})
        #ChromaDB.CLIENT.get(where={"$and": [{"category": "chroma"}, {"author": "john"}]})
        #ChromaDB.CLIENT.get(where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
        #ChromaDB.CLIENT.get(where_document={"$contains": "Article"},where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
        #ChromaDB.CLIENT.get(where={'author': {'$in': ['john', 'jill']}})
        docs = self.chroma.CLIENT.get(where=dictQuery)
        return docs


    def BuildRetrieverDb(self,text,*args):

        retriever_db = self.chroma.as_retriever()
        _search_type = None
        _k_limit = None
        _score_threshold = None

        if(args > 0):
            _search_type = args[0]
            _k_limit = args[1]
            _score_threshold = args[2]

            if(_search_type=="score_threshold"):
                retriever_db = self.chroma.as_retriever(search_type=_search_type,search_kwargs={"k": _k_limit,"score_threshold": _score_threshold})
            elif(_search_type=="mmr"):
                #retriever_db = self.chroma.as_retriever(search_type="_search_type")
                retriever_db = self.chroma.as_retriever(search_type=_search_type,search_kwargs={"k": _k_limit})

            print("USING RETRIEVER DB")
            print("SEARCH TYPE = "+_search_type)
            print("K LIMIT = "+_k_limit)
            print("SCORE threshold = "+_score_threshold)
        else:
            print("DEFAULT RETRIEVER")


        docs = retriever_db.get_relevant_documents(text)

        return docs