from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.RetriverSingle import RetrieverSingle
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

#https://www.reddit.com/r/LangChain/comments/18i9hh8/better_search_with_chroma_vector_store/
#https://www.youtube.com/watch?v=Uh9bYiVrW_s


if __name__ == '__main__':
    
    BERT_NAME = "test1"
    BERT = BertTopicClass(BERT_NAME,restore=1)


    settings.init()
    persist_directory = settings.persist_directory+"init_dataset_small"+"/"
    embed_model = settings.embed_model
    collection_name_local = "TestCollection"
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)
    Retriever = RetrieverSingle(ChromaDB)
    
    #https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore
    retriever_db = ChromaDB.CLIENT.as_retriever() #vedere parametri #qui posso passare un filter per fare un pre selezione
    
    
    retrieved_docs = retriever_db.invoke("What cars are beatiful?")
    




    exit()


   