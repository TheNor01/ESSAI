from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
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
    
    #https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore
    retriever_db = ChromaDB.CLIENT.as_retriever() #vedere parametri #qui posso passare un filter per fare un pre selezione
    
    
    retrieved_docs = retriever_db.invoke("What cars are beatiful?")
    print(retrieved_docs[0].page_content)
    
    
    print(retriever_db.get_relevant_documents("hello world"))
    
    
    #WHERE FILTER, in order to retrieve documents passing metadatas
    
    #Where filters only search embeddings where the key exists. If you search collection.get(where={"version": {"$ne": 1}}). 
    #Metadata that does not have the key version will not be returned.
    # where: A Where type dict used to filter results by. E.g. `{"$and": ["color" : "red", "price": {"$gte": 4.20}]}`. Optional.
    # where_document: A WhereDocument type dict used to filter by the documents. E.g. `{$contains: {"text": "hello"}}`. Optional.
    
    #https://docs.trychroma.com/usage-guide 
    # metadatas are into storage
    #ChromaDB.CLIENT.get(where={"status": "read"}, where_document={"$contains": "affairs"})
    #ChromaDB.CLIENT.get(where_document={"$or": [{"$contains": "global affairs"}, {"$contains": "domestic policy"}]})
    #ChromaDB.CLIENT.get(where={"$or": [{"author": "john"}, {"author": "jack"}]})
    #ChromaDB.CLIENT.get(where={"$and": [{"category": "chroma"}, {"author": "john"}]})
    #ChromaDB.CLIENT.get(where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
    #ChromaDB.CLIENT.get(where_document={"$contains": "Article"},where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
    #ChromaDB.CLIENT.get(where={'author': {'$in': ['john', 'jill']}})


    # SIMPLER SECTION
    #1 type of retriver, VectorStores
    #similarity, mmr, similarity treshold


    #2 Assembler

    print("ASSEMBLE TEST")

    doc_list = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    ]
    
    # initialize the bm25 retriever and faiss retriever oppure https://python.langchain.com/docs/integrations/retrievers/tf_idf
    bm25_retriever = BM25Retriever.from_texts(doc_list)
    bm25_retriever.k = 2
    
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_db], weights=[0.5, 0.5])
    
    docs = ensemble_retriever.get_relevant_documents("apples")

    print(docs)
    

    print("PARENT TEST")
    #PARENT 

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=ChromaDB.CLIENT,
        docstore=store,
        child_splitter=child_splitter,
    )

    sub_docs = ChromaDB.CLIENT.similarity_search("food")
    print(sub_docs[0])

    exit()


    #LLM SECTION
    
    #https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
    
    #https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
    
    #self query use llm https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query