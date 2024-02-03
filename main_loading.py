#from langchain.vectorstores import Chroma
from essai.bin.modules.LoaderEmbeddings import InitChromaDocsFromPath
import os
from essai.bin.transformersCustom.ConvertAndFormatDataset import process_directory
from datetime import datetime
from essai.config import settings
from essai.bin.modules.ChromaSingle import ChromaClass
from tqdm import tqdm
#SingleTon Chroma cross interface



#Populate croma collections
"""

https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever#retrieving-full-documents


"""


if __name__ == '__main__':


    settings.init()

    collection_name_local = settings.collection_name

    LOAD_DOCS = 1
    PREPROCESS = 1
    # Getting the current date and time
    dt = datetime.now()

    # getting the timestamp
    ts = datetime.timestamp(dt)

    source_directory = settings.source_directory+settings.init_dataset

    basenameDataset = source_directory.split("/")[-1]
    print(basenameDataset)
    destination_directory = "essai/dataset_transformed/"+basenameDataset+"_"+str(ts)
    custom_headers = ["content", "user","category","created_at_year","created_at_month","created_at_day"]

    settings.init()
    persist_directory = settings.persist_directory+"init_dataset"+"/"
    embed_model = settings.embed_model
    collection_name_local = settings.collection_name

    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

    if(PREPROCESS==1):
        if os.path.isdir(source_directory):
            #convert folder of txt files into csv
            process_directory(source_directory, destination_directory,custom_headers)
        else:
            print("DIR DOES NOT EXIST")
            exit()

    #FIRST LOAD OF DOCUMENTS (csv documents separated by PIPE)
    split_docs_chunked = InitChromaDocsFromPath(destination_directory) #deve essere transformed

    embed_model = settings.embed_model

    vectordb = None
    if LOAD_DOCS==1:
        #os.remove(settings.persist_directory+basenameDataset+"/") capire con chroma
        for split_docs_chunk in tqdm(split_docs_chunked):
            vectordb =  ChromaDB.CLIENT.from_documents(
                documents=split_docs_chunk,
                embedding=embed_model,
                collection_name = collection_name_local,
                persist_directory=settings.persist_directory+basenameDataset+"/" #settings
            )

            
            
        vectordb.persist()  
        collection = vectordb.get()

        print("COLLECTION COUNT: "+str(len(collection["ids"]))) #ids, metadata, documents
        print("DONE CHROMA PERSIST")
        print(vectordb._collection.name)
        print(vectordb._collection.id)

        users = ChromaDB.GetListOfUsers()

        