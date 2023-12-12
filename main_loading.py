from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from keywords_suggester.bin.modules.LoaderEmbeddings import InitChromaDocsFromPath
import os
from keywords_suggester.bin.transformersCustom.ConvertAndFormatDataset import process_directory
import chromadb
from datetime import datetime
from keywords_suggester.config import settings

from langchain.indexes import SQLRecordManager, index
from chromadb.config import Settings
#SingleTon Chroma cross interface



#Populate croma collections
"""

https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever#retrieving-full-documents


"""


if __name__ == '__main__':


    settings.init()

    collection_name_local = "TestCollection"

    LOAD_DOCS = 1
    # Getting the current date and time
    dt = datetime.now()

    # getting the timestamp
    ts = datetime.timestamp(dt)

    source_directory = settings.source_directory+"init_dataset_small"

    basenameDataset = source_directory.split("/")[-1]
    print(basenameDataset)
    destination_directory = "keywords_suggester/dataset_transformed/"+basenameDataset+"_"+str(ts)
    custom_headers = ["content", "user","category","created_at"]

    if os.path.isdir(source_directory):
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
        for split_docs_chunk in split_docs_chunked:
            vectordb = Chroma.from_documents(
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


        