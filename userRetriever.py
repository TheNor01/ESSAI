
from pydoc import doc
from essai.config import settings
from essai.bin.modules.ChromaSingle import ChromaClass



if __name__ == '__main__':

    settings.init()
    persist_directory = settings.persist_directory+settings.init_dataset+"/"
    embed_model = settings.embed_model

    collection_name_local = settings.collection_name
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)


    users_collection = ChromaDB.GetListOfUsers()
    
    print("GETTING user")
    docs = ChromaDB.CLIENT.get(where={"user": "f4225"},limit=5,include=["metadatas","documents"])
    
    print(docs)
    if(len(docs["documents"])==0):
        print("NO DOCS")
        exit()
    
    metadata = docs["metadatas"]
    content = docs["documents"]
    for(meta, cont) in zip(metadata, content):
        print(meta,cont)
            
