
from pydoc import doc
from keywords_suggester.bin.modules.LoaderEmbeddings import ProcessChunksFromLocal
from keywords_suggester.config import settings
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass



if __name__ == '__main__':

    settings.init()
    persist_directory = settings.persist_directory+"init_dataset_small"+"/"
    embed_model = settings.embed_model

    collection_name_local = "TestCollection"
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)


    users_collection = ChromaDB.GetListOfUsers()

    docs = ChromaDB.CLIENT.get(where={"user": "52254"},limit=5,include=["metadatas","documents"])
    metadata = docs["metadatas"]
    content = docs["documents"]
    for(meta, cont) in zip(metadata, content):
        print(meta,cont)
            
